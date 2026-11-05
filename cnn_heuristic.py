"""
cnn_heuristic.py
Phase 5: the learned exploration heuristic.

WHAT IT IS ASKED TO DO
----------------------
The oracle bound established the target. Classical policies reach the optimal route having
mapped ~44 % of the maze; perfect decisions need ~26 %. Under a search-time budget the
classical gap runs 0.07 s (25 s budget) to 0.70 s (12 s budget), every 95 % CI clear of
zero. Roughly 40 % of classical sensing effort goes to cells that cannot affect the answer.

So the network's job is NOT to find the goal -- flood-fill already does that. Its job is to
predict WHICH CELLS ARE WORTH SENSING from a partial belief map.

TWO HEADS
---------
    time head   [4,16,16]  time-to-goal per (heading, cell)   -> a better search heuristic
    route head  [1,16,16]  P(cell lies on the true optimal route) -> where to spend budget

The route head is the one that targets the measured failure. The time head alone would
just reproduce distance-to-goal, which is what the classical policies already use.

ARCHITECTURE NOTE
-----------------
Dilated convolutions, not a deep plain stack. A wall ten cells away can force a long detour,
so the receptive field has to span the grid; a shallow plain CNN physically cannot represent
that dependency and will look like it "failed to learn" when it was never able to.

USAGE
    python3 cnn_heuristic.py train                 # train on data_snapshots/
    python3 cnn_heuristic.py eval                  # benchmark the policy vs classical+oracle
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from maze import DIR_DELTA
from belief import BeliefMap, UNKNOWN, WALL, NORTH
from dataset_generator import T_CELL, turn_cost, UNREACHABLE, NUM_HEADINGS
from solver import belief_passable, time_to_goal
from episode import ExplorationPolicy
from snapshot_dataset import unpack_belief

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def belief_planes(packed: np.ndarray, pose: np.ndarray, width=16, height=16) -> np.ndarray:
    """
    Packed belief -> [11,16,16] input planes. Vectorised; the per-cell version in
    snapshot_dataset is fine for inspection but far too slow inside a training loop.

        0-3   wall present (N,E,S,W)
        4-7   wall UNKNOWN (N,E,S,W)
        8     mouse position
        9     goal region
        10    heading as a constant plane

    Two planes per direction rather than one three-valued plane: encoding
    {OPEN, UNKNOWN, WALL} as {0, 0.5, 1} would assert that "unknown" lies numerically
    between "open" and "wall", which is meaningless and something a convolution will try
    to interpolate across.
    """
    n_h = (height + 1) * width
    h_e = packed[:n_h].reshape(height + 1, width)
    v_e = packed[n_h:n_h + height * (width + 1)].reshape(height, width + 1)

    # heading order N,E,S,W -> (array slice for that side of each cell)
    sides = [h_e[:height, :], v_e[:, 1:], h_e[1:, :], v_e[:, :width]]

    out = np.zeros((11, height, width), dtype=np.float32)
    for h, s in enumerate(sides):
        out[h] = (s == WALL)
        out[4 + h] = (s == UNKNOWN)
    out[8, int(pose[1]), int(pose[0])] = 1.0
    mid = width // 2
    out[9, mid - 1:mid + 1, mid - 1:mid + 1] = 1.0
    out[10, :, :] = int(pose[2]) / 3.0
    return out


class SnapshotDataset(Dataset):
    def __init__(self, out_dir: str, split: str):
        man = json.load(open(os.path.join(out_dir, "manifest.json")))
        self.mean = man["label_mean"]
        self.std = man["label_std"]
        packs = [np.load(os.path.join(out_dir, f)) for f in man["shards"][split]]
        self.belief = np.concatenate([p["belief"] for p in packs])
        self.pose = np.concatenate([p["pose"] for p in packs])
        self.time = np.concatenate([p["time_cost"] for p in packs])
        self.route = np.concatenate([p["on_route"] for p in packs])
        self.reach = np.concatenate([p["reachable"] for p in packs])
        self.frac = np.concatenate([p["step_frac"] for p in packs])

    def __len__(self):
        return len(self.belief)

    def __getitem__(self, i):
        x = belief_planes(self.belief[i], self.pose[i])
        t = (self.time[i].reshape(4, 16, 16) - self.mean) / self.std
        return (torch.from_numpy(x),
                torch.from_numpy(t.astype(np.float32)),
                torch.from_numpy(self.route[i].reshape(16, 16).astype(np.float32)),
                torch.from_numpy(self.reach[i].reshape(4, 16, 16)),
                float(self.frac[i]))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DilatedBlock(nn.Module):
    def __init__(self, c, dilation):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=dilation, dilation=dilation)
        self.norm = nn.GroupNorm(8, c)

    def forward(self, x):
        return F.relu(x + self.norm(self.conv(x)))


class MazeNet(nn.Module):
    """
    Dilations 1,2,4,8,1,2 give a receptive field wider than 16 cells, so a wall anywhere on
    the grid can influence any prediction. GroupNorm rather than BatchNorm: batches here are
    highly correlated (six snapshots from the same maze), which is exactly the regime where
    BatchNorm's batch statistics mislead.
    """

    def __init__(self, in_ch=11, width=64):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, width, 3, padding=1),
                                  nn.GroupNorm(8, width), nn.ReLU())
        self.blocks = nn.Sequential(*[DilatedBlock(width, d) for d in (1, 2, 4, 8, 1, 2)])
        self.time_head = nn.Conv2d(width, 4, 1)
        self.route_head = nn.Conv2d(width, 1, 1)

    def forward(self, x):
        z = self.blocks(self.stem(x))
        return self.time_head(z), self.route_head(z).squeeze(1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    data_dir: str = "data_snapshots"
    out: str = "cnn_heuristic.pt"
    epochs: int = 30
    batch: int = 64
    lr: float = 2e-3
    route_weight: float = 1.0
    overestimate_penalty: float = 3.0     # asymmetric: overestimating breaks admissibility
    seed: int = 0                         # weight init + batch order; vary it for repeat runs


def masked_asymmetric_mse(pred, target, mask, over_w):
    """
    MSE, but overestimation is punished harder.

    An A* heuristic that overestimates can return a suboptimal path; one that
    underestimates only searches more. The asymmetry pushes the network toward the
    admissible side without claiming a guarantee it cannot have.
    """
    err = pred - target
    w = torch.where(err > 0, over_w, 1.0)
    loss = (w * err ** 2)[mask]
    return loss.mean() if loss.numel() else pred.sum() * 0.0


def train(cfg: TrainConfig = TrainConfig()):
    import random
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    tr = SnapshotDataset(cfg.data_dir, "train")
    va = SnapshotDataset(cfg.data_dir, "val")
    print(f"train {len(tr)} / val {len(va)} snapshots, device {DEVICE}")

    dl_tr = DataLoader(tr, batch_size=cfg.batch, shuffle=True, num_workers=2)
    dl_va = DataLoader(va, batch_size=cfg.batch, num_workers=2)

    net = MazeNet().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)

    # Route cells are a small minority of the grid (~25 of 256), so an unweighted BCE
    # collapses to predicting "not on route" everywhere.
    pos_weight = torch.tensor([(256 - 25) / 25.0], device=DEVICE)

    best = float("inf")
    for ep in range(cfg.epochs):
        net.train()
        for x, t, r, m, _ in dl_tr:
            x, t, r, m = x.to(DEVICE), t.to(DEVICE), r.to(DEVICE), m.to(DEVICE)
            pt, pr = net(x)
            loss = (masked_asymmetric_mse(pt, t, m, cfg.overestimate_penalty)
                    + cfg.route_weight * F.binary_cross_entropy_with_logits(
                        pr, r, pos_weight=pos_weight))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        net.eval()
        tl, rl, n = 0.0, 0.0, 0
        with torch.no_grad():
            for x, t, r, m, _ in dl_va:
                x, t, r, m = x.to(DEVICE), t.to(DEVICE), r.to(DEVICE), m.to(DEVICE)
                pt, pr = net(x)
                tl += masked_asymmetric_mse(pt, t, m, cfg.overestimate_penalty).item() * len(x)
                rl += F.binary_cross_entropy_with_logits(
                    pr, r, pos_weight=pos_weight).item() * len(x)
                n += len(x)
        tl, rl = tl / n, rl / n
        print(f"  epoch {ep + 1:3d}  val time {tl:.4f}  val route {rl:.4f}")
        if tl + rl < best:
            best = tl + rl
            torch.save({"state": net.state_dict(),
                        "mean": tr.mean, "std": tr.std, "cfg": asdict(cfg)}, cfg.out)
    print(f"saved {cfg.out} (best val {best:.4f})")


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class CNNExplorationPolicy(ExplorationPolicy):
    """
    Learned exploration. Reads only the belief map -- same contract as every other policy.

    The route head is what changes behaviour: moves entering a cell the network believes
    lies on the true optimal route get a discount, so budget is spent verifying cells that
    matter instead of surveying the maze. That is precisely the classical failure the
    oracle bound exposed.

    Inference is once per step, cached on belief change. On the real STM32F103 this network
    would not run inside an A* loop -- but it does not have to: one forward pass per cell
    entry at ~250 cells per run is well within budget once quantised.
    """

    name = "cnn_learned"

    def __init__(self, ckpt="cnn_heuristic.pt", route_bonus=0.06):
        if route_bonus >= T_CELL:
            raise ValueError("route_bonus must be < T_CELL to keep edge costs non-negative")
        blob = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        self.net = MazeNet().to(DEVICE)
        self.net.load_state_dict(blob["state"])
        self.net.eval()
        self.mean, self.std = blob["mean"], blob["std"]
        self.route_bonus = route_bonus
        self._key = None
        self._route = None

    def reset(self, belief):
        self._key = None

    def _route_prob(self, belief, x, y, heading):
        key = (belief.unknown_count(), int(belief.visited.sum()), x, y, heading)
        if key != self._key:
            packed = np.concatenate([belief.h_edges.ravel(), belief.v_edges.ravel()])
            planes = belief_planes(packed, np.array([x, y, heading]))
            with torch.no_grad():
                _, pr = self.net(torch.from_numpy(planes)[None].to(DEVICE))
            self._route = torch.sigmoid(pr)[0].cpu().numpy()
            self._key = key
        return self._route

    def choose(self, belief, x, y, heading, targets):
        route = self._route_prob(belief, x, y, heading)
        p = belief_passable(belief, optimistic=True)
        field = time_to_goal(p, targets, belief.width, belief.height)

        best = None
        for h in range(NUM_HEADINGS):
            if not belief.is_passable(x, y, h, optimistic=True):
                continue
            d = DIR_DELTA[h]
            nx, ny = x + d["x"], y + d["y"]
            if not belief.in_bounds(nx, ny):
                continue
            c = float(field[h, ny * belief.width + nx])
            if c >= UNREACHABLE:
                continue
            step = T_CELL
            if not belief.visited[ny, nx]:
                step -= self.route_bonus * float(route[ny, nx])
            s = turn_cost(heading, h) + step + c
            if best is None or s < best[0] - 1e-12:
                best = (s, h)
        return None if best is None else best[1]


# ---------------------------------------------------------------------------

def evaluate(n_mazes=100, ckpt="cnn_heuristic.pt"):
    """Same mazes, same budgets, classical vs learned vs oracle bound."""
    import random
    from dataset_generator import generate_maze
    from episode import EpisodeConfig, FloodFillPolicy, OptimisticShortcutPolicy, run_episode
    from oracle_bound import OracleGuidedPolicy

    NO_ROUTE_ = 1e5

    budgets = [None, 22, 20, 18, 15, 12]
    print(f"\n{'budget':>8}  {'policy':<24}{'mean-gap':>10}{'median':>9}{'p90':>8}"
          f"{'known':>7}{'no-route':>10}")
    for b in budgets:
        for factory in (FloodFillPolicy, OptimisticShortcutPolicy,
                        lambda: CNNExplorationPolicy(ckpt), OracleGuidedPolicy):
            gaps, known, fail = [], [], 0
            for i in range(n_mazes):
                random.seed(i)
                m = generate_maze()
                pol = factory()
                if isinstance(pol, OracleGuidedPolicy):
                    pol.bind(m)
                r = run_episode(m.clone(), pol, EpisodeConfig(search_time_budget=b))
                known.append(r.known_fraction)
                if r.fast_run_time >= NO_ROUTE_:
                    fail += 1
                    gaps.append(5.0)
                else:
                    gaps.append(r.exploration_gap)
            g = np.array(gaps)
            print(f"{str(b):>8}  {pol.name:<24}{g.mean():>10.3f}{np.median(g):>9.3f}"
                  f"{np.percentile(g, 90):>8.3f}{np.mean(known):>7.2f}{fail:>10}")
        print()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        # python3 cnn_heuristic.py train [epochs] [--seed N] [--data DIR] [--out FILE]
        a = sys.argv[2:]
        def opt(flag, default):
            return a[a.index(flag) + 1] if flag in a else default
        epochs = int(a[0]) if a and not a[0].startswith("--") else 30
        train(TrainConfig(epochs=epochs, seed=int(opt("--seed", 0)),
                          data_dir=opt("--data", "data_snapshots"),
                          out=opt("--out", "cnn_heuristic.pt")))
    else:
        evaluate(int(sys.argv[2]) if len(sys.argv) > 2 else 100)
