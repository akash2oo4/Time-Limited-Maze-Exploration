"""
finetune_official.py -- does a little real-maze data fix the distribution shift?

Two-fold cross-validation on the official mazes:
    * the 142 mazes are split into halves A and B, balanced by contest
    * a base model is fine-tuned on one half (mixed 50/50 with random-maze snapshots,
      so it does not forget them) and tested on the OTHER half, then the reverse
    * every official maze is therefore tested by a model that never saw it

Commands (run_finetune.sh calls these in order):
    python3 finetune_official.py folds    --dir mazefiles/classic
    python3 finetune_official.py data     --dir mazefiles/classic --fold A --out data_off_A
    python3 finetune_official.py finetune --base cnn_outbound_s0.pt --data data_off_A \
                                          --random data_outbound --out cnn_ft_s0_onA.pt [--seed 0]
"""

import argparse
import json
import os
import random
import sys
from dataclasses import asdict

import numpy as np

FOLD_SEED = 20261014


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------

def folds(maze_dir, seed=FOLD_SEED):
    """{'A': [...names], 'B': [...names]}, alternating within each contest after a shuffle."""
    from official_mazes import select
    by_contest = {}
    for name, label in select(maze_dir, verbose=False):
        by_contest.setdefault(label, []).append(name)
    rng = random.Random(seed)
    out = {"A": [], "B": []}
    for label in sorted(by_contest):
        names = sorted(by_contest[label])
        rng.shuffle(names)
        for i, n in enumerate(names):
            out["A" if i % 2 == 0 else "B"].append(n)
    return out


# ---------------------------------------------------------------------------
# Snapshot data from official mazes
# ---------------------------------------------------------------------------

def _maze_samples(args):
    """One official maze x one driver -> snapshot dict (same keys as snapshot_dataset)."""
    path, driver, snaps = args
    from official_mazes import load_maze_file
    from snapshot_dataset import SnapshotConfig, capture_trajectory, true_route_mask
    from solver import truth_passable, time_to_goal
    from episode import FloodFillPolicy, OptimisticShortcutPolicy

    maze = load_maze_file(path)
    cfg = SnapshotConfig(snapshots_per_maze=snaps, outbound_only=True)
    policy = FloodFillPolicy() if driver == 0 else OptimisticShortcutPolicy()
    s = capture_trajectory(maze, policy, cfg)
    if not s:
        return None
    labels = time_to_goal(truth_passable(maze), sorted(maze.goal_indices))
    route = true_route_mask(maze)
    n = len(s)
    return {
        "belief": np.stack([x[1] for x in s]),
        "pose": np.array([[x[2], x[3], x[4]] for x in s], dtype=np.int16),
        "step_frac": np.array([x[0] / max(1, s[-1][0]) for x in s], dtype=np.float32),
        "leg": np.array([x[5] for x in s], dtype=np.int8),
        "time_cost": np.repeat(labels[None], n, axis=0),
        "on_route": np.repeat(route[None], n, axis=0),
    }


def build_data(maze_dir, names, out_dir, snaps=20, val_frac=0.15, jobs=None):
    """
    Snapshots from both classical drivers on each maze. 15 % of the mazes are held out as
    a validation split, used only to pick the best fine-tuning epoch -- never the test half.
    """
    from multiprocessing import Pool
    from dataset_generator import UNREACHABLE
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(FOLD_SEED)
    names = sorted(names)
    rng.shuffle(names)
    n_val = max(1, int(round(val_frac * len(names))))
    split = {"val": names[:n_val], "train": names[n_val:]}

    man = {"shards": {}, "mazes": split, "snapshots_per_maze_per_driver": snaps}
    for part, ns in split.items():
        jobs_list = [(os.path.join(maze_dir, n), d, snaps) for n in ns for d in (0, 1)]
        with Pool(jobs or os.cpu_count()) as pool:
            out = [o for o in pool.map(_maze_samples, jobs_list) if o is not None]
        merged = {k: np.concatenate([o[k] for o in out]) for k in out[0]}
        f = f"{part}_000.npz"
        np.savez_compressed(os.path.join(out_dir, f), **merged,
                            reachable=(merged["time_cost"] < UNREACHABLE))
        man["shards"][part] = [f]
        man[f"n_snapshots_{part}"] = int(len(merged["belief"]))
    # label_mean/std are placeholders here: fine-tuning always uses the BASE model's
    # normalisation, so the two datasets are scaled identically.
    man["label_mean"], man["label_std"] = 0.0, 1.0
    json.dump(man, open(os.path.join(out_dir, "manifest.json"), "w"), indent=2)
    print(f"{out_dir}: {len(split['train'])} train / {len(split['val'])} val mazes -> "
          f"{man['n_snapshots_train']} / {man['n_snapshots_val']} snapshots")


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def finetune(base, data_dir, random_dir, out, epochs=10, lr=3e-4, batch=64, seed=0):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
    from cnn_heuristic import (MazeNet, SnapshotDataset, masked_asymmetric_mse,
                               TrainConfig, DEVICE)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    blob = torch.load(base, map_location=DEVICE, weights_only=False)
    mean, std = blob["mean"], blob["std"]

    off_tr = SnapshotDataset(data_dir, "train")
    off_va = SnapshotDataset(data_dir, "val")
    rnd_tr = SnapshotDataset(random_dir, "train")
    for d in (off_tr, off_va, rnd_tr):          # one normalisation for everything
        d.mean, d.std = mean, std

    # Each epoch: as many official snapshots as we have, plus the same number of random
    # ones drawn fresh, so the model keeps seeing random mazes and does not forget them.
    mix = ConcatDataset([off_tr, rnd_tr])
    w = np.concatenate([np.full(len(off_tr), 0.5 / len(off_tr)),
                        np.full(len(rnd_tr), 0.5 / len(rnd_tr))])
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                    num_samples=2 * len(off_tr), replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    dl_tr = DataLoader(mix, batch_size=batch, sampler=sampler, num_workers=2)
    dl_va = DataLoader(off_va, batch_size=batch, num_workers=2)

    net = MazeNet().to(DEVICE)
    net.load_state_dict(blob["state"])
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    pos_weight = torch.tensor([(256 - 25) / 25.0], device=DEVICE)
    over_w = TrainConfig().overestimate_penalty

    def val_loss():
        net.eval()
        tl = rl = n = 0.0
        with torch.no_grad():
            for x, t, r, m, _ in dl_va:
                x, t, r, m = x.to(DEVICE), t.to(DEVICE), r.to(DEVICE), m.to(DEVICE)
                pt, pr = net(x)
                tl += masked_asymmetric_mse(pt, t, m, over_w).item() * len(x)
                rl += F.binary_cross_entropy_with_logits(pr, r, pos_weight=pos_weight).item() * len(x)
                n += len(x)
        return tl / n, rl / n

    tl, rl = val_loss()
    print(f"train {len(off_tr)} official (+ same number random per epoch) / "
          f"val {len(off_va)} official snapshots, device {DEVICE}")
    print(f"  base model      val time {tl:.4f}  val route {rl:.4f}")
    best = tl + rl
    save = lambda: torch.save({"state": net.state_dict(), "mean": mean, "std": std,
                               "cfg": {"base": base, "finetune_data": data_dir,
                                       "epochs": epochs, "lr": lr, "seed": seed}}, out)
    save()                                       # epoch 0 = base, in case nothing improves
    for ep in range(epochs):
        net.train()
        for x, t, r, m, _ in dl_tr:
            x, t, r, m = x.to(DEVICE), t.to(DEVICE), r.to(DEVICE), m.to(DEVICE)
            pt, pr = net(x)
            loss = (masked_asymmetric_mse(pt, t, m, over_w)
                    + F.binary_cross_entropy_with_logits(pr, r, pos_weight=pos_weight))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        tl, rl = val_loss()
        print(f"  epoch {ep + 1:3d}       val time {tl:.4f}  val route {rl:.4f}")
        if tl + rl < best:
            best = tl + rl
            save()
    print(f"saved {out} (best official val {best:.4f})")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["folds", "data", "finetune"])
    ap.add_argument("--dir", default="mazefiles/classic")
    ap.add_argument("--fold", choices=["A", "B"])
    ap.add_argument("--out")
    ap.add_argument("--base")
    ap.add_argument("--data")
    ap.add_argument("--random", default="data_outbound")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snaps", type=int, default=20)
    a = ap.parse_args()

    if a.cmd == "folds":
        f = folds(a.dir)
        for k in "AB":
            print(f"fold {k}: {len(f[k])} mazes")
    elif a.cmd == "data":
        build_data(a.dir, folds(a.dir)[a.fold], a.out, snaps=a.snaps)
    else:
        finetune(a.base, a.data, a.random, a.out, epochs=a.epochs, seed=a.seed)


if __name__ == "__main__":
    main()
