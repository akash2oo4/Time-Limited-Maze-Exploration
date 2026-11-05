"""
snapshot_dataset.py
Phase 4 (revised): training data for the exploration heuristic.

WHY THIS REPLACES THE COMPLETE-MAZE DATASET
-------------------------------------------
`dataset_generator.py` labels time-to-goal on FULLY KNOWN mazes. That was the right dataset
when the CNN's job was the fast run. It is the wrong dataset now.

The oracle bound established where the actual headroom is: classical policies reach optimal
having mapped ~44 % of the maze, while perfect decisions need only ~26 %, and under a budget
the classical gap runs 0.07-0.70 s with every 95 % CI clear of zero. The CNN's job is
therefore to guide EXPLORATION, and an exploring mouse never sees a complete maze -- it sees
a partial belief map with a specific, structured pattern of unknowns produced by a
two-boundary forward sensor. Train on complete mazes and deploy on partial ones and the
network is being asked to generalise across a distribution shift nobody measured.

So the inputs here are belief-map snapshots captured mid-exploration, and the labels are
true time-to-goal computed by the oracle on the complete maze behind them.

WHAT EACH SAMPLE HOLDS
----------------------
    belief_walls    [17,16] + [16,17] three-state edges, packed  -> 9 CNN planes
    time_cost       [4,256] float32 true time-to-goal per (heading, cell)
    on_route        [256]   bool: is this cell on the TRUE optimal route?
    reachable       [4,256] bool
    step_frac       float: how far into the search this snapshot was taken

`on_route` is the extra label that matters. Predicting time-to-goal teaches the network
where the goal is; predicting route membership teaches it which cells are worth SENSING,
which is precisely the decision the classical policies get wrong. Train both heads.

USAGE
    python3 snapshot_dataset.py                # 500 mazes -> data_snapshots/
    python3 snapshot_dataset.py 5000 8         # 5000 mazes, 8 workers
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass, asdict

import numpy as np

from maze import Maze, DIR_DELTA
from belief import BeliefMap, sense, fresh_belief, UNKNOWN, OPEN, WALL, NORTH
from dataset_generator import generate_maze, T_CELL, turn_cost, UNREACHABLE, NUM_HEADINGS
from solver import truth_passable, belief_passable, time_to_goal, extract_path
from episode import (EpisodeConfig, FloodFillPolicy, OptimisticShortcutPolicy,
                     _step, make_path_verified_stop_rule)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SnapshotConfig:
    n_mazes: int = 500
    snapshots_per_maze: int = 6
    seed: int = 20260825
    out_dir: str = "data_snapshots"
    shard_size: int = 500
    split: tuple = (0.70, 0.15, 0.15)     # BY MAZE, never by snapshot

    # Snapshots are taken along a real exploration trajectory. Which policy drives it
    # matters: the network must see the belief states it will actually be asked about.
    # Both classical policies are used so the data is not tied to one heuristic's habits.
    max_search_steps: int = 400

    # Sorties, not a single outbound dash. Competition rules require the mouse to return
    # to the start before each fast run, and strong teams deliberately route home through
    # unmapped cells rather than retracing -- the return trip is free exploration on a
    # journey that has to be made anyway. Training only on outbound belief states while
    # deploying across both legs is a distribution mismatch: return-leg beliefs are far
    # more complete, and the mouse is travelling the other way.
    max_sorties: int = 3

    # True reproduces the original dataset: one outbound dash, start -> goal, snapshots
    # from that leg only. Kept as an option because the outbound-only model tested
    # slightly better than the sortie-trained one, so it is the paper's main model.
    outbound_only: bool = False


# ---------------------------------------------------------------------------
# Snapshot capture
# ---------------------------------------------------------------------------

def true_route_mask(maze: Maze) -> np.ndarray:
    """
    Cells on the genuinely optimal start-to-goal route.

    This is the supervision signal the classical policies lack. They decide where to sense
    using distance-to-goal heuristics; the oracle result showed that spends ~40 % of the
    budget on cells that cannot affect the answer. A network that can predict route
    membership from a partial belief map can spend its sensing where it counts.
    """
    p = truth_passable(maze)
    cost = time_to_goal(p, sorted(maze.goal_indices))
    _, path = extract_path(cost, p, (maze.start_x, maze.start_y), NORTH,
                           sorted(maze.goal_indices))
    mask = np.zeros(maze.total_cells, dtype=bool)
    for x, y, _ in path:
        mask[y * maze.width + x] = True
    return mask


def pack_belief(belief: BeliefMap) -> np.ndarray:
    """
    Compact three-state edge storage: h_edges [17,16] and v_edges [16,17] flattened.

    Stored packed rather than as expanded CNN planes. Expansion is two lines at load time
    (`BeliefMap.to_channels`) and costs nothing; storing 9 float planes per snapshot would
    inflate the dataset roughly 40x.
    """
    return np.concatenate([belief.h_edges.ravel(), belief.v_edges.ravel()]).astype(np.uint8)


def unpack_belief(packed: np.ndarray, width: int = 16, height: int = 16) -> BeliefMap:
    b = BeliefMap(width, height)
    n_h = (height + 1) * width
    b.h_edges = packed[:n_h].reshape(height + 1, width).copy()
    b.v_edges = packed[n_h:n_h + height * (width + 1)].reshape(height, width + 1).copy()
    return b


def capture_trajectory(maze: Maze, policy, cfg: SnapshotConfig):
    """
    Drive a full search run -- out to the goal, back to the start, repeat -- snapshotting
    the belief map at evenly spaced points across the WHOLE run.

    Two things this gets right that a single outbound dash does not:

      * The return leg is real exploration. Competition mice must return to the start
        before each fast run, and strong teams route home through unmapped cells rather
        than retracing, hunting for long straights or lower-turn alternatives. Those
        belief states are a large fraction of what the deployed policy actually sees.
      * Later sorties begin from a much richer belief. A network trained only on sparse
        outbound maps is being asked to generalise to a regime it never saw.

    Snapshots are spread across the whole run rather than concentrated at the end, because
    the network has to decide well EARLY, while the budget still has room to be spent.
    """
    belief, heading = fresh_belief(maze)
    policy.reset(belief)
    x, y = maze.start_x, maze.start_y
    goals = sorted(maze.goal_indices)
    start = [maze.start_index]
    stop_rule = make_path_verified_stop_rule(0)

    trail = []
    step = 0
    reached_goal = False

    legs = ((goals, 0),) if cfg.outbound_only else ((goals, 0), (start, 1))
    for _ in range(1 if cfg.outbound_only else cfg.max_sorties):
        for targets, leg in legs:
            target_set = set(targets)
            while step < cfg.max_search_steps:
                if (y * belief.width + x) in target_set:
                    break
                h2 = policy.choose(belief, x, y, heading, targets)
                if h2 is None:
                    break
                x, y, heading, _ = _step(maze, belief, x, y, heading, h2)
                step += 1
                trail.append((step, pack_belief(belief), x, y, heading, leg))
            if leg == 0 and (y * belief.width + x) in maze.goal_indices:
                reached_goal = True
        if step >= cfg.max_search_steps or stop_rule(belief, maze, reached_goal):
            break

    if len(trail) < 2:
        return []

    idx = np.linspace(1, len(trail) - 1, cfg.snapshots_per_maze).astype(int)
    return [trail[i] for i in sorted(set(idx.tolist()))]


def build_maze_samples(args):
    """One maze -> several snapshots. Top-level so multiprocessing can pickle it."""
    seed, cfg_dict = args
    cfg = SnapshotConfig(**cfg_dict)
    random.seed(seed)
    maze = generate_maze()

    labels = time_to_goal(truth_passable(maze), sorted(maze.goal_indices))
    route = true_route_mask(maze)
    walls = np.asarray(maze.walls, dtype=np.uint8)

    policy = FloodFillPolicy() if seed % 2 == 0 else OptimisticShortcutPolicy()
    snaps = capture_trajectory(maze, policy, cfg)
    if not snaps:
        return None

    n = len(snaps)
    return {
        "belief": np.stack([s[1] for s in snaps]),
        "pose": np.array([[s[2], s[3], s[4]] for s in snaps], dtype=np.int16),
        "step_frac": np.array([s[0] / max(1, snaps[-1][0]) for s in snaps], dtype=np.float32),
        # 0 = outbound leg, 1 = return leg. Not a network input -- route membership does
        # not depend on which way the mouse is travelling -- but kept so the balance of
        # training data across legs can be reported and checked.
        "leg": np.array([s[5] for s in snaps], dtype=np.int8),
        "time_cost": np.repeat(labels[None], n, axis=0),
        "on_route": np.repeat(route[None], n, axis=0),
        "walls": np.repeat(walls[None], n, axis=0),
        "maze_seed": np.full(n, seed, dtype=np.int32),
    }


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(cfg: SnapshotConfig = SnapshotConfig(), n_jobs: int = 1):
    os.makedirs(cfg.out_dir, exist_ok=True)
    t0 = time.time()

    jobs = [(cfg.seed + i, asdict(cfg)) for i in range(cfg.n_mazes)]
    if n_jobs > 1:
        import multiprocessing as mp
        with mp.Pool(n_jobs) as pool:
            out = pool.map(build_maze_samples, jobs, chunksize=4)
    else:
        out = [build_maze_samples(j) for j in jobs]
    out = [o for o in out if o is not None]

    # ---- split BY MAZE. Snapshots from one maze share its labels almost entirely, so
    # ---- splitting by snapshot would leak the answer across train and test.
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(out))
    n_tr = int(cfg.split[0] * len(order))
    n_va = int(cfg.split[1] * len(order))
    splits = {"train": order[:n_tr],
              "val": order[n_tr:n_tr + n_va],
              "test": order[n_tr + n_va:]}

    manifest = {"config": asdict(cfg),
                "n_mazes": len(out),
                "heading_order": ["N", "E", "S", "W"],
                "unreachable_sentinel": UNREACHABLE,
                "t_cell_seconds": T_CELL,
                "shards": {}}

    label_stats = []
    for name, ids in splits.items():
        files, total = [], 0
        for s, lo in enumerate(range(0, len(ids), cfg.shard_size)):
            sel = ids[lo:lo + cfg.shard_size]
            merged = {k: np.concatenate([out[i][k] for i in sel]) for k in out[0]}
            total += len(merged["belief"])
            if name == "train":
                tc = merged["time_cost"]
                label_stats.append(tc[tc < UNREACHABLE])
            path = os.path.join(cfg.out_dir, f"{name}_{s:03d}.npz")
            np.savez_compressed(path, **merged,
                                reachable=(merged["time_cost"] < UNREACHABLE))
            files.append(os.path.basename(path))
        manifest["shards"][name] = files
        manifest[f"n_snapshots_{name}"] = total

    # Normalisation from TRAIN ONLY. Computing it over everything leaks test statistics.
    allv = np.concatenate([a.ravel() for a in label_stats])
    manifest["label_mean"] = float(allv.mean())
    manifest["label_std"] = float(allv.std())

    with open(os.path.join(cfg.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"{len(out)} mazes -> "
          f"{manifest['n_snapshots_train']}/{manifest['n_snapshots_val']}/"
          f"{manifest['n_snapshots_test']} snapshots (train/val/test) "
          f"in {time.time() - t0:.0f}s")
    return manifest


# ---------------------------------------------------------------------------
# Loader side
# ---------------------------------------------------------------------------

def to_channels(packed_belief: np.ndarray, pose: np.ndarray,
                width: int = 16, height: int = 16) -> np.ndarray:
    """
    Expand one packed snapshot into CNN input planes [11, 16, 16]:

        0-3   wall present   (N, E, S, W)
        4-7   wall UNKNOWN   (N, E, S, W)
        8     mouse position
        9     goal region
        10    heading, encoded as a constant plane

    Two planes per direction, not one three-valued plane: encoding {OPEN, UNKNOWN, WALL}
    as {0, 0.5, 1} would tell the convolution that "unknown" sits numerically between
    "open" and "wall", which is meaningless and something it will try to interpolate.
    """
    b = unpack_belief(packed_belief, width, height)
    out = np.zeros((11, height, width), dtype=np.float32)
    for h in range(4):
        for y in range(height):
            for x in range(width):
                s = b.get(x, y, h)
                if s == WALL:
                    out[h, y, x] = 1.0
                elif s == UNKNOWN:
                    out[4 + h, y, x] = 1.0
    mx, my, mh = int(pose[0]), int(pose[1]), int(pose[2])
    out[8, my, mx] = 1.0
    mid = width // 2
    out[9, mid - 1:mid + 1, mid - 1:mid + 1] = 1.0
    out[10, :, :] = mh / 3.0
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    build(SnapshotConfig(n_mazes=n), n_jobs=jobs)
