"""
dataset_generator.py
Phase 4 dataset builder for the micromouse learned-heuristic study.

Produces sharded .npz files containing, for each maze:
  - walls        : the compact ground-truth wall bitmask (uint8, 256)
  - bfs_dist     : hop-count distance-to-goal per cell        (int16,   256)   [sanity reference]
  - time_cost    : turn-weighted time-to-goal per (heading,cell) (float32, 4,256) [CNN label]
  - reachable    : mask of cells with a finite time cost      (bool,    4,256)

Design notes
------------
* Inputs are stored as the raw 4-bit wall mask, NOT as expanded CNN channels.
  Expanding 256 bytes into 6x16x16 float planes is a two-line operation at load
  time and costs nothing; storing the expansion would inflate the dataset ~24x.
  See `walls_to_channels()` for the loader-side expansion.
* Labels live in the SAME file as the mazes they describe. See README note.
* Splitting is by maze, never by cell.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict

import numpy as np

from maze import (Maze, WALL_NORTH, WALL_EAST, WALL_SOUTH, WALL_WEST, DIR_DELTA,
                  CELL_SIZE_CM)
from maze_generator import IEEEMazeGenerator

# ---------------------------------------------------------------------------
# Kinematic cost model  (Phase 0 constants -- justify each in the paper)
# ---------------------------------------------------------------------------

CELL_M = CELL_SIZE_CM / 100.0        # 0.18 m centre-to-centre

# The TQD-Micromouse-JL is rated 5 m/s design top speed, but that is a straight-line
# sprint figure on a long run. A conservative sustained cell-to-cell speed is used
# here; the whole model is relative, so state the value and vary it in an ablation.
V_STRAIGHT = 2.0                     # m/s
T_CELL = CELL_M / V_STRAIGHT         # ~0.09 s per cell traversed

TURN_COST = {0: 0.00,                # straight on
             1: 0.15,                # 90 deg
             2: 0.35,                # 180 deg (stop and reverse)
             3: 0.15}                # 90 deg the other way

UNREACHABLE = 1.0e6                  # sentinel written to the label array

NUM_HEADINGS = 4                     # 0=N 1=E 2=S 3=W, matching DIR_DELTA


def turn_cost(h_from: int, h_to: int) -> float:
    return TURN_COST[(h_to - h_from) % 4]


# ---------------------------------------------------------------------------
# IEEE post rule -- patch for add_loops
# ---------------------------------------------------------------------------

def post_wall_count(maze: Maze, i: int, j: int) -> int:
    """How many wall segments touch lattice post (i, j)."""
    n = 0
    for kind, x, y in (('V', i, j - 1), ('V', i, j), ('H', i - 1, j), ('H', i, j)):
        if not (0 <= x < maze.width and 0 <= y < maze.height):
            continue
        n += maze.has_wall(x, y, WALL_NORTH if kind == 'H' else WALL_WEST)
    return n


def _segment_posts(x: int, y: int, wall_flag: int):
    """The two lattice posts at the ends of the given wall of cell (x, y)."""
    if wall_flag == WALL_NORTH:
        return (x, y), (x + 1, y)
    if wall_flag == WALL_SOUTH:
        return (x, y + 1), (x + 1, y + 1)
    if wall_flag == WALL_WEST:
        return (x, y), (x, y + 1)
    return (x + 1, y), (x + 1, y + 1)          # WALL_EAST


def safe_add_loops(maze: Maze, probability: float = 0.14, goal_margin: int = 2):
    """
    Replacement for IEEEMazeGenerator.add_loops.

    Same job -- punch bypass walls out to create alternative routes -- but refuses any
    removal that would strand a lattice post with zero walls. The IEEE spec requires
    every post outside the goal area to touch at least one partition wall; the original
    version violated this on ~5 posts per maze, which produces open plazas that do not
    occur in real competition mazes.
    """
    goal_set = set(maze.goal_indices)
    mx, my = maze.width // 2, maze.height // 2

    def in_goal_area(i, j):
        return abs(i - mx) <= goal_margin and abs(j - my) <= goal_margin

    for y in range(1, maze.height - 1):
        for x in range(1, maze.width - 1):
            if maze.coord_to_index(x, y) in goal_set:
                continue
            if random.random() >= probability:
                continue

            candidates = []
            for wall, dx, dy in ((WALL_EAST, 1, 0), (WALL_SOUTH, 0, 1)):
                n_idx = maze.coord_to_index(x + dx, y + dy)
                if n_idx < 0 or n_idx in goal_set:
                    continue
                if not maze.has_wall(x, y, wall):
                    continue                      # already open, nothing to remove
                candidates.append(wall)

            random.shuffle(candidates)
            for wall in candidates:
                p1, p2 = _segment_posts(x, y, wall)
                # Removing this wall drops each endpoint post by one.
                if any(post_wall_count(maze, *p) <= 1 and not in_goal_area(*p)
                       for p in (p1, p2)):
                    continue
                maze.set_wall(x, y, wall, False)
                break


def generate_maze(loop_prob: float = 0.14, max_attempts: int = 20) -> Maze:
    """One IEEE-legal maze, using the post-safe loop pass."""
    maze = Maze()
    for _ in range(max_attempts):
        IEEEMazeGenerator.fill_solid(maze)

        visited = [False] * maze.total_cells
        goal_set = set(maze.goal_indices)
        seed_idx = IEEEMazeGenerator.prepare_start_enclosure(maze)
        visited[maze.start_index] = True
        visited[seed_idx] = True
        stack = [seed_idx]

        while stack:
            curr_idx = stack[-1]
            cx, cy = maze.index_to_coord(curr_idx)
            neighbors = []
            for d in DIR_DELTA:
                n_idx = maze.coord_to_index(cx + d["x"], cy + d["y"])
                if n_idx >= 0 and not visited[n_idx] and n_idx not in goal_set:
                    neighbors.append((n_idx, d["wall"], d["opp"]))
            if neighbors:
                n_idx, w_wall, w_opp = random.choice(neighbors)
                maze.walls[curr_idx] &= ~w_wall
                maze.walls[n_idx] &= ~w_opp
                visited[n_idx] = True
                stack.append(n_idx)
            else:
                stack.pop()

        IEEEMazeGenerator.setup_center_goal(maze)
        if loop_prob > 0:
            safe_add_loops(maze, loop_prob)
        IEEEMazeGenerator.isolate_goal_island(maze)

        if IEEEMazeGenerator.is_wall_hugger_proof(maze):
            maze.reset_exploration()
            return maze

    raise RuntimeError(f"no wall-hugger-proof maze in {max_attempts} attempts")


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def time_cost_to_goal(maze: Maze, use_known: bool = False) -> np.ndarray:
    """
    Turn-weighted time-to-goal for every (heading, cell) state, by Dijkstra run
    backwards from the goal.

    State (c, h) means "mouse sitting in cell c, facing h". Its value is the minimum
    time to reach any goal cell from there. The transition out of (c, h) is: rotate to
    some heading h2 (paying turn_cost), then drive one cell forward (paying T_CELL),
    landing in (c2, h2). Heading has to be part of the state -- without it there is no
    way to charge for a turn, and the whole time-optimal claim collapses back to
    distance-optimal.

    Returns float32 [4, 256]; unreachable states hold UNREACHABLE.
    """
    n = maze.total_cells
    dist = np.full((NUM_HEADINGS, n), math.inf, dtype=np.float64)
    pq = []

    for g in maze.goals:
        for h in range(NUM_HEADINGS):
            dist[h, g["index"]] = 0.0
            heapq.heappush(pq, (0.0, g["index"], h))

    while pq:
        d, c2, h2 = heapq.heappop(pq)
        if d > dist[h2, c2]:
            continue
        # Who could have arrived at (c2, h2)? Whoever sat one cell back along h2.
        delta = DIR_DELTA[h2]
        cx, cy = maze.index_to_coord(c2)
        px, py = cx - delta["x"], cy - delta["y"]
        c = maze.coord_to_index(px, py)
        if c < 0:
            continue
        if maze.has_wall(px, py, delta["wall"], use_known):
            continue
        for h in range(NUM_HEADINGS):
            nd = d + T_CELL + turn_cost(h, h2)
            if nd < dist[h, c]:
                dist[h, c] = nd
                heapq.heappush(pq, (nd, c, h))

    dist[~np.isfinite(dist)] = UNREACHABLE
    return dist.astype(np.float32)


def bfs_distance(maze: Maze) -> np.ndarray:
    return np.asarray(maze.compute_ground_truth_bfs(), dtype=np.int16)


# ---------------------------------------------------------------------------
# Loader-side channel expansion (import this from your PyTorch Dataset)
# ---------------------------------------------------------------------------

def walls_to_channels(walls: np.ndarray, size: int = 16) -> np.ndarray:
    """
    Expand a compact wall mask [.., 256] into CNN input planes [.., 6, 16, 16]:
    N / E / S / W wall planes, a goal mask, and a start mask.

    Call this in __getitem__, not at write time -- it is cheap and keeps the stored
    dataset ~24x smaller.
    """
    w = np.asarray(walls).reshape(-1, size, size)
    planes = np.stack([(w & WALL_NORTH) > 0,
                       (w & WALL_EAST) > 0,
                       (w & WALL_SOUTH) > 0,
                       (w & WALL_WEST) > 0], axis=1).astype(np.float32)

    goal = np.zeros((1, size, size), dtype=np.float32)
    mid = size // 2
    goal[0, mid - 1:mid + 1, mid - 1:mid + 1] = 1.0
    start = np.zeros((1, size, size), dtype=np.float32)
    start[0, size - 1, 0] = 1.0

    b = planes.shape[0]
    extra = np.broadcast_to(np.concatenate([goal, start], axis=0), (b, 2, size, size))
    out = np.concatenate([planes, extra], axis=1)
    return out[0] if np.ndim(walls) == 1 else out


# ---------------------------------------------------------------------------
# Dataset build
# ---------------------------------------------------------------------------

@dataclass
class BuildConfig:
    n_mazes: int = 2000
    loop_prob: float = 0.14
    seed: int = 20260825
    shard_size: int = 2000
    out_dir: str = "data"
    split: tuple = (0.70, 0.15, 0.15)          # train / val / test, BY MAZE
    v_straight: float = V_STRAIGHT
    turn_cost: dict = None

    def __post_init__(self):
        if self.turn_cost is None:
            self.turn_cost = dict(TURN_COST)


def build(cfg: BuildConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    walls_all, bfs_all, time_all = [], [], []
    seen_hashes = set()
    duplicates = 0
    t0 = time.time()

    while len(walls_all) < cfg.n_mazes:
        m = generate_maze(cfg.loop_prob)
        w = np.asarray(m.walls, dtype=np.uint8)

        # Deduplicate. Identical mazes across the train/test boundary are silent
        # leakage, and DFS carving does repeat on small grids.
        digest = hashlib.blake2b(w.tobytes(), digest_size=16).digest()
        if digest in seen_hashes:
            duplicates += 1
            continue
        seen_hashes.add(digest)

        walls_all.append(w)
        bfs_all.append(bfs_distance(m))
        time_all.append(time_cost_to_goal(m))

        if len(walls_all) % 500 == 0:
            print(f"  {len(walls_all)}/{cfg.n_mazes}  ({time.time() - t0:.1f}s)")

    walls_all = np.stack(walls_all)
    bfs_all = np.stack(bfs_all)
    time_all = np.stack(time_all)

    # ---- split BY MAZE ----
    idx = np.random.permutation(len(walls_all))
    n_tr = int(cfg.split[0] * len(idx))
    n_va = int(cfg.split[1] * len(idx))
    splits = {"train": idx[:n_tr], "val": idx[n_tr:n_tr + n_va], "test": idx[n_tr + n_va:]}

    # ---- label normalisation: computed on TRAIN ONLY, applied everywhere ----
    tr_time = time_all[splits["train"]]
    finite = tr_time < UNREACHABLE
    label_mean = float(tr_time[finite].mean())
    label_std = float(tr_time[finite].std())

    manifest = {"config": asdict(cfg),
                "n_mazes": int(len(walls_all)),
                "duplicates_rejected": duplicates,
                "label_mean": label_mean,
                "label_std": label_std,
                "unreachable_sentinel": UNREACHABLE,
                "heading_order": ["N", "E", "S", "W"],
                "t_cell_seconds": T_CELL,
                "shards": {}}

    for name, ids in splits.items():
        shard_files = []
        for s, lo in enumerate(range(0, len(ids), cfg.shard_size)):
            sel = ids[lo:lo + cfg.shard_size]
            path = os.path.join(cfg.out_dir, f"{name}_{s:03d}.npz")
            np.savez_compressed(
                path,
                walls=walls_all[sel],
                bfs_dist=bfs_all[sel],
                time_cost=time_all[sel],
                reachable=(time_all[sel] < UNREACHABLE),
                maze_id=sel.astype(np.int32),
            )
            shard_files.append(os.path.basename(path))
        manifest["shards"][name] = shard_files

    with open(os.path.join(cfg.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"built {len(walls_all)} mazes in {time.time() - t0:.1f}s "
          f"({duplicates} duplicates rejected)")
    return manifest


if __name__ == "__main__":
    build(BuildConfig(n_mazes=500, out_dir="data_smoke"))
