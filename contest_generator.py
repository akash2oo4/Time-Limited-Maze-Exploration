"""
contest_generator.py -- generate IEEE-legal mazes that look like competition mazes.

Measured (maze_stats.py): the time-optimal route in a contest maze is ~2.8x longer than
in our random generator's mazes (72 vs 26 cells), with longer straights and fewer loops.
Two knobs alone cannot close that gap, so this generator:

    1. draws per-maze knobs: loop_prob ~ U(0, 0.08) (fewer shortcuts than the default
       0.14) and straight_bias ~ U(0.3, 0.7) (carver keeps going straight more often);
    2. keeps or rejects each candidate so that the optimal ROUTE LENGTH follows the
       distribution measured on official fold A only -- fold B stays untouched for testing;
    3. adds decoy loops (10-20 per maze) that do not shorten the route, as designers do.

Every maze passes the same IEEE rules as before (start enclosure, single-entrance goal,
post rule via safe_add_loops, wall-hugger-proof goal island). Nothing in the existing
files is changed; this module only reuses their helpers.

    python3 contest_generator.py calibrate --dir mazefiles/classic --fold A --out contest_target.json
"""

import argparse
import json
import os
import random

import numpy as np

from maze import Maze, DIR_DELTA
from maze_generator import IEEEMazeGenerator
from dataset_generator import safe_add_loops, post_wall_count, _segment_posts
from maze import WALL_EAST, WALL_SOUTH
from solver import truth_passable, time_to_goal, extract_path
from belief import NORTH

BINS = [0, 30, 40, 50, 60, 70, 80, 90, 100, 110, 10_000]     # route length, cells
LOOP_RANGE = (0.0, 0.08)
STRAIGHT_RANGE = (0.3, 0.7)
DECOY_LOOPS = (10, 20)


def generate_styled(loop_prob, straight_bias, max_attempts=20):
    """The original DFS carver plus a straight-ahead bias. Same IEEE guarantees."""
    maze = Maze()
    for _ in range(max_attempts):
        IEEEMazeGenerator.fill_solid(maze)
        visited = [False] * maze.total_cells
        goal_set = set(maze.goal_indices)
        seed_idx = IEEEMazeGenerator.prepare_start_enclosure(maze)
        visited[maze.start_index] = True
        visited[seed_idx] = True
        stack = [seed_idx]
        arrived = {seed_idx: 0}                  # start cell exits north
        while stack:
            cur = stack[-1]
            cx, cy = maze.index_to_coord(cur)
            nbrs = []
            for h, d in enumerate(DIR_DELTA):
                n = maze.coord_to_index(cx + d["x"], cy + d["y"])
                if n >= 0 and not visited[n] and n not in goal_set:
                    nbrs.append((n, d["wall"], d["opp"], h))
            if nbrs:
                ahead = [b for b in nbrs if b[3] == arrived.get(cur)]
                if ahead and random.random() < straight_bias:
                    n, w, wo, h = ahead[0]
                else:
                    n, w, wo, h = random.choice(nbrs)
                maze.walls[cur] &= ~w
                maze.walls[n] &= ~wo
                visited[n] = True
                arrived[n] = h
                stack.append(n)
            else:
                stack.pop()
        IEEEMazeGenerator.setup_center_goal(maze)
        if loop_prob > 0:
            safe_add_loops(maze, loop_prob)
        IEEEMazeGenerator.isolate_goal_island(maze)
        if IEEEMazeGenerator.is_wall_hugger_proof(maze):
            maze.reset_exploration()
            return maze
    raise RuntimeError("no wall-hugger-proof maze")


def route_cells(maze):
    p = truth_passable(maze)
    goals = sorted(maze.goal_indices)
    _, path = extract_path(time_to_goal(p, goals), p, (maze.start_x, maze.start_y),
                           NORTH, goals)
    return len(path) - 1


def count_loops(maze):
    """Openings beyond a spanning tree of the reachable cells (goal's own cycle excluded)."""
    W, H = maze.width, maze.height
    edges = sum(1 for y in range(H) for x in range(W) for w, dx, dy in
                ((WALL_EAST, 1, 0), (WALL_SOUTH, 0, 1))
                if x + dx < W and y + dy < H and not maze.has_wall(x, y, w))
    return edges - (W * H - 1) - 1


def add_decoy_loops(maze, target_loops, keep=0.95, tries=80):
    """
    Open extra walls -- side loops, like a designer's decoys -- but keep an opening only
    if the optimal route stays at least `keep` of its length, the post rule holds, and
    the goal island stays wall-hugger-proof. Contest mazes have long routes AND many
    loops; random loops alone shorten the route.
    """
    base = route_cells(maze)
    goal = set(maze.goal_indices)
    mx, my = maze.width // 2, maze.height // 2
    for _ in range(tries):
        if count_loops(maze) >= target_loops:
            break
        x, y = random.randint(1, maze.width - 2), random.randint(1, maze.height - 2)
        wall, dx, dy = random.choice(((WALL_EAST, 1, 0), (WALL_SOUTH, 0, 1)))
        if maze.coord_to_index(x, y) in goal or maze.coord_to_index(x + dx, y + dy) in goal:
            continue
        if not maze.has_wall(x, y, wall):
            continue
        p1, p2 = _segment_posts(x, y, wall)
        if any(post_wall_count(maze, *p) <= 1 and not (abs(p[0] - mx) <= 2 and abs(p[1] - my) <= 2)
               for p in (p1, p2)):
            continue
        maze.set_wall(x, y, wall, False)
        if route_cells(maze) < keep * base or not IEEEMazeGenerator.is_wall_hugger_proof(maze):
            maze.set_wall(x, y, wall, True)          # undo: it was a real shortcut
    maze.reset_exploration()
    return maze


def _bin(n):
    return int(np.searchsorted(BINS, n, side="right") - 1)


def _candidate():
    lp = random.uniform(*LOOP_RANGE)
    sb = random.uniform(*STRAIGHT_RANGE)
    m = generate_styled(lp, sb)
    return m, route_cells(m)


def _pilot_one(seed):
    random.seed(seed)
    return _bin(_candidate()[1])


def calibrate(maze_dir, fold, out, pilot=3000, jobs=None):
    """Target histogram from ONE official fold + candidate histogram -> acceptance table."""
    from multiprocessing import Pool
    from official_mazes import load_maze_file
    from finetune_official import folds
    names = folds(maze_dir)[fold]
    target = np.zeros(len(BINS) - 1)
    for n in names:
        target[_bin(route_cells(load_maze_file(os.path.join(maze_dir, n))))] += 1
    with Pool(jobs or os.cpu_count()) as p:
        pool_bins = p.map(_pilot_one, range(90_000_000, 90_000_000 + pilot))
    pool = np.bincount(pool_bins, minlength=len(BINS) - 1).astype(float)
    t, q = target / target.sum(), pool / pool.sum()
    ratio = np.where(q > 0, t / np.maximum(q, 1e-12), 0.0)
    accept = ratio / ratio.max()
    expected = float((q * accept).sum())
    table = {"bins": BINS, "fold": fold, "n_official": len(names),
             "target_share": t.round(4).tolist(), "pool_share": q.round(4).tolist(),
             "accept_prob": accept.round(4).tolist(),
             "loop_range": LOOP_RANGE, "straight_range": STRAIGHT_RANGE,
             "expected_accept_rate": round(expected, 4),
             # target loop count per maze; official mazes average ~15 (maze_stats.py)
             "decoy_loops": DECOY_LOOPS}
    if (t[q == 0] > 0).any():
        table["warning"] = "some official route lengths never occur in the candidate pool"
    json.dump(table, open(out, "w"), indent=2)
    print(f"{out}: calibrated on fold {fold} ({len(names)} mazes); "
          f"about 1 in {1 / expected:.1f} candidates accepted")
    for lo, hi, a, b, c in zip(BINS[:-1], BINS[1:], t, q, accept):
        print(f"  route {lo:>3}-{hi if hi < 10000 else '+':<4} official {a:5.2f}  pool {b:5.2f}  accept {c:4.2f}")
    return table


# ---------------------------------------------------------------------------
# Drop-in generator for snapshot_dataset (no arguments, deterministic per seed)

TABLE = None


def generate_contest_maze():
    """Uses the module-level random state, which build_maze_samples seeds per maze."""
    acc = TABLE["accept_prob"]
    lo, hi = TABLE.get("decoy_loops", (0, 0))
    for _ in range(10_000):
        m, n = _candidate()
        if random.random() < acc[_bin(n)]:
            if hi > 0:
                add_decoy_loops(m, random.randint(lo, hi))
            return m
    raise RuntimeError("acceptance loop did not terminate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calibrate"])
    ap.add_argument("--dir", default="mazefiles/classic")
    ap.add_argument("--fold", default="A", choices=["A", "B"])
    ap.add_argument("--out", default="contest_target.json")
    ap.add_argument("--pilot", type=int, default=3000)
    a = ap.parse_args()
    calibrate(a.dir, a.fold, a.out, a.pilot)


if __name__ == "__main__":
    main()
