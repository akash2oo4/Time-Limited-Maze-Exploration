"""
maze_stats.py -- how do generated mazes differ from competition mazes?

    python3 maze_stats.py --dir mazefiles/classic --n 500

Per maze it measures:
    route_cells     cells on the time-optimal start->goal route
    route_turns     90 deg turns on that route
    staircase       share of route turns that immediately reverse the previous turn
                    (L-R-L-R zigzags = diagonal runs, a hallmark of designed mazes)
    route_time_s    optimal fast-run time under the simulator's cost model
    longest_straight longest straight corridor anywhere in the maze (cells)
    dead_ends       cells with exactly one opening
    junctions       cells with three or four openings
    loops           extra openings beyond a spanning tree (0 = perfect maze, no loops)
    explore_s       flood-fill search time to verify its route (no time limit)
and prints mean / median for both groups plus the ratio official / generated.
"""

import argparse
import os
import random

import numpy as np

from maze import DIR_DELTA, WALL_NORTH, WALL_EAST, WALL_SOUTH, WALL_WEST
from solver import truth_passable, time_to_goal, extract_path
from belief import NORTH


def stats(m, with_explore=True):
    W = H = 16
    opens = np.zeros((H, W), int)
    n_open_edges = 0
    for y in range(H):
        for x in range(W):
            for d in DIR_DELTA:
                if not m.has_wall(x, y, d["wall"]):
                    if 0 <= x + d["x"] < W and 0 <= y + d["y"] < H:
                        opens[y, x] += 1
            if x + 1 < W and not m.has_wall(x, y, WALL_EAST):
                n_open_edges += 1
            if y + 1 < H and not m.has_wall(x, y, WALL_SOUTH):
                n_open_edges += 1
    reach = int((opens > 0).sum())
    goal_cells = 4
    # goal interior has 4 open edges forming one cycle by rule; do not count it as a loop
    loops = n_open_edges - (reach - 1) - 1

    # longest straight corridor (count consecutive cells joined by open edges in a line)
    longest = 0
    for y in range(H):
        run = 1
        for x in range(1, W):
            run = run + 1 if not m.has_wall(x - 1, y, WALL_EAST) else 1
            longest = max(longest, run)
    for x in range(W):
        run = 1
        for y in range(1, H):
            run = run + 1 if not m.has_wall(x, y - 1, WALL_SOUTH) else 1
            longest = max(longest, run)

    p = truth_passable(m)
    goals = sorted(m.goal_indices)
    cost = time_to_goal(p, goals)
    t, path = extract_path(cost, p, (m.start_x, m.start_y), NORTH, goals)
    heads = [h for _, _, h in path]
    turns = []
    for i in range(1, len(heads)):
        d = (heads[i] - heads[i - 1]) % 4
        if d in (1, 3):
            turns.append(d)
    stair = sum(1 for i in range(1, len(turns)) if turns[i] != turns[i - 1])
    out = {
        "route_cells": len(path) - 1,
        "route_turns": len(turns),
        "staircase": stair / max(1, len(turns) - 1),
        "route_time_s": float(t),
        "longest_straight": longest,
        "dead_ends": int((opens == 1).sum()),
        "junctions": int((opens >= 3).sum()),
        "loops": int(loops),
    }
    if with_explore:
        from episode import run_episode, EpisodeConfig, FloodFillPolicy
        out["explore_s"] = run_episode(m.clone(), FloodFillPolicy(), EpisodeConfig()).search_time
    return out


def _official(path):
    from official_mazes import load_maze_file
    return stats(load_maze_file(path))


def _generated(seed):
    from dataset_generator import generate_maze
    random.seed(seed)
    return stats(generate_maze())


def summarise(label, rows):
    keys = list(rows[0])
    return {k: (np.mean([r[k] for r in rows]), np.median([r[k] for r in rows])) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="mazefiles/classic")
    ap.add_argument("--n", type=int, default=500, help="generated mazes to sample")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    from multiprocessing import Pool
    from official_mazes import select
    names = [os.path.join(a.dir, n) for n, _ in select(a.dir, verbose=False)]
    with Pool(a.jobs) as pool:
        off = pool.map(_official, names)
        gen = pool.map(_generated, range(10_000_000, 10_000_000 + a.n))
    so, sg = summarise("official", off), summarise("generated", gen)
    print(f"\n{'metric':<18}{'official mean':>14}{'(median)':>10}{'generated mean':>16}"
          f"{'(median)':>10}{'ratio':>8}")
    for k in so:
        r = so[k][0] / sg[k][0] if sg[k][0] else float("nan")
        print(f"{k:<18}{so[k][0]:>14.2f}{so[k][1]:>10.2f}{sg[k][0]:>16.2f}{sg[k][1]:>10.2f}{r:>8.2f}")
    print(f"\n{len(off)} official mazes vs {len(gen)} generated")


if __name__ == "__main__":
    main()
