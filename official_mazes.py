"""
official_mazes.py -- test CNN vs flood-fill on real competition mazes.

Maze files come from the public micromouseonline collection:
    git clone https://github.com/micromouseonline/mazefiles.git

    python3 official_mazes.py --dir mazefiles/classic --list            # what gets used
    python3 official_mazes.py --dir mazefiles/classic --model cnn_outbound_s0.pt --rows rows_official_s0.csv

Which mazes are used: All-Japan, APEC, UK, Taiwan and Portugal contest files (named by
contest and year), kept only if they pass the same rules our generator enforces --
full 16x16 active area (at least 240 of 256 cells reachable), start cell closed on east/south/west,
goal reachable -- with exact duplicates removed. Every rejection is printed with its
reason, so the selection can be reported in the paper.

Output rows have the same columns as batch_csv.py, so paper_eval.py --summary-only and
seeds_summary.py work on them unchanged. The 'seed' column holds the maze's index in
the selected list; the file name is in the 'maze' column.
"""

import argparse
import csv
import os
import re
import shutil
import sys
from multiprocessing import Pool

from maze import Maze, WALL_NORTH, WALL_EAST, WALL_SOUTH, WALL_WEST, DIR_DELTA

CONTESTS = {                 # filename pattern -> contest name for the paper
    r"^alljapan-": "All-Japan",
    r"^apec": "APEC",
    r"^uk": "UK",
    r"^taiwan": "Taiwan",
    r"^[Pp]ortugal": "Portugal",
}
MIN_REACHABLE = 240
# Official mazes take far longer to explore than random ones (flood-fill needs a median
# 61.4 s to verify its route, vs 24.2 s on random mazes), so budgets are scaled to the
# SAME FRACTIONS of full exploration time used for random mazes: 22/20/18/15/12 s of
# 24.2 s = 91/83/74/62/50 %  ->  56/51/45/38/31 s of 61.4 s.
BUDGETS = ["none", "56", "51", "45", "38", "31"]


def contest_of(name):
    for pat, label in CONTESTS.items():
        if re.match(pat, name):
            return label
    return None


def load_maze_file(path):
    """Parse the o---o / | text format into our Maze (y = 0 is the top row, start bottom-left)."""
    rows = [r for r in open(path).read().split("\n") if r.startswith(("o", "|"))]
    if len(rows) != 33:
        raise ValueError(f"expected 33 text rows, got {len(rows)}")
    m = Maze(16, 16)
    m.walls = [0] * 256
    for y in range(16):
        top, mid, bot = rows[2 * y], rows[2 * y + 1], rows[2 * y + 2]
        for x in range(16):
            c = 4 * x
            w = 0
            if top[c + 1:c + 4] == "---":
                w |= WALL_NORTH
            if bot[c + 1:c + 4] == "---":
                w |= WALL_SOUTH
            if mid[c:c + 1] == "|":
                w |= WALL_WEST
            if mid[c + 4:c + 5] == "|":
                w |= WALL_EAST
            m.walls[y * 16 + x] = w
    m.reset_exploration()
    return m


def reachable_cells(m):
    seen, stack = {m.start_index}, [m.start_index]
    while stack:
        i = stack.pop()
        x, y = m.index_to_coord(i)
        for d in DIR_DELTA:
            if not m.has_wall(x, y, d["wall"]):
                j = m.coord_to_index(x + d["x"], y + d["y"])
                if j >= 0 and j not in seen:
                    seen.add(j)
                    stack.append(j)
    return seen


def check(m):
    """Return None if the maze is usable, else the reason it is not."""
    sx, sy = m.start_x, m.start_y
    if not (m.has_wall(sx, sy, WALL_EAST) and m.has_wall(sx, sy, WALL_WEST)
            and m.has_wall(sx, sy, WALL_SOUTH)) or m.has_wall(sx, sy, WALL_NORTH):
        return "start cell not closed on east/south/west with exit north"
    r = reachable_cells(m)
    if not (m.goal_indices & r):
        return "goal not reachable"
    # Real contest mazes often seal off a few cells; that is still a full-size maze.
    # Fewer than MIN_REACHABLE means the event ran on a smaller active area.
    if len(r) < MIN_REACHABLE:
        return f"only {len(r)}/256 cells reachable (smaller active area)"
    return None


def select(maze_dir, verbose=True):
    keep, seen = [], {}
    for name in sorted(os.listdir(maze_dir)):
        label = contest_of(name)
        if not label or not name.endswith(".txt"):
            continue
        try:
            m = load_maze_file(os.path.join(maze_dir, name))
            why = check(m)
        except Exception as e:
            why = f"unreadable: {e}"
        if why is None:
            key = tuple(m.walls)
            if key in seen:
                why = f"duplicate of {seen[key]}"
            else:
                seen[key] = name
        if why:
            if verbose:
                print(f"  skip {name:<36} {why}")
            continue
        keep.append((name, label))
    return keep


# ---------------------------------------------------------------------------

def one_run(args):
    idx, name, maze_dir, pol_name, budget, alpha = args
    import batch_csv as B
    from episode import run_episode, EpisodeConfig
    from solver import oracle_time

    maze = load_maze_file(os.path.join(maze_dir, name))
    policy = dict(B.build_policies(alpha))[pol_name]()
    r = run_episode(maze.clone(), policy, EpisodeConfig(search_time_budget=budget))
    failed = r.fast_run_time >= B.NO_ROUTE
    fc, f90, f180, fm, fw = B.path_shape(r.fast_path) if not failed else (0, 0, 0, 0.0, 0.0)
    return {
        "seed": idx, "maze": name, "policy": pol_name,
        "budget": "none" if budget is None else budget,
        "reached_goal": int(r.reached_goal), "reached_route": int(not failed),
        "oracle_time_s": round(oracle_time(maze), 4),
        "fast_run_s": round(r.fast_run_time, 4) if not failed else "",
        "search_time_s": round(r.search_time, 4),
        "score": round(r.score, 4) if not failed else B.FAILURE_SCORE,
        "exploration_gap_s": round(r.exploration_gap, 4) if not failed else "",
        "planning_gap_s": round(r.planning_gap, 4) if not failed else "",
        "fast_cells": fc, "fast_metres": round(fm, 3), "fast_turns_90": f90,
        "fast_turns_180": f180, "fast_pivot_wheel_m": round(fw, 3),
        "search_steps": r.search_steps, "search_metres": round(r.search_steps * B.CELL_M, 3),
        "sorties": r.sorties, "cells_visited": r.cells_visited,
        "known_fraction": round(r.known_fraction, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="mazefiles/classic")
    ap.add_argument("--list", action="store_true", help="only show which mazes are used")
    ap.add_argument("--model", help="trained CNN .pt file")
    ap.add_argument("--rows", default="rows_official.csv")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--budgets", nargs="*", default=BUDGETS)
    ap.add_argument("--fold", choices=["A", "B"],
                    help="test only this half (see finetune_official.py); maze index stays global")
    a = ap.parse_args()

    keep = select(a.dir)
    by = {}
    for _, label in keep:
        by[label] = by.get(label, 0) + 1
    print(f"\nusing {len(keep)} official mazes: " +
          ", ".join(f"{k} {v}" for k, v in sorted(by.items())))
    if a.list:
        for name, label in keep:
            print(f"  {label:<9} {name}")
        return
    if not a.model:
        sys.exit("--model is required unless --list")

    active = "cnn_heuristic.pt"           # the file the CNN policy loads
    backup = None
    if os.path.abspath(a.model) != os.path.abspath(active):
        if os.path.exists(active):
            backup = active + ".bak"
            shutil.copy2(active, backup)
        shutil.copy2(a.model, active)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        only = None
        if a.fold:
            from finetune_official import folds
            only = set(folds(a.dir)[a.fold])
            print(f"fold {a.fold}: testing {len(only)} of {len(keep)} mazes")
        jobs = [(i, name, a.dir, pol, None if b == "none" else float(b), a.alpha)
                for i, (name, _) in enumerate(keep) if only is None or name in only
                for b in a.budgets for pol in ("flood_fill", "cnn_route_weighted")]
        print(f"{len(jobs)} runs on {a.jobs} cores ...", flush=True)
        from progress import run_with_progress
        with Pool(a.jobs) as pool:
            rows = run_with_progress(pool, one_run, jobs, chunksize=2)
    finally:
        if backup:
            shutil.move(backup, active)

    with open(a.rows, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {a.rows}")


if __name__ == "__main__":
    main()
