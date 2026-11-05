"""
contest_dataset.py -- build the training dataset from contest-style mazes.

Same snapshot pipeline as snapshot_dataset.py (outbound-only, 2 snapshots per maze, split
by maze, same labels); only the maze source changes, to contest_generator. The existing
files are not modified: the generator is swapped in for this process only.

    python3 contest_dataset.py build --table contest_target.json --n 20000 --out data_contest
    python3 contest_dataset.py check --table contest_target.json --n 300     # stats vs official
"""

import argparse
import os
import random

import numpy as np

import contest_generator as cg
import snapshot_dataset as sd


def _use_table(path):
    import json
    cg.TABLE = json.load(open(path))
    sd.generate_maze = cg.generate_contest_maze      # this process (and its workers) only


def build(table, n, out, jobs):
    _use_table(table)
    sd.build(sd.SnapshotConfig(n_mazes=n, snapshots_per_maze=2, out_dir=out,
                               outbound_only=True, seed=20261015), n_jobs=jobs)


def _check_one(seed):
    from maze_stats import stats
    random.seed(seed)
    return stats(cg.generate_contest_maze(), with_explore=True)


def check(table, n, jobs, maze_dir):
    """Print the same statistics as maze_stats.py for contest-style mazes vs official."""
    from multiprocessing import Pool
    from maze_stats import _official
    from official_mazes import select
    _use_table(table)
    names = [os.path.join(maze_dir, x) for x, _ in select(maze_dir, verbose=False)]
    with Pool(jobs) as p:
        gen = p.map(_check_one, range(70_000_000, 70_000_000 + n))
        off = p.map(_official, names)
    print(f"\n{'metric':<18}{'official':>10}{'contest-style':>15}{'ratio':>8}")
    for k in gen[0]:
        o, g = np.mean([r[k] for r in off]), np.mean([r[k] for r in gen])
        print(f"{k:<18}{o:>10.2f}{g:>15.2f}{o / g:>8.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "check"])
    ap.add_argument("--table", default="contest_target.json")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--out", default="data_contest")
    ap.add_argument("--dir", default="mazefiles/classic")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.table, a.n, a.out, a.jobs)
    else:
        check(a.table, a.n, a.jobs, a.dir)


if __name__ == "__main__":
    main()
