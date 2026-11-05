"""
baselines_eval.py -- run the two reference policies of Figs 2 and 3 on the same 200 generated
mazes and the same six budgets as paper_eval.py:

    optimistic_shortcut   flood-fill variant that reduces value for cells and treats unknown
                          walls as open when looking for a shortcut
    oracle_guided_BOUND   an oracle that reads the true maze and follows its optimal route
                          (upper limit on what any exploration policy can achieve)

    python3 baselines_eval.py --out results/rows_baselines.csv

Output has the same columns as the rows_seed*.csv files written by batch_csv.py, so
make_figures.py can read both. No trained network is needed.
"""

import argparse
import csv
import multiprocessing as mp
import os

from batch_csv import FIELDS, one_cell
from progress import run_with_progress

POLICIES = ["optimistic_shortcut", "oracle_guided_BOUND"]
BUDGETS = [None, 22.0, 20.0, 18.0, 15.0, 12.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="number of generated mazes")
    ap.add_argument("--seed0", type=int, default=0, help="first maze seed")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="rows_baselines.csv")
    a = ap.parse_args()

    jobs = [(a.seed0 + i, p, b, 2.0, None, True)
            for b in BUDGETS for p in POLICIES for i in range(a.n)]
    print(f"{a.n} mazes x {len(POLICIES)} policies x {len(BUDGETS)} budgets = {len(jobs)} runs")
    with mp.Pool(a.jobs) as pool:
        rows = run_with_progress(pool, one_cell, jobs, chunksize=4)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
