"""
alpha_cv.py -- retune the coupling strength alpha for official mazes, without touching
the test half.

The CNN's route predictions become driving costs as  T_CELL * (1 + alpha * (1 - p_route)).
alpha = 2.0 was tuned on random mazes. Here, for each fine-tuned model (trained on one
half of the official mazes):

    1. SELECT: try every alpha in the grid on the half the model was fine-tuned on;
       keep the alpha with the lowest mean CNN score over the finite budgets.
    2. TEST:   run that one alpha (and flood-fill) on the OTHER half, which neither the
       fine-tuning nor the selection ever saw.

    OMP_NUM_THREADS=1 python3 alpha_cv.py                 # seeds 0 1 2, grid 0.5 1 1.5 2 3
    python3 seeds_summary.py rows_alpha_s0.csv rows_alpha_s1.csv rows_alpha_s2.csv --out alpha_results.md

Outputs: rows_alpha_s<seed>.csv (both test halves joined, same columns as the other rows
files) and alpha_choices.csv (what was picked, with every candidate's selection score).
"""

import argparse
import csv
import os
import shutil
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")    # one thread per worker (see speed fix)
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from multiprocessing import Pool

import numpy as np

from official_mazes import select, one_run, BUDGETS
from finetune_official import folds
from progress import run_with_progress

ACTIVE = "cnn_heuristic.pt"


def run_jobs(model, jobs, n_jobs):
    """Run (idx, name, dir, policy, budget, alpha) jobs with `model` as the active CNN."""
    shutil.copy2(model, ACTIVE)
    with Pool(n_jobs) as pool:
        return run_with_progress(pool, one_run, jobs, chunksize=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="mazefiles/classic")
    ap.add_argument("--seeds", nargs="*", default=["0", "1", "2"])
    ap.add_argument("--grid", nargs="*", type=float, default=[0.5, 1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--max-mazes", type=int, default=None, help="debug: use only the first N of each half")
    a = ap.parse_args()

    keep = select(a.dir, verbose=False)
    index = {name: i for i, (name, _) in enumerate(keep)}
    halves = folds(a.dir)
    if a.max_mazes:
        halves = {k: v[:a.max_mazes] for k, v in halves.items()}
    finite = [b for b in BUDGETS if b != "none"]
    budget_val = lambda b: None if b == "none" else float(b)

    backup = None
    if os.path.exists(ACTIVE):
        backup = ACTIVE + ".bak"
        shutil.copy2(ACTIVE, backup)

    choices = []
    try:
        for s in a.seeds:
            out_rows = []
            for test in "AB":
                train = "B" if test == "A" else "A"
                model = f"cnn_ft_s{s}_on{train}.pt"
                if not os.path.exists(model):
                    sys.exit(f"missing {model} -- run run_finetune.sh first")

                # 1. select alpha on the fine-tuning half
                print(f"\n=== seed {s}: choose alpha on fold {train} "
                      f"({len(halves[train])} mazes, grid {a.grid}) ===", flush=True)
                jobs = [(index[n], n, a.dir, "cnn_route_weighted", budget_val(b), al)
                        for al in a.grid for n in halves[train] for b in finite]
                rows = run_jobs(model, jobs, a.jobs)
                score = {}
                for (_, _, _, _, _, al), r in zip(jobs, rows):
                    score.setdefault(al, []).append(float(r["score"]))
                mean = {al: float(np.mean(v)) for al, v in score.items()}
                # lowest mean score wins; ties go to the value closest to the original 2.0
                best = min(mean, key=lambda al: (round(mean[al], 6), abs(al - 2.0)))
                print("  mean CNN score by alpha: " +
                      "  ".join(f"{al:g}: {m:.3f}" for al, m in sorted(mean.items())) +
                      f"   -> chose alpha {best:g}")
                choices.append({"seed": s, "test_fold": test, "chosen_alpha": best,
                                **{f"score_alpha_{al:g}": round(m, 4) for al, m in mean.items()}})

                # 2. test the chosen alpha on the unseen half
                print(f"=== seed {s}: test alpha {best:g} on fold {test} "
                      f"({len(halves[test])} mazes, never seen) ===", flush=True)
                jobs = [(index[n], n, a.dir, pol, budget_val(b), best)
                        for n in halves[test] for b in BUDGETS
                        for pol in ("flood_fill", "cnn_route_weighted")]
                out_rows += run_jobs(model, jobs, a.jobs)

            path = f"rows_alpha_s{s}.csv"
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
                w.writeheader()
                w.writerows(out_rows)
            print(f"wrote {len(out_rows)} rows to {path}")
    finally:
        if backup:
            shutil.move(backup, ACTIVE)

    with open("alpha_choices.csv", "w", newline="") as f:
        keys = sorted({k for c in choices for k in c}, key=lambda k: (not k.startswith(("seed", "test", "chosen")), k))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(choices)
    print("\nalpha chosen per (seed, test half):")
    for c in choices:
        print(f"  seed {c['seed']}  test {c['test_fold']}: alpha {c['chosen_alpha']:g}")
    print("saved alpha_choices.csv")


if __name__ == "__main__":
    main()
