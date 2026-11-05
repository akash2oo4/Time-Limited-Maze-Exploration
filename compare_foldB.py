"""
compare_foldB.py -- three ways of getting a CNN for competition mazes, on the SAME
69 official fold-B mazes, none of which any of them saw:

    base      trained on random mazes                  rows_official_s<k>.csv (filtered to B)
    finetune  base + fine-tuned on official fold A     rows_ft_s<k>_B.csv
    contest   trained on contest-style mazes            rows_contest_s<k>_B.csv
              (generator calibrated on fold A)

    python3 compare_foldB.py                 -> prints and saves contest_results.md
"""

import os
import sys

import numpy as np

from paper_eval import load_rows, summarise
from seeds_summary import budgets_in
from finetune_official import folds

SEEDS = ["0", "1", "2"]
METHODS = [
    ("base (random mazes)", "rows_official_s{}.csv"),
    ("fine-tuned on fold A", "rows_ft_s{}_B.csv"),
    ("contest-style training", "rows_contest_s{}_B.csv"),
]


def main(maze_dir="mazefiles/classic", out="contest_results.md"):
    fold_b = set(folds(maze_dir)["B"])
    L = ["# Official fold B (69 mazes, unseen by every method): CNN vs flood-fill", "",
         "Score margin = flood-fill score - CNN score, mean over 3 seeds [min - max]. "
         "Positive = CNN better. Sig = seeds whose paired 95% CI excludes zero "
         "(better / worse).", ""]
    budgets = None
    tables = {}
    for label, pat in METHODS:
        files = [pat.format(s) for s in SEEDS]
        missing = [f for f in files if not os.path.exists(f)]
        if missing:
            print(f"skipping {label}: missing {', '.join(missing)}")
            continue
        per = []
        for f in files:
            rows = [r for r in load_rows(f) if r.get("maze") in fold_b]
            budgets = budgets or budgets_in(f)
            per.append({r["budget"]: r for r in summarise(rows, budgets)})
        tables[label] = per

    head = "| Budget (s) | " + " | ".join(f"{m} margin | sig" for m in tables) + " |"
    L += [head, "|---|" + "---|---|" * len(tables)]
    for b in budgets:
        cells = []
        for label, per in tables.items():
            m = np.array([p[b]["margin_mean"] for p in per])
            better = sum(1 for p in per if p[b]["significant"] == "yes" and p[b]["margin_mean"] > 0)
            worse = sum(1 for p in per if p[b]["significant"] == "yes" and p[b]["margin_mean"] < 0)
            cells.append(f"{m.mean():+.3f} [{m.min():+.3f} - {m.max():+.3f}] | {better}/{worse}")
        L.append(f"| {b} | " + " | ".join(cells) + " |")
    n = next(iter(tables.values()))[0][budgets[0]]["n"]
    L += ["", f"{n} mazes per budget per seed."]
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
