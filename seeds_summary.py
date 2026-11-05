"""
seeds_summary.py -- combine CNN-vs-flood-fill results from several training seeds.

    python3 seeds_summary.py rows_seed0.csv rows_seed1.csv rows_seed2.csv

Each file is the paper_rows.csv written by one paper_eval.py run (same mazes, different
trained model). For every budget it reports, across seeds: the mean and the min-max range
of gap closed, failures and score margin, plus how many seeds beat flood-fill
significantly. Writes seeds_results.md.
"""

import sys

import numpy as np

from paper_eval import load_rows, summarise



def budgets_in(path):
    """Budgets present in a rows file: 'none' first, then largest to smallest."""
    vals = {r["budget"] for r in load_rows(path)}
    nums = sorted((float(v) for v in vals if v != "none"), reverse=True)
    return (["none"] if "none" in vals else []) + [f"{v:g}" for v in nums]


def main(files):
    budgets = budgets_in(files[0])
    per_seed = [{r["budget"]: r for r in summarise(load_rows(f), budgets)} for f in files]
    k = len(files)

    L = [f"# CNN vs flood-fill across {k} training seeds "
         f"({per_seed[0]['none']['n']} mazes each)", "",
         "Values: mean over seeds [min - max]. Flood-fill is identical across seeds "
         "(same mazes), so its columns are single numbers.", "",
         "| Budget (s) | Gap closed | CNN failures | FF failures | Score margin (s) "
         "| Seeds significantly better |",
         "|---|---|---|---|---|---|"]

    def rng(vals, fmt):
        v = np.array(vals, float)
        if np.all(np.isnan(v)):
            return "--"
        return f"{fmt.format(np.nanmean(v))} [{fmt.format(np.nanmin(v))} - {fmt.format(np.nanmax(v))}]"

    for b in budgets:
        rows = [s[b] for s in per_seed if b in s]
        if len(rows) != k:
            continue
        sig_better = sum(1 for r in rows if r["significant"] == "yes" and r["margin_mean"] > 0)
        sig_worse = sum(1 for r in rows if r["significant"] == "yes" and r["margin_mean"] < 0)
        sig = f"{sig_better} / {k}" + (f" (worse in {sig_worse})" if sig_worse else "")
        L.append(f"| {b} | {rng([r['gap_closed_pct'] for r in rows], '{:.0f}%')} "
                 f"| {rng([r['cnn_fail'] for r in rows], '{:.1f}')} | {rows[0]['ff_fail']} "
                 f"| {rng([r['margin_mean'] for r in rows], '{:+.3f}')} | {sig} |")

    L += ["", "Positive margin = CNN better. 'Seeds significantly better' counts seeds whose "
          "paired 95% CI excludes zero in the CNN's favour."]
    open(OUT, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nsaved {OUT}")


OUT = "seeds_results.md"

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--out" in args:
        i = args.index("--out")
        OUT = args[i + 1]
        del args[i:i + 2]
    if len(args) < 1:
        sys.exit(__doc__)
    main(args)
