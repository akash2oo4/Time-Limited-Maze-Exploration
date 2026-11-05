"""
paper_eval.py -- test a trained CNN against flood-fill at every search budget and
write the tables for the paper.

    python3 paper_eval.py --model cnn_heuristic_sortie.pt

Defaults: 200 mazes, budgets none 22 20 18 15 12, all CPU cores.
Outputs (in the current folder):
    paper_rows.csv      one row per (maze, policy, budget) -- the raw data
    paper_results.csv   one summary row per budget -- for plotting / spreadsheets
    paper_results.md    the same tables, ready to paste into the paper

Both mice see exactly the same mazes (same seeds), so every comparison is paired.
Failed runs (no verified route when time ran out) are counted, never dropped.
"""

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys

import numpy as np

ACTIVE = "cnn_heuristic.pt"          # the file batch_csv.py loads
FF, CNN = "flood_fill", "cnn_route_weighted"


def run_benchmark(model, n, budgets, jobs, rows_csv):
    """Put the chosen model in place, run batch_csv.py once for all budgets."""
    if not os.path.exists(model):
        sys.exit(f"model not found: {model}")
    backup = None
    if os.path.abspath(model) != os.path.abspath(ACTIVE):
        if os.path.exists(ACTIVE):
            backup = ACTIVE + ".bak"
            shutil.copy2(ACTIVE, backup)
        shutil.copy2(model, ACTIVE)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")   # CPU workers, no GPU contention
    cmd = [sys.executable, "batch_csv.py", str(n), "--budgets", *budgets,
           "--jobs", str(jobs), "--out", rows_csv]
    print("running:", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=True, env=env)
    finally:
        if backup:                                   # put the old active model back
            shutil.move(backup, ACTIVE)


def load_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(x):
    return float(x) if x not in ("", None) else math.nan


def ci95(x):
    """Mean and 95% confidence half-width (normal approximation, fine for n >= 30)."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()) if len(x) else math.nan, math.nan
    return float(x.mean()), float(1.96 * x.std(ddof=1) / math.sqrt(len(x)))


def summarise(rows, budgets):
    out = []
    for b in budgets:
        by = {}
        for r in rows:
            if r["budget"] == b or (b != "none" and r["budget"] != "none"
                                    and fnum(r["budget"]) == fnum(b)):
                by.setdefault(r["policy"], {})[int(r["seed"])] = r
        if FF not in by or CNN not in by:
            continue
        seeds = sorted(set(by[FF]) & set(by[CNN]))
        ff = [by[FF][s] for s in seeds]
        cn = [by[CNN][s] for s in seeds]

        def pol_stats(rs):
            ok = [r for r in rs if r["reached_route"] == "1"]
            gaps = np.array([fnum(r["exploration_gap_s"]) for r in ok])
            return {
                "fail": len(rs) - len(ok),
                "gap_mean": float(gaps.mean()) if len(gaps) else math.nan,
                "gap_median": float(np.median(gaps)) if len(gaps) else math.nan,
                "gap_p90": float(np.percentile(gaps, 90)) if len(gaps) else math.nan,
                "score_mean": float(np.mean([fnum(r["score"]) for r in rs])),
                "search_s": float(np.mean([fnum(r["search_time_s"]) for r in rs])),
                "sensed": float(np.mean([fnum(r["known_fraction"]) for r in rs])),
            }

        s_ff, s_cn = pol_stats(ff), pol_stats(cn)
        # margin > 0 means the CNN scored better (lower score is better)
        margin = np.array([fnum(a["score"]) - fnum(c["score"]) for a, c in zip(ff, cn)])
        draws = int(np.sum(np.abs(margin) < 1e-6))
        cnn_w = int(np.sum(margin >= 1e-6))
        ff_w = int(np.sum(margin <= -1e-6))
        m, h = ci95(margin)
        closed = (100 * (s_ff["gap_mean"] - s_cn["gap_mean"]) / s_ff["gap_mean"]
                  if s_ff["gap_mean"] > 1e-9 else math.nan)
        out.append({
            "budget": b, "n": len(seeds),
            "ff_gap_mean": s_ff["gap_mean"], "cnn_gap_mean": s_cn["gap_mean"],
            "gap_closed_pct": closed,
            "ff_gap_median": s_ff["gap_median"], "cnn_gap_median": s_cn["gap_median"],
            "ff_gap_p90": s_ff["gap_p90"], "cnn_gap_p90": s_cn["gap_p90"],
            "ff_fail": s_ff["fail"], "cnn_fail": s_cn["fail"],
            "ff_score": s_ff["score_mean"], "cnn_score": s_cn["score_mean"],
            "ff_search_s": s_ff["search_s"], "cnn_search_s": s_cn["search_s"],
            "ff_sensed_pct": 100 * s_ff["sensed"], "cnn_sensed_pct": 100 * s_cn["sensed"],
            "cnn_wins": cnn_w, "ff_wins": ff_w, "draws": draws,
            "margin_mean": m, "margin_ci95": h,
            "significant": "yes" if not math.isnan(h) and abs(m) > h else "no",
        })
    return out


def f3(x):
    return "--" if isinstance(x, float) and math.isnan(x) else f"{x:.3f}"


def write_outputs(summary, model, csv_path, md_path):
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        for r in summary:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    L = [f"# CNN vs flood-fill -- model `{os.path.basename(model)}`, "
         f"{summary[0]['n']} mazes per budget", ""]
    L += ["## Table 1 -- Route quality (exploration gap = seconds slower than the optimal route)",
          "", "| Budget (s) | Flood-fill mean | CNN mean | Gap closed | Flood-fill p90 | CNN p90 "
          "| Flood-fill median | CNN median | Failures FF / CNN |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in summary:
        gc = "--" if math.isnan(r["gap_closed_pct"]) else f"{r['gap_closed_pct']:.0f}%"
        L.append(f"| {r['budget']} | {f3(r['ff_gap_mean'])} | {f3(r['cnn_gap_mean'])} | {gc} "
                 f"| {f3(r['ff_gap_p90'])} | {f3(r['cnn_gap_p90'])} | {f3(r['ff_gap_median'])} "
                 f"| {f3(r['cnn_gap_median'])} | {r['ff_fail']} / {r['cnn_fail']} |")
    L += ["", "## Table 2 -- Head-to-head on total score (lower score is better)", "",
          "| Budget (s) | CNN wins | Flood-fill wins | Draws | Mean margin (s) | 95% CI | Significant |",
          "|---|---|---|---|---|---|---|"]
    for r in summary:
        n = r["n"]
        L.append(f"| {r['budget']} | {r['cnn_wins']} ({100*r['cnn_wins']/n:.0f}%) "
                 f"| {r['ff_wins']} ({100*r['ff_wins']/n:.0f}%) | {r['draws']} "
                 f"| {r['margin_mean']:+.3f} | ±{f3(r['margin_ci95'])} | {r['significant']} |")
    L += ["", "## Table 3 -- Search effort", "",
          "| Budget (s) | Score FF | Score CNN | Search time FF (s) | Search time CNN (s) "
          "| Maze sensed FF | Maze sensed CNN |", "|---|---|---|---|---|---|---|"]
    for r in summary:
        L.append(f"| {r['budget']} | {f3(r['ff_score'])} | {f3(r['cnn_score'])} "
                 f"| {r['ff_search_s']:.1f} | {r['cnn_search_s']:.1f} "
                 f"| {r['ff_sensed_pct']:.0f}% | {r['cnn_sensed_pct']:.0f}% |")
    L += ["", "Notes: positive margin = CNN better. Gap statistics are over runs that found a "
          "route; failed runs are counted in the Failures column and charged a score of 15 s "
          "in Tables 2-3. 'Significant' = the 95% CI of the paired margin excludes zero."]
    with open(md_path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="trained CNN .pt file")
    ap.add_argument("--n", type=int, default=200, help="number of mazes")
    ap.add_argument("--budgets", nargs="*", default=["none", "22", "20", "18", "15", "12"])
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--rows", default="paper_rows.csv")
    ap.add_argument("--prefix", default="paper_results",
                    help="name for the summary files (prefix.md / prefix.csv)")
    ap.add_argument("--summary-only", action="store_true",
                    help="skip the run, just rebuild tables from an existing --rows file")
    a = ap.parse_args()

    if not a.summary_only:
        run_benchmark(a.model, a.n, a.budgets, a.jobs, a.rows)
    summary = summarise(load_rows(a.rows), a.budgets)
    if not summary:
        sys.exit("no paired flood-fill / CNN rows found -- did the CNN load?")
    write_outputs(summary, a.model, a.prefix + ".csv", a.prefix + ".md")
    print(f"\nsaved {a.prefix}.md, {a.prefix}.csv and", a.rows)


if __name__ == "__main__":
    main()
