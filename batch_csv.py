"""
batch_csv.py
Run N random IEEE mazes through every policy and write one CSV row per (maze, policy).

    python3 batch_csv.py 200                       # 200 mazes, default budgets
    python3 batch_csv.py 200 --budgets 18 15 12    # only these budgets
    python3 batch_csv.py 50  --out pilot.csv --jobs 4

Every maze is generated from a recorded seed, and every policy sees the SAME mazes, so
the rows are paired: a difference between two policies on the same seed is a difference
between the policies, not between two draws of maze difficulty. That pairing is what makes
the confidence intervals in the analysis tight enough to be worth reporting.

Row-level output rather than pre-aggregated summaries, because the loss distribution is
heavy-tailed -- medians sit at zero while 90th percentiles reach several seconds. Anything
that reports only means will misdescribe it. Aggregate in a spreadsheet or pandas from
these rows and you can compute medians, quantiles and paired tests yourself.

Failures are written as rows with `reached_route=0`, not dropped. Excluding them flatters
exactly the tight budgets where they happen, and since the classical policies fail more
often than the learned one, dropping them would hide a real difference.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time

import numpy as np

from maze import DIR_DELTA
from belief import fresh_belief, NORTH
from dataset_generator import generate_maze, T_CELL, turn_cost, UNREACHABLE, CELL_M
from solver import belief_passable, truth_passable, time_to_goal, extract_path, oracle_time
from episode import (EpisodeConfig, run_episode, FloodFillPolicy, OptimisticShortcutPolicy)

# Turn geometry -- platform properties. Measure these on your own mouse and cite them;
# the defaults are plausible for a 16x16 competition machine, not measured truth.
TURN_R = 0.05          # m, radius of a smooth 90 deg corner
WHEELBASE_M = 0.07     # m, track width, sets wheel travel during an in-place pivot

NO_ROUTE = 1e5
FAILURE_SCORE = 15.0   # charged when a run ends with no verified route


def build_policies(alpha=2.0, dqn_ckpt=None, extras=False):
    """
    Flood-fill and the learned policy by default -- the head-to-head this script is for.

    `extras` adds the optimistic-shortcut baseline and the oracle upper bound. Both are
    worth having in the paper (the oracle is what turns "0.29 s better" into "closed 66 %
    of what was available"), but they are noise in a two-way comparison.
    """
    pols = [("flood_fill", FloodFillPolicy)]
    try:
        from route_seeking import RouteWeightedPolicy
        pols.append(("cnn_route_weighted", lambda: RouteWeightedPolicy(alpha=alpha)))
    except Exception as e:
        print(f"  skipping cnn_route_weighted: {e}", file=sys.stderr)

    if extras:
        pols.insert(1, ("optimistic_shortcut", OptimisticShortcutPolicy))
        try:
            from oracle_bound import OracleGuidedPolicy
            pols.append(("oracle_guided_BOUND", OracleGuidedPolicy))
        except Exception as e:
            print(f"  skipping oracle bound: {e}", file=sys.stderr)

    if dqn_ckpt:
        try:
            from dqn_agent import DQNExplorationPolicy
            pols.append(("dqn_double", lambda: DQNExplorationPolicy(dqn_ckpt)))
        except Exception as e:
            print(f"  skipping dqn_double: {e}", file=sys.stderr)
    return pols


def path_shape(path):
    """
    Cells, turns by angle, centre-trajectory distance, and pivot wheel travel.

    Distance is NOT cells x pitch. A 90 deg corner is driven as an arc, so rounding it
    replaces two straight legs of TURN_R with a quarter circle and is slightly SHORTER
    than the square path. A 180 deg reversal is a pivot: the centre covers no ground, but
    the wheels roll, and that is reported separately because odometry sees it and the
    robot's displacement does not.
    """
    if len(path) < 2:
        return 0, 0, 0, 0.0, 0.0
    cells = len(path) - 1
    t90 = t180 = 0
    for i in range(2, len(path)):
        d = (path[i][2] - path[i - 1][2]) % 4
        if d in (1, 3):
            t90 += 1
        elif d == 2:
            t180 += 1
    corner_saving = (2 * TURN_R) - (math.pi * TURN_R / 2)
    metres = cells * CELL_M - t90 * corner_saving
    wheel = t180 * (math.pi * WHEELBASE_M / 2)
    return cells, t90, t180, metres, wheel


def one_cell(args):
    """One (seed, policy, budget) row. Top level so multiprocessing can pickle it."""
    seed, pol_name, budget, alpha, dqn_ckpt, extras = args
    random.seed(seed)
    maze = generate_maze()

    pols = dict(build_policies(alpha, dqn_ckpt, extras))
    policy = pols[pol_name]()
    if pol_name == "oracle_guided_BOUND":
        policy.bind(maze)

    r = run_episode(maze.clone(), policy, EpisodeConfig(search_time_budget=budget))
    opt = oracle_time(maze)
    failed = r.fast_run_time >= NO_ROUTE

    fc, f90, f180, fm, fw = path_shape(r.fast_path) if not failed else (0, 0, 0, 0.0, 0.0)

    return {
        "seed": seed,
        "policy": pol_name,
        "budget": "none" if budget is None else budget,
        "reached_goal": int(r.reached_goal),
        "reached_route": int(not failed),
        "oracle_time_s": round(opt, 4),
        "fast_run_s": round(r.fast_run_time, 4) if not failed else "",
        "search_time_s": round(r.search_time, 4),
        "score": round(r.score, 4) if not failed else FAILURE_SCORE,
        "exploration_gap_s": round(r.exploration_gap, 4) if not failed else "",
        "planning_gap_s": round(r.planning_gap, 4) if not failed else "",
        "fast_cells": fc,
        "fast_metres": round(fm, 3),
        "fast_turns_90": f90,
        "fast_turns_180": f180,
        "fast_pivot_wheel_m": round(fw, 3),
        "search_steps": r.search_steps,
        "search_metres": round(r.search_steps * CELL_M, 3),
        "sorties": r.sorties,
        "cells_visited": r.cells_visited,
        "known_fraction": round(r.known_fraction, 4),
    }


FIELDS = ["seed", "policy", "budget", "reached_goal", "reached_route", "oracle_time_s",
          "fast_run_s", "search_time_s", "score", "exploration_gap_s", "planning_gap_s",
          "fast_cells", "fast_metres", "fast_turns_90", "fast_turns_180",
          "fast_pivot_wheel_m", "search_steps", "search_metres", "sorties",
          "cells_visited", "known_fraction"]


def main():
    ap = argparse.ArgumentParser(description="Batch micromouse benchmark -> CSV")
    ap.add_argument("n", type=int, help="number of random mazes")
    ap.add_argument("--budgets", nargs="*", default=["none"],
                    help="search-time budgets in seconds; 'none' means unlimited time. "
                         "Default is unlimited only -- pass e.g. --budgets 18 15 12 to sweep.")
    ap.add_argument("--seed0", type=int, default=0, help="first maze seed")
    ap.add_argument("--alpha", type=float, default=2.0, help="route-weight strength")
    ap.add_argument("--dqn", default=None, help="path to dqn_agent.pt to include the RL arm")
    ap.add_argument("--extras", action="store_true",
                    help="also run optimistic_shortcut and the oracle upper bound")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()

    budgets = [None if b in ("none", "None") else float(b) for b in a.budgets]
    pol_names = [n for n, _ in build_policies(a.alpha, a.dqn, a.extras)]
    print(f"{a.n} mazes x {len(pol_names)} policies x {len(budgets)} budgets "
          f"= {a.n * len(pol_names) * len(budgets)} runs")
    print(f"policies: {', '.join(pol_names)}")

    jobs = [(a.seed0 + i, p, b, a.alpha, a.dqn, a.extras)
            for b in budgets for p in pol_names for i in range(a.n)]

    t0 = time.time()
    if a.jobs > 1:
        import multiprocessing as mp
        from progress import run_with_progress
        with mp.Pool(a.jobs) as pool:
            rows = run_with_progress(pool, one_cell, jobs, chunksize=4)
    else:
        rows = []
        for k, j in enumerate(jobs):
            rows.append(one_cell(j))
            if (k + 1) % 50 == 0:
                print(f"  {k + 1}/{len(jobs)}  ({time.time() - t0:.0f}s)")

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {len(rows)} rows to {a.out}  ({time.time() - t0:.0f}s)")
    summarise(rows, budgets, pol_names)
    head_to_head(rows, budgets, a.out)


def summarise(rows, budgets, pol_names):
    """Console summary. The CSV is the deliverable; this is a sanity check on it."""
    print(f"\n{'budget':>7}  {'policy':<22}{'score':>8}{'gap':>8}{'p90':>8}"
          f"{'fast_m':>9}{'turns':>7}{'known':>7}{'fail':>6}")
    for b in budgets:
        key = "none" if b is None else b
        for p in pol_names:
            sel = [r for r in rows if r["budget"] == key and r["policy"] == p]
            if not sel:
                continue
            gaps = np.array([float(r["exploration_gap_s"]) if r["exploration_gap_s"] != "" else 5.0
                             for r in sel])
            print(f"{str(key):>7}  {p:<22}"
                  f"{np.mean([r['score'] for r in sel]):>8.3f}"
                  f"{gaps.mean():>8.3f}{np.percentile(gaps, 90):>8.3f}"
                  f"{np.mean([r['fast_metres'] for r in sel]):>9.2f}"
                  f"{np.mean([r['fast_turns_90'] + r['fast_turns_180'] for r in sel]):>7.1f}"
                  f"{np.mean([r['known_fraction'] for r in sel]):>7.2f}"
                  f"{sum(1 - r['reached_route'] for r in sel):>6}")
        print()


def head_to_head(rows, budgets, out_path, tol=1e-6):
    """
    Win / loss / draw between flood-fill and the CNN, maze by maze.

    Paired on seed: both policies solved the same maze, so a lower score is a genuine
    win rather than an easier draw. `tol` treats scores within a microsecond as a draw --
    most mazes are easy enough that both policies find the same optimal route, so exact
    ties are common and should not be attributed to either side.

    Also written to <out>_headtohead.csv, one row per maze, so the ties and the margins
    can be inspected rather than taken on trust.
    """
    import csv as _csv
    import os

    for b in budgets:
        key = "none" if b is None else b
        ff = {r["seed"]: r for r in rows
              if r["budget"] == key and r["policy"] == "flood_fill"}
        cn = {r["seed"]: r for r in rows
              if r["budget"] == key and r["policy"] == "cnn_route_weighted"}
        seeds = sorted(set(ff) & set(cn))
        if not seeds:
            continue

        detail, cnn_w, ff_w, draw, margins = [], 0, 0, 0, []
        for sd in seeds:
            a_, b_ = ff[sd], cn[sd]
            d = a_["score"] - b_["score"]          # positive -> CNN faster
            if abs(d) <= tol:
                draw += 1
                who = "draw"
            elif d > 0:
                cnn_w += 1
                who = "cnn"
            else:
                ff_w += 1
                who = "flood_fill"
            margins.append(d)
            detail.append({"seed": sd, "budget": key, "winner": who,
                           "margin_s": round(d, 4),
                           "flood_fill_score": a_["score"], "cnn_score": b_["score"],
                           "flood_fill_gap": a_["exploration_gap_s"],
                           "cnn_gap": b_["exploration_gap_s"],
                           "oracle_time_s": a_["oracle_time_s"]})

        n = len(seeds)
        m = np.array(margins)
        se = m.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        wins = m[m > tol]
        losses = m[m < -tol]

        print(f"\n=== head to head, budget {key} ({n} mazes) ===")
        print(f"  CNN wins        {cnn_w:4d}   ({100 * cnn_w / n:5.1f}%)"
              f"   mean margin {wins.mean() if len(wins) else 0:+.3f}s")
        print(f"  flood-fill wins {ff_w:4d}   ({100 * ff_w / n:5.1f}%)"
              f"   mean margin {losses.mean() if len(losses) else 0:+.3f}s")
        print(f"  draws           {draw:4d}   ({100 * draw / n:5.1f}%)")
        print(f"  overall mean margin {m.mean():+.3f}s  +/- {2 * se:.3f} (95% CI)"
              f"   {'significant' if abs(m.mean()) > 2 * se else 'not significant'}")
        print(f"  median margin {np.median(m):+.3f}s   "
              f"CNN best {m.max():+.3f}s   worst {m.min():+.3f}s")

        base, ext = os.path.splitext(out_path)
        path = f"{base}_headtohead_{key}{ext}"
        with open(path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(detail[0]))
            w.writeheader()
            w.writerows(detail)
        print(f"  per-maze detail -> {path}")

    print("\nDraws are mazes where both policies found the same route. They are common "
          "because\nmost mazes are easy; the interesting cases are in the margins, not "
          "the win count.")


if __name__ == "__main__":
    main()
