"""
make_figures.py -- redraw the paper's figures from the result files in results/.

    pip install matplotlib
    python3 make_figures.py                          # all figures -> figures/*.pdf
    python3 make_figures.py --only random sensing    # a subset
    python3 make_figures.py --maze-dir mazefiles/classic --jobs 8

    figure file          paper    data used
    fig_overview.pdf     Fig. 1   (diagram only)
    fig_sensing.pdf      Fig. 2   results/rows_seed{0,1,2}.csv, results/rows_baselines.csv
    fig_random.pdf       Fig. 3   results/rows_seed{0,1,2}.csv, results/rows_baselines.csv
    fig_routes.pdf       Fig. 4   generates mazes; needs --maze-dir for the official ones
    fig_alpha.pdf        Fig. 5   results/alpha_choices.csv
    fig_official.pdf     Fig. 6   results/rows_official_s*.csv, rows_ft_s*.csv, rows_contest_s*_B.csv
    fig_curves.pdf       Fig. 7   results/curves_training.csv, results/curves_finetune.csv
    fig_mazes.pdf        Fig. 8   generates mazes; needs --maze-dir for the All-Japan one

rows_baselines.csv comes from baselines_eval.py; the curves_*.csv files from extract_curves.py.
Figures 4 and 8 draw their own sample of mazes (fixed seeds), so they show the same kind of
picture as the paper but not necessarily the identical mazes.
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------------------
# style
# ---------------------------------------------------------------------------
ORANGE, GRAY, BLUE, GREEN = "#e8683a", "#8a8a8a", "#2f6fd0", "#1fa98a"
LIGHT_BLUE, LIGHT_GRAY = "#e9f0fc", "#f2f1ee"

plt.rcParams.update({
    "font.family": "serif", "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": "#e4e4e4", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "legend.frameon": False,
})

SEEDS = [0, 1, 2]
BUDGETS = ["none", "22", "20", "18", "15", "12"]          # generated mazes
BUDGET_LABELS = ["no limit", "91%", "83%", "74%", "62%", "50%"]       # share of flood-fill's search
OFF_BUDGETS = ["none", "56", "51", "45", "38", "31"]                 # official mazes
OFF_LABELS = ["no limit", "56 s\n91%", "51 s\n83%", "45 s\n74%", "38 s\n62%", "31 s\n50%"]
FAIL_SCORE = 15.0


def rows_of(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def norm_budget(b):
    return b if b == "none" else f"{float(b):g}"


def by_key(rows, group, value):
    """{(policy, budget): np.array of `value`, ordered by maze} -- every policy sees the same mazes."""
    d = defaultdict(list)
    for r in sorted(rows, key=lambda r: (r.get("maze") or "", int(r["seed"]) if r.get("seed") else 0)):
        d[(r["policy"], norm_budget(r["budget"]))].append(float(r[value]) if r[value] != "" else np.nan)
    return {k: np.array(v) for k, v in d.items()}


def save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path)
    plt.close(fig)
    print("  wrote", path)


# ---------------------------------------------------------------------------
# generated-maze data (Figs 2, 3)
# ---------------------------------------------------------------------------
def load_generated(res):
    """per-policy, per-budget arrays: score, failed (0/1), known fraction."""
    out = {}
    cnn = [rows_of(os.path.join(res, f"rows_seed{s}.csv")) for s in SEEDS]
    base = rows_of(os.path.join(res, "rows_baselines.csv"))
    ff_rows = [r for r in cnn[0] if r["policy"] == "flood_fill"]
    sets = {"flood_fill": ff_rows,
            "optimistic_shortcut": [r for r in base if r["policy"] == "optimistic_shortcut"],
            "oracle_guided_BOUND": [r for r in base if r["policy"] == "oracle_guided_BOUND"]}
    for pol, rows in sets.items():
        for b in BUDGETS:
            sel = [r for r in rows if norm_budget(r["budget"]) == b]
            out[(pol, b)] = {
                "score": np.array([float(r["score"]) for r in sel]),
                "fail": np.array([r["reached_route"] == "0" for r in sel], float),
                "known": np.array([float(r["known_fraction"]) for r in sel]),
            }
    out["cnn"] = []
    for rows in cnn:
        d = {}
        for b in BUDGETS:
            sel = [r for r in rows if r["policy"] == "cnn_route_weighted" and norm_budget(r["budget"]) == b]
            d[b] = {"score": np.array([float(r["score"]) for r in sel]),
                    "fail": np.array([r["reached_route"] == "0" for r in sel], float),
                    "known": np.array([float(r["known_fraction"]) for r in sel])}
        out["cnn"].append(d)
    return out


# ---------------------------------------------------------------------------
# Fig 1 -- overview diagram
# ---------------------------------------------------------------------------
def fig_overview(a):
    fig, ax = plt.subplots(figsize=(2.75, 1.39))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")
    ax.grid(False)
    w, h = 21, 11
    xs = [1, 26, 51, 76]
    y_top, y_bot = 36, 12

    def box(x, y, text, fc=LIGHT_GRAY, ec="#444", lw=0.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                    fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=4.3, linespacing=1.15)

    def arrow(p, q, **kw):
        ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#444",
                                                       shrinkA=0, shrinkB=0, mutation_scale=5, **kw))

    box(xs[0], y_top, "Sensors\n(L, R, 2 ahead)")
    box(xs[1], y_top, "Belief map\nopen / wall /\nunknown")
    box(xs[2], y_top, "Dilated CNN\nroute prob. $p_c$", fc=LIGHT_BLUE, ec=BLUE, lw=0.9)
    box(xs[3], y_top, "Edge cost\nraised where\n$p_c$ is low")
    box(xs[0], y_bot, "Fast run on\nverified\nopenings only")
    box(xs[1], y_bot, "Sortie done?\nverified route\nor budget out")
    box(xs[2], y_bot, "Drive one cell")
    box(xs[3], y_bot, "Dijkstra over\n(cell, heading)\n$\\rightarrow$ next move")
    for i in range(3):
        arrow((xs[i] + w + 0.4, y_top + h / 2), (xs[i + 1] - 0.4, y_top + h / 2))
        arrow((xs[i + 1] - 0.4, y_bot + h / 2), (xs[i] + w + 0.4, y_bot + h / 2))
    arrow((xs[3] + w / 2, y_top - 0.4), (xs[3] + w / 2, y_bot + h + 0.4))
    arrow((xs[2] + 3, y_bot + h + 0.4), (xs[0] + w - 3, y_top - 0.4))
    ax.text(50, 28.8, "repeat: sense new walls", ha="center", va="center", fontsize=4.3, color="#444")
    ax.text(50, 3.6, "If a whole sortie reveals no new wall, the remaining sorties\n"
                     "use flood-fill (stall fallback). Score = $T_{fast}$ + $T_{search}$/30.",
            ha="center", va="center", fontsize=4.6, color="#555", linespacing=1.3)
    save(fig, a.out, "fig_overview.pdf")


# ---------------------------------------------------------------------------
# Fig 2 -- sensing, Fig 3 -- total score and failures
# ---------------------------------------------------------------------------
def fig_sensing(a, data):
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    x = np.arange(len(BUDGETS))
    series = [
        ("Flood-fill", ORANGE, "o", "-", [100 * data[("flood_fill", b)]["known"].mean() for b in BUDGETS]),
        ("Optimistic shortcut", GRAY, "s", "--", [100 * data[("optimistic_shortcut", b)]["known"].mean() for b in BUDGETS]),
        ("Learned (CNN), 3 seeds", BLUE, "^", "-", [100 * np.mean([d[b]["known"].mean() for d in data["cnn"]]) for b in BUDGETS]),
        ("Oracle bound", GREEN, "D", ":", [100 * data[("oracle_guided_BOUND", b)]["known"].mean() for b in BUDGETS]),
    ]
    for label, c, m, ls, y in series:
        ax.plot(x, y, color=c, marker=m, ls=ls, lw=1.4, ms=3.5, label=label)
    ax.set_xticks(x, BUDGET_LABELS)
    ax.set_ylim(15, 50)
    ax.set_ylabel("maze sensed (%)")
    ax.set_xlabel("search budget")
    ax.legend(ncol=2, loc="lower left", handlelength=2.2, columnspacing=1.2)
    save(fig, a.out, "fig_sensing.pdf")


def fig_random(a, data):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.16, 2.2), gridspec_kw={"width_ratios": [1.15, 1]})
    x = np.arange(len(BUDGETS))
    mean = lambda pol: np.array([data[(pol, b)]["score"].mean() for b in BUDGETS])
    cnn = np.array([[d[b]["score"].mean() for b in BUDGETS] for d in data["cnn"]])
    ax.plot(x, mean("flood_fill"), color=ORANGE, marker="o", lw=1.6, ms=4, label="Flood-fill (turn-weighted)")
    ax.plot(x, mean("optimistic_shortcut"), color=GRAY, marker="s", ls="--", lw=1.2, ms=3.5, label="Optimistic shortcut")
    ax.fill_between(x, cnn.min(0), cnn.max(0), color=BLUE, alpha=0.18, lw=0)
    ax.plot(x, cnn.mean(0), color=BLUE, marker="^", lw=1.6, ms=4.5, label="Learned (CNN), 3 seeds")
    ax.plot(x, mean("oracle_guided_BOUND"), color=GREEN, marker="D", ls=":", lw=1.4, ms=3.5, label="Oracle bound")
    ax.set_xticks(x, BUDGET_LABELS)
    ax.set_xlabel("search budget (share of flood-fill's full exploration time)")
    ax.set_ylabel("mean score (s), lower is better")
    ax.set_title("(a) Total score, 200 generated mazes")
    ax.legend(ncol=2, loc="upper left", handlelength=2.2, columnspacing=1.0)

    fails = lambda pol: np.array([data[(pol, b)]["fail"].sum() for b in BUDGETS])
    cf = np.array([[d[b]["fail"].sum() for b in BUDGETS] for d in data["cnn"]])
    wd = 0.26
    ff, op, cm = fails("flood_fill"), fails("optimistic_shortcut"), cf.mean(0)
    bx.bar(x - wd, ff, wd, color=ORANGE, label="Flood-fill")
    bx.bar(x, op, wd, color=GRAY, label="Optimistic")
    bx.bar(x + wd, cm, wd, color=BLUE, label="CNN (mean)")
    bx.errorbar(x + wd, cm, yerr=[cm - cf.min(0), cf.max(0) - cm], fmt="none", ecolor="k", elinewidth=0.8, capsize=2)
    for xi, v in zip(x - wd, ff):
        if v:
            bx.text(xi, v + 0.2, f"{v:g}", ha="center", va="bottom", fontsize=5.5)
    for xi, v in zip(x + wd, cm):
        if v:
            bx.text(xi + 0.1, v + 0.2, f"{v:.1f}", ha="left", va="bottom", fontsize=5.5)
    bx.set_xticks(x, BUDGET_LABELS)
    bx.set_xlabel("search budget")
    bx.set_ylabel("failed runs (of 200)")
    bx.set_title("(b) Runs ending with no verified route")
    bx.legend(loc="upper left", handlelength=1.2)
    bx.grid(axis="x", visible=False)
    save(fig, a.out, "fig_random.pdf")


# ---------------------------------------------------------------------------
# Fig 5 -- coupling strength selection
# ---------------------------------------------------------------------------
def fig_alpha(a):
    rows = rows_of(os.path.join(a.results, "alpha_choices.csv"))
    alphas = [0.5, 1, 1.5, 2, 3]
    fig, ax = plt.subplots(figsize=(3.5, 1.6))
    for r in rows:
        ref = float(r["score_alpha_2"])
        y = [float(r[f"score_alpha_{al:g}"]) - ref for al in alphas]
        ax.plot(alphas, y, color=BLUE, marker="o", lw=1.0, ms=2.8, alpha=0.85)
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set_xticks(alphas, ["0.5", "1.0", "1.5", "2.0", "3.0"])
    ax.set_xlabel(r"coupling strength $\alpha$")
    ax.set_ylabel(r"score vs $\alpha=2$ (s)")
    save(fig, a.out, "fig_alpha.pdf")


# ---------------------------------------------------------------------------
# Fig 6 -- official fold B, three ways of training
# ---------------------------------------------------------------------------
def margins(path, fold_b):
    """{budget: mean over fold-B mazes of (flood-fill score - CNN score)}"""
    rows = [r for r in rows_of(path) if r["maze"] in fold_b]
    s = {(r["policy"], norm_budget(r["budget"]), r["maze"]): float(r["score"]) for r in rows}
    out = {}
    for b in OFF_BUDGETS:
        mz = sorted({k[2] for k in s if k[1] == b})
        out[b] = np.mean([s[("flood_fill", b, m)] - s[("cnn_route_weighted", b, m)] for m in mz])
    return out


def fig_official(a):
    res = a.results
    fold_b = {r["maze"] for r in rows_of(os.path.join(res, "rows_contest_s0_B.csv"))}
    methods = [("Trained on generated mazes", GRAY, "rows_official_s{}.csv"),
               ("+ fine-tuned on official fold A", ORANGE, "rows_ft_s{}.csv"),
               ("Trained on contest-style mazes", BLUE, "rows_contest_s{}_B.csv")]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    x = np.arange(len(OFF_BUDGETS))
    wd = 0.26
    for i, (label, c, pat) in enumerate(methods):
        per = [margins(os.path.join(res, pat.format(s)), fold_b) for s in SEEDS]
        m = np.array([[p[b] for b in OFF_BUDGETS] for p in per])
        ax.bar(x + (i - 1) * wd, m.mean(0), wd, color=c, label=label)
        ax.errorbar(x + (i - 1) * wd, m.mean(0), yerr=[m.mean(0) - m.min(0), m.max(0) - m.mean(0)],
                    fmt="none", ecolor="#222", elinewidth=0.7, capsize=1.6)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, OFF_LABELS)
    ax.set_ylabel("score margin (s); > 0 = CNN better")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, handlelength=1.2)
    save(fig, a.out, "fig_official.pdf")


# ---------------------------------------------------------------------------
# Fig 7 -- validation curves
# ---------------------------------------------------------------------------
def fig_curves(a):
    tr = rows_of(os.path.join(a.results, "curves_training.csv"))
    ft = rows_of(os.path.join(a.results, "curves_finetune.csv"))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(3.5, 1.75))
    for s in sorted({r["seed"] for r in tr}):
        rr = [r for r in tr if r["seed"] == s]
        ep = [int(r["epoch"]) for r in rr]
        ax.plot(ep, [float(r["val_time"]) for r in rr], color=ORANGE, lw=0.9, alpha=0.8)
        ax.plot(ep, [float(r["val_route"]) for r in rr], color=BLUE, lw=0.9, alpha=0.8)
    ax.plot([], [], color=BLUE, lw=0.9, label="route")
    ax.plot([], [], color=ORANGE, lw=0.9, label="time-to-goal")
    ax.legend(loc="center right", handlelength=1.4, bbox_to_anchor=(1.0, 0.58))
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation loss")
    ax.set_title("(a) training, 3 seeds")
    runs = sorted({(r["seed"], r["fold"]) for r in ft})
    for s, f in runs:
        rr = [r for r in ft if r["seed"] == s and r["fold"] == f]
        bx.plot([int(r["epoch"]) for r in rr], [float(r["val_route"]) for r in rr], color=BLUE, lw=0.9, alpha=0.8)
    level = np.mean([float(r["val_route"]) for r in tr if int(r["epoch"]) == 30])
    bx.axhline(level, color="#444", ls="--", lw=0.8)
    bx.text(10, level + 0.35, "generated-maze level", ha="right", va="bottom", fontsize=5.5, color="#555")
    bx.set_xlabel("epoch (0 = base model)")
    bx.set_ylabel("route loss, official val")
    bx.set_title("(b) fine-tuning, %d runs" % len(runs))
    save(fig, a.out, "fig_curves.pdf")


# ---------------------------------------------------------------------------
# Figs 4 and 8 -- maze samples
# ---------------------------------------------------------------------------
def _route(m):
    from belief import NORTH
    from solver import extract_path, time_to_goal, truth_passable
    p = truth_passable(m)
    goals = sorted(m.goal_indices)
    _, path = extract_path(time_to_goal(p, goals), p, (m.start_x, m.start_y), NORTH, goals)
    return path


def _len_generated(seed):
    from dataset_generator import generate_maze
    random.seed(seed)
    return len(_route(generate_maze())) - 1


def _len_contest(seed):
    import contest_generator as cg
    random.seed(seed)
    return len(_route(cg.generate_contest_maze())) - 1


def _len_official(path):
    from official_mazes import load_maze_file
    return len(_route(load_maze_file(path))) - 1


def _init_table(path):
    """Load the contest-style calibration table (also inside every worker process)."""
    import contest_generator as cg
    cg.TABLE = json.load(open(path))
    return cg


def _load_contest_table(a):
    return _init_table(os.path.join(a.results, "contest_target.json"))


def fig_routes(a):
    from multiprocessing import Pool
    from official_mazes import select
    cg = _load_contest_table(a)
    names = [os.path.join(a.maze_dir, n) for n, _ in select(a.maze_dir, verbose=False)]
    with Pool(a.jobs, initializer=_init_table,
              initargs=(os.path.join(a.results, "contest_target.json"),)) as pool:
        gen = pool.map(_len_generated, range(10_000_000, 10_000_400))
        con = pool.map(_len_contest, range(70_000_000, 70_000_250))
        off = pool.map(_len_official, names)
    edges = np.arange(5, 146, 10)
    fig, ax = plt.subplots(figsize=(3.5, 1.8))
    for data, c, lw, label in [(gen, GRAY, 1.6, f"generated (mean {np.mean(gen):.0f})"),
                               (con, BLUE, 1.4, f"contest-style (mean {np.mean(con):.0f})"),
                               (off, ORANGE, 1.8, f"official, {len(off)} (mean {np.mean(off):.0f})")]:
        h = 100 * np.histogram(data, bins=edges)[0] / len(data)
        ax.stairs(h, edges, color=c, lw=lw, label=label)
    ax.set_xlim(0, 145)
    ax.set_xlabel("time-optimal route length (cells)")
    ax.set_ylabel("share of mazes (%)")
    ax.legend(loc="upper right", handlelength=2.2)
    save(fig, a.out, "fig_routes.pdf")


def draw_maze(ax, m, path, title):
    from maze import WALL_EAST, WALL_NORTH, WALL_SOUTH, WALL_WEST
    n = m.width
    for g in m.goals:
        ax.add_patch(plt.Rectangle((g["x"], g["y"]), 1, 1, fc="#f7d9c8", ec="none", zorder=0))
    for y in range(n):
        for x in range(n):
            for flag, (x0, y0, x1, y1) in ((WALL_NORTH, (x, y, x + 1, y)), (WALL_SOUTH, (x, y + 1, x + 1, y + 1)),
                                           (WALL_WEST, (x, y, x, y + 1)), (WALL_EAST, (x + 1, y, x + 1, y + 1))):
                if m.has_wall(x, y, flag):
                    ax.plot([x0, x1], [y0, y1], color="k", lw=0.7, solid_capstyle="projecting")
    ax.plot([p[0] + 0.5 for p in path], [p[1] + 0.5 for p in path], color=BLUE, lw=1.5, alpha=0.95,
            solid_joinstyle="round", zorder=3)
    ax.set_xlim(-0.2, n + 0.2)
    ax.set_ylim(n + 0.2, -0.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.grid(False)
    ax.set_title(title, fontsize=6.5, linespacing=1.3)


def find_seed(fn, start, want, limit=20000):
    for s in range(start, start + limit):
        if fn(s) == want:
            return s
    raise RuntimeError(f"no maze with a {want}-cell route in {limit} seeds from {start}")


def fig_mazes(a):
    from dataset_generator import generate_maze
    from official_mazes import load_maze_file
    cg = _load_contest_table(a)
    sg = find_seed(_len_generated, 0, 25)
    sc = find_seed(_len_contest, 70_000_000, 53)
    random.seed(sg)
    mg = generate_maze()
    random.seed(sc)
    mc = cg.generate_contest_maze()
    mo = load_maze_file(os.path.join(a.maze_dir, "alljapan-045-2024-exp-fin.txt"))
    fig, axes = plt.subplots(1, 3, figsize=(3.5, 1.75))
    for ax, m, t in zip(axes, (mg, mc, mo), ("(a) generated", "(b) contest-style", "(c) All-Japan 2024")):
        path = _route(m)
        draw_maze(ax, m, path, f"{t}\nroute {len(path) - 1} cells")
    fig.subplots_adjust(wspace=0.05)
    save(fig, a.out, "fig_mazes.pdf")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--maze-dir", default="mazefiles/classic",
                    help="official maze files (only needed for the routes and mazes figures)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--only", nargs="*", choices=["overview", "sensing", "random", "routes", "alpha",
                                                  "official", "curves", "mazes"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    want = set(a.only or ["overview", "sensing", "random", "routes", "alpha", "official", "curves", "mazes"])
    need_mazes = {"routes", "mazes"} & want
    if need_mazes and not os.path.isdir(a.maze_dir):
        print(f"skipping {', '.join(sorted(need_mazes))}: {a.maze_dir} not found "
              f"(run run_official.sh first or pass --maze-dir)")
        want -= need_mazes
    data = load_generated(a.results) if {"sensing", "random"} & want else None
    if "overview" in want:
        fig_overview(a)
    if "sensing" in want:
        fig_sensing(a, data)
    if "random" in want:
        fig_random(a, data)
    if "routes" in want:
        fig_routes(a)
    if "alpha" in want:
        fig_alpha(a)
    if "official" in want:
        fig_official(a)
    if "curves" in want:
        fig_curves(a)
    if "mazes" in want:
        fig_mazes(a)


if __name__ == "__main__":
    main()
