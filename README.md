# Time-Limited Maze Exploration with a Learned Route Map

Code, trained models and result files for the paper

> **System And Method For The Time-Limited Maze Exploration By an Autonomous Robot Using A Learned Route Map**
> Dr. Ulagammai M, Akash S, Anou Zecharia W B, Gokul S — SRM Institute of Science and Technology, Chennai.

A micromouse must explore an unknown 16×16 maze before its timed fast run, and in competition the
exploration time is limited. The paper asks when a convolutional network that predicts, from a partial wall map,
which cells lie on the time-optimal route can make exploration better than turn-weighted flood-fill. It reports
positive results on generated mazes, a negative result on 142 official contest mazes, and three remedies.

Everything here is a pure-Python simulator (hardware-matched sensor and timing model, Dijkstra over
(cell, heading) states) plus a small PyTorch network. It runs on a CPU-only laptop.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 test_belief.py          # quick sanity check, should print ALL TESTS PASSED
```

## Repository layout

The scripts import each other by module name and read/write files in the current directory, so run them
from the repository root.

| Group | Files | What it does |
|---|---|---|
| Simulator | `maze.py`, `maze_generator.py`, `belief.py`, `dataset_generator.py` | Maze representation, the spanning-tree-plus-loops generator (Section III), the partial wall map (belief) and the 11-plane network input |
| Planner & episode | `solver.py`, `episode.py` | Dijkstra over (cell, heading), turn/cell timing, the sortie loop, scoring `S = T_fast + T_search/30` |
| Policies | `route_seeking.py`, `oracle_bound.py` | The learned route-weighted policy with stall fallback (Section IV), the oracle upper bound |
| Network | `cnn_heuristic.py`, `snapshot_dataset.py` | Dilated residual CNN (229,189 parameters), snapshot dataset builder, training (`train` sub-command) |
| Evaluation | `batch_csv.py`, `paper_eval.py`, `seeds_summary.py`, `baselines_eval.py`, `progress.py` | Paired benchmark over budgets, tables and 95% intervals, multi-seed summary, optimistic/oracle reference policies |
| Figures | `make_figures.py`, `extract_curves.py` | Redraw the paper's figures into `figures/` from `results/` |
| Official mazes | `official_mazes.py`, `maze_stats.py` | Filtering and running the public contest mazes (Section V), maze statistics (Table III) |
| Remedies | `finetune_official.py`, `alpha_cv.py`, `contest_generator.py`, `contest_dataset.py`, `compare_foldB.py` | Fine-tuning on half the official mazes, coupling-strength re-tuning, contest-style generator, comparison on the unseen half (Section VI-C) |
| Pipelines | `run_seeds.sh`, `run_official.sh`, `run_finetune.sh`, `run_contest.sh` | The whole experiment, stage by stage |
| Tests | `test_belief.py` | Belief-map and input-tensor checks |
| `models/` | 13 `.pt` files | Trained networks (see below) |
| `results/` | `.md` / `.csv` / `.json` | Per-maze rows and summary tables behind the paper's numbers |

## Reproducing the paper

Each script is safe to re-run: finished steps are skipped. Run them in this order.

```bash
bash run_seeds.sh    2>&1 | tee seeds_log.txt      # Section VI-A: build 20,000-maze dataset, train 3 seeds, test on 200 generated mazes
bash run_official.sh 2>&1 | tee official_log.txt    # Section VI-B: 142 official mazes (downloads the maze collection)
bash run_finetune.sh 2>&1 | tee finetune_log.txt    # Section VI-C: fine-tuning on one half, tested on the other
python3 alpha_cv.py                                  # Section VI-C: re-tune coupling strength alpha
python3 seeds_summary.py rows_alpha_s0.csv rows_alpha_s1.csv rows_alpha_s2.csv --out alpha_results.md
bash run_contest.sh  2>&1 | tee contest_log.txt     # Section VI-C: contest-style generator
python3 maze_stats.py --dir mazefiles/classic --n 300   # Table III: official vs generated maze statistics
```

Training uses seeds 0, 1, 2 and takes roughly half an hour per seed on a laptop CPU; the evaluation of 200 mazes
× 6 budgets takes about 50 minutes on 12 CPU workers.

### Using the shipped models and results instead of retraining

`models/` and `results/` are kept out of the working directory on purpose: the run scripts skip any step whose
output file already exists, so shipping them in the root would silently skip training. To reuse them, copy what
you need into the repository root, for example:

```bash
cp models/cnn_outbound_s*.pt .                      # skip training, go straight to testing
python3 seeds_summary.py results/rows_seed0.csv results/rows_seed1.csv results/rows_seed2.csv   # Table II
```

| Paper item | Model(s) in `models/` | Result files in `results/` |
|---|---|---|
| Table II, Figs 2, 3, 7a (generated mazes, 3 seeds) | `cnn_outbound_s{0,1,2}.pt` | `rows_seed{0,1,2}.csv`, `results_seed{0,1,2}.md`, `seeds_results.md`, `rows_baselines.csv`, `curves_training.csv` |
| Official mazes, base networks (Section VI-B, Table IV "Base") | `cnn_outbound_s{0,1,2}.pt` | `rows_official_s{0,1,2}.csv`, `official_results.md`, `official_maze_list.txt` |
| Fine-tuning (Section VI-C, Fig 7b, Table IV "Fine-tuned") | `cnn_ft_s{k}_on{A,B}.pt` (trained on half A / half B) | `rows_ft_s{0,1,2}.csv`, `finetune_results.md`, `curves_finetune.csv` |
| Coupling re-tuning (Fig 5, Table IV "α re-tuned") | same fine-tuned networks | `rows_alpha_s{0,1,2}.csv`, `alpha_choices.csv`, `alpha_results.md` |
| Contest-style generator (Fig 6, Table IV "Contest-style") | `cnn_contest_s{0,1,2}.pt` | `rows_contest_s{0,1,2}_B.csv`, `contest_target.json`, `contest_results.md`, `rows_contest_random_s0.csv`, `contest_random_s0.md` |
| Return-leg training data (Section VI-D, one seed) | `cnn_return_leg_s0.pt` | `rows_return_leg_s0.csv`, `results_return_leg.md` |

Notes on the `results/` files: `results_return_leg.md` was written when the network file was still called
`cnn_heuristic_sortie.pt` (now `models/cnn_return_leg_s0.pt`). Each `rows_*.csv` has one row per
(maze seed, policy, budget); `seeds_summary.py` and `paper_eval.py` turn them into the tables. The evaluation
scripts temporarily copy the model under test to `cnn_heuristic.pt` in the working directory (an existing file
is backed up to `cnn_heuristic.pt.bak` and restored).

## Redrawing the figures

`make_figures.py` redraws all eight figures of the paper from `results/` and writes them to `figures/`
(the PDFs are included). It needs `matplotlib` in addition to `requirements.txt`.

```bash
pip install matplotlib
python3 make_figures.py                           # Figs 1-3, 5-7 need nothing but results/
python3 make_figures.py --maze-dir mazefiles/classic   # Figs 4 and 8 also need the official mazes
```

Figs 4 and 8 draw their own samples of generated and contest-style mazes with fixed seeds, so they show the
same kind of picture as in the paper (route lengths 25 / 53 / 62 cells in Fig. 8) but not necessarily the
identical mazes, and the generated-maze histogram in Fig. 4 differs slightly in its bins. The training curves
in `results/curves_*.csv` were extracted from the run logs with `extract_curves.py`.

## Official maze data

The 142 competition mazes come from the public collection
[micromouseonline/mazefiles](https://github.com/micromouseonline/mazefiles) (reference [19] in the paper). It is
not redistributed here. `run_official.sh` clones it and checks out upstream commit
`548dcdeb359bbd3aca997371f8542ec9644840ce`, the version used for the paper, so that the same 142 mazes are
selected (`results/official_maze_list.txt` lists which files were kept and why others were skipped).

## Notes

- `batch_csv.py` contains an optional `--dqn` branch that imports a `dqn_agent` module. That reinforcement-learning
  arm was an abandoned experiment, is not part of the paper and is not included; the code path is never taken
  unless `--dqn` is passed.
- `results/rows_baselines.csv` (optimistic-shortcut and oracle policies, Figs 2–3) was regenerated with
  `baselines_eval.py` for this release, because the original rows for those two reference policies had not been
  kept. It reproduces the paper's values (optimistic failures 0/0/1/1/5/10 across the six budgets, oracle 0).

## Citation

If you use this code, please cite the paper above.
