#!/usr/bin/env bash
# run_official.sh -- CNN vs flood-fill on real competition mazes, for each trained seed.
#
#   bash run_official.sh 2>&1 | tee official_log.txt
#
# 1. downloads the public maze collection (once) into mazefiles/
# 2. for each seed model cnn_outbound_s0/1/2.pt: 142 official mazes x 6 budgets
#                                                   -> rows_official_s0.csv ...
# 3. combines the seeds                            -> official_results.md
#
# Budgets are 56/51/45/38/31 s: the same fractions of full exploration time as the
# random-maze budgets (official mazes take ~2.5x longer to explore).
# Safe to re-run: finished seeds are skipped.
set -euo pipefail
SEEDS=${SEEDS:-"0 1 2"}
JOBS=${JOBS:-$(python3 -c "import os; print(os.cpu_count())")}  # not nproc: it obeys OMP_NUM_THREADS

if [ ! -d mazefiles/classic ]; then
  echo "=== downloading maze files ==="
  # pinned to the upstream commit used for the paper (the collection keeps growing)
  git clone https://github.com/micromouseonline/mazefiles.git
  git -C mazefiles checkout 548dcdeb359bbd3aca997371f8542ec9644840ce
fi

python3 official_mazes.py --dir mazefiles/classic --list > official_maze_list.txt
grep "using" official_maze_list.txt

for s in $SEEDS; do
  if [ ! -f rows_official_s$s.csv ]; then
    echo "=== seed $s ==="
    python3 official_mazes.py --dir mazefiles/classic --model cnn_outbound_s$s.pt \
        --rows rows_official_s$s.csv --jobs $JOBS
  else
    echo "=== rows_official_s$s.csv exists, skipping ==="
  fi
done

python3 seeds_summary.py $(for s in $SEEDS; do echo rows_official_s$s.csv; done) \
    --out official_results.md
