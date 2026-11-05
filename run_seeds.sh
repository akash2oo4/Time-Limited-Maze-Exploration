#!/usr/bin/env bash
# run_seeds.sh -- the whole 3-seed experiment in one go.
#
#   bash run_seeds.sh 2>&1 | tee seeds_log.txt
#
# 1. builds the original outbound-only dataset (20,000 mazes x 2 snapshots) in data_outbound/
# 2. trains the CNN 3 times (seeds 0, 1, 2)        -> cnn_outbound_s0.pt, _s1.pt, _s2.pt
# 3. tests each against flood-fill, 200 mazes x 6 budgets -> rows_seed0.csv ... results_seed0.md ...
# 4. combines the three                             -> seeds_results.md
#
# Safe to re-run: any step whose output file already exists is skipped, so if the laptop
# sleeps or the terminal closes, just run the same command again and it carries on.
set -euo pipefail

MAZES=${MAZES:-20000}     # dataset size
EPOCHS=${EPOCHS:-30}
N=${N:-200}               # test mazes
SEEDS=${SEEDS:-"0 1 2"}
JOBS=${JOBS:-$(python3 -c "import os; print(os.cpu_count())")}  # not nproc: it obeys OMP_NUM_THREADS

if [ ! -f data_outbound/manifest.json ]; then
  echo "=== [1/4] building outbound-only dataset ($MAZES mazes) ==="
  python3 -c "
from snapshot_dataset import build, SnapshotConfig
build(SnapshotConfig(n_mazes=$MAZES, snapshots_per_maze=2, out_dir='data_outbound',
                     outbound_only=True), n_jobs=$JOBS)"
else
  echo "=== [1/4] dataset exists, skipping ==="
fi

for s in $SEEDS; do
  if [ ! -f cnn_outbound_s$s.pt ]; then
    echo "=== [2/4] training seed $s ==="
    python3 cnn_heuristic.py train $EPOCHS --seed $s --data data_outbound --out cnn_outbound_s$s.pt
  else
    echo "=== [2/4] cnn_outbound_s$s.pt exists, skipping ==="
  fi
done

for s in $SEEDS; do
  if [ ! -f results_seed$s.md ]; then
    echo "=== [3/4] testing seed $s ==="
    python3 paper_eval.py --model cnn_outbound_s$s.pt --n $N --jobs $JOBS \
        --rows rows_seed$s.csv --prefix results_seed$s
  else
    echo "=== [3/4] results_seed$s.md exists, skipping ==="
  fi
done

echo "=== [4/4] combining seeds ==="
python3 seeds_summary.py $(for s in $SEEDS; do echo rows_seed$s.csv; done)
