#!/usr/bin/env bash
# run_contest.sh -- train on contest-style mazes, test on unseen official fold B.
#
#   bash run_contest.sh 2>&1 | tee contest_log.txt
#
# 1. calibrate the contest-style generator on official fold A   -> contest_target.json
# 2. build 20,000-maze dataset (outbound-only, 2 snapshots)      -> data_contest/
# 3. train 3 seeds                                               -> cnn_contest_s0/1/2.pt
# 4. test each on official fold B (69 unseen mazes)              -> rows_contest_s<k>_B.csv
# 5. random-maze check, seed 0 (did it get worse there?)         -> contest_random_s0.md
# 6. compare base / fine-tuned / contest-style on fold B         -> contest_results.md
#
# New files only; nothing existing is changed. Safe to re-run: finished steps are skipped.
set -euo pipefail
SEEDS=${SEEDS:-"0 1 2"}
JOBS=${JOBS:-$(python3 -c "import os; print(os.cpu_count())")}
EPOCHS=${EPOCHS:-30}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1        # one thread per parallel worker

[ -f contest_target.json ] || python3 contest_generator.py calibrate --dir mazefiles/classic --fold A --out contest_target.json

if [ ! -f data_contest/manifest.json ]; then
  echo "=== building contest-style dataset ==="
  python3 contest_dataset.py build --table contest_target.json --n 20000 --out data_contest --jobs $JOBS
fi

for s in $SEEDS; do
  if [ ! -f cnn_contest_s$s.pt ]; then
    echo "=== training seed $s ==="
    # training is one process: let torch use every core here
    OMP_NUM_THREADS=$JOBS MKL_NUM_THREADS=$JOBS python3 cnn_heuristic.py train $EPOCHS \
        --seed $s --data data_contest --out cnn_contest_s$s.pt
  fi
  if [ ! -f rows_contest_s${s}_B.csv ]; then
    echo "=== seed $s: test on official fold B (unseen) ==="
    python3 official_mazes.py --dir mazefiles/classic --model cnn_contest_s$s.pt \
        --fold B --rows rows_contest_s${s}_B.csv --jobs $JOBS
  fi
done

if [ ! -f contest_random_s0.md ]; then
  echo "=== random-maze check, seed 0 (compare with results_seed0.md) ==="
  python3 paper_eval.py --model cnn_contest_s0.pt --n 200 --jobs $JOBS \
      --rows rows_contest_random_s0.csv --prefix contest_random_s0
fi

python3 compare_foldB.py
