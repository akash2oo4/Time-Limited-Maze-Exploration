#!/usr/bin/env bash
# run_finetune.sh -- 2-fold fine-tuning on official mazes, for each trained seed.
#
#   OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash run_finetune.sh 2>&1 | tee finetune_log.txt
#
# Needs (from earlier runs): mazefiles/, data_outbound/, cnn_outbound_s0/1/2.pt,
#                            rows_official_s0/1/2.csv (the before-fine-tuning results)
# Produces: finetune_results.md  (after)  and  official_results.md (before), same mazes.
# Safe to re-run: finished steps are skipped.
set -euo pipefail
SEEDS=${SEEDS:-"0 1 2"}
JOBS=${JOBS:-$(python3 -c "import os; print(os.cpu_count())")}

python3 finetune_official.py folds --dir mazefiles/classic

for f in A B; do
  if [ ! -f data_off_$f/manifest.json ]; then
    echo "=== snapshots from official fold $f ==="
    python3 finetune_official.py data --dir mazefiles/classic --fold $f --out data_off_$f
  fi
done

for s in $SEEDS; do
  for train in A B; do
    test=$([ $train = A ] && echo B || echo A)
    if [ ! -f cnn_ft_s${s}_on$train.pt ]; then
      echo "=== seed $s: fine-tune on fold $train ==="
      # torch inside one process: let it use all cores here
      OMP_NUM_THREADS=$JOBS MKL_NUM_THREADS=$JOBS python3 finetune_official.py finetune \
          --base cnn_outbound_s$s.pt --data data_off_$train --random data_outbound \
          --out cnn_ft_s${s}_on$train.pt --seed $s
    fi
    if [ ! -f rows_ft_s${s}_$test.csv ]; then
      echo "=== seed $s: test on fold $test (never seen in fine-tuning) ==="
      python3 official_mazes.py --dir mazefiles/classic --model cnn_ft_s${s}_on$train.pt \
          --fold $test --rows rows_ft_s${s}_$test.csv --jobs $JOBS
    fi
  done
  if [ ! -f rows_ft_s$s.csv ]; then        # join the two halves: every maze, unseen
    { cat rows_ft_s${s}_A.csv; tail -n +2 rows_ft_s${s}_B.csv; } > rows_ft_s$s.csv
  fi
done

echo "=== BEFORE fine-tuning (base models, all official mazes) ==="
python3 seeds_summary.py $(for s in $SEEDS; do echo rows_official_s$s.csv; done) --out official_results.md
echo "=== AFTER fine-tuning (each maze tested by a model that never saw it) ==="
python3 seeds_summary.py $(for s in $SEEDS; do echo rows_ft_s$s.csv; done) --out finetune_results.md
