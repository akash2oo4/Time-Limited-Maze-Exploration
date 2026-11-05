"""
extract_curves.py -- pull the validation-loss curves (Fig. 7) out of the training logs.

    python3 extract_curves.py seeds_log.txt finetune_log.txt

    seeds_log.txt     written by run_seeds.sh     -> results/curves_training.csv
    finetune_log.txt  written by run_finetune.sh  -> results/curves_finetune.csv

curves_training.csv : seed, epoch, val_time, val_route   (training on generated mazes, 30 epochs)
curves_finetune.csv : seed, fold, epoch, val_time, val_route
                      (fine-tuning on official fold `fold`; epoch 0 is the base model, validated
                       on that fold's official snapshots)
"""

import csv
import os
import re
import sys

EPOCH = re.compile(r"^\s*epoch\s+(\d+)\s+val time\s+([\d.]+)\s+val route\s+([\d.]+)")
BASE = re.compile(r"^\s*base model\s+val time\s+([\d.]+)\s+val route\s+([\d.]+)")
TRAIN_HDR = re.compile(r"training seed (\d+)")
FT_HDR = re.compile(r"seed (\d+): fine-tune on fold ([AB])")


def parse(path, header):
    """Yield (header-groups, epoch, val_time, val_route) for every epoch line in the log."""
    key = None
    for line in open(path, errors="replace"):
        m = header.search(line)
        if m and line.startswith("==="):
            key = m.groups()
            continue
        if key is None:
            continue
        m = BASE.match(line)
        if m:
            yield key, 0, float(m.group(1)), float(m.group(2))
            continue
        m = EPOCH.match(line)
        if m:
            yield key, int(m.group(1)), float(m.group(2)), float(m.group(3))


def main(seeds_log, finetune_log, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "curves_training.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "epoch", "val_time", "val_route"])
        for (seed,), ep, t, r in parse(seeds_log, TRAIN_HDR):
            w.writerow([seed, ep, t, r])
    with open(os.path.join(out_dir, "curves_finetune.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "fold", "epoch", "val_time", "val_route"])
        for (seed, fold), ep, t, r in parse(finetune_log, FT_HDR):
            w.writerow([seed, fold, ep, t, r])
    print(f"wrote {out_dir}/curves_training.csv and {out_dir}/curves_finetune.csv")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(*sys.argv[1:3])
