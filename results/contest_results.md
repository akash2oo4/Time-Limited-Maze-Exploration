# Official fold B (69 mazes, unseen by every method): CNN vs flood-fill

Score margin = flood-fill score - CNN score, mean over 3 seeds [min - max]. Positive = CNN better. Sig = seeds whose paired 95% CI excludes zero (better / worse).

| Budget (s) | base (random mazes) margin | sig | fine-tuned on fold A margin | sig | contest-style training margin | sig |
|---|---|---|---|---|---|---|
| none | -1.135 [-1.200 - -1.026] | 0/3 | -0.551 [-0.632 - -0.481] | 0/3 | -0.981 [-1.030 - -0.926] | 0/3 |
| 56 | -0.360 [-0.420 - -0.319] | 0/3 | -0.145 [-0.161 - -0.124] | 0/3 | -0.252 [-0.291 - -0.223] | 0/3 |
| 51 | -0.296 [-0.330 - -0.249] | 0/3 | -0.078 [-0.149 - +0.057] | 0/1 | -0.101 [-0.149 - -0.061] | 0/1 |
| 45 | -0.396 [-0.515 - -0.202] | 0/3 | -0.047 [-0.133 - +0.070] | 0/0 | -0.055 [-0.132 - +0.023] | 0/0 |
| 38 | -0.315 [-0.466 - -0.204] | 0/2 | -0.037 [-0.181 - +0.191] | 0/1 | -0.017 [-0.083 - +0.037] | 0/0 |
| 31 | -0.413 [-0.503 - -0.318] | 0/0 | -0.225 [-0.375 - -0.056] | 0/0 | +0.025 [-0.170 - +0.274] | 0/0 |

69 mazes per budget per seed.
