# CNN vs flood-fill across 3 training seeds (200 mazes each)

Values: mean over seeds [min - max]. Flood-fill is identical across seeds (same mazes), so its columns are single numbers.

| Budget (s) | Gap closed | CNN failures | FF failures | Score margin (s) | Seeds significantly better |
|---|---|---|---|---|---|
| none | -- | 0.0 [0.0 - 0.0] | 0 | -0.362 [-0.373 - -0.351] | 0 / 3 (worse in 3) |
| 22 | 75% [72% - 78%] | 0.0 [0.0 - 0.0] | 0 | +0.108 [+0.103 - +0.114] | 0 / 3 |
| 20 | 74% [72% - 77%] | 0.0 [0.0 - 0.0] | 0 | +0.155 [+0.151 - +0.163] | 3 / 3 |
| 18 | 69% [65% - 75%] | 0.0 [0.0 - 0.0] | 0 | +0.228 [+0.215 - +0.254] | 2 / 3 |
| 15 | 42% [41% - 43%] | 2.7 [2.0 - 3.0] | 5 | +0.239 [+0.226 - +0.258] | 2 / 3 |
| 12 | 38% [36% - 40%] | 6.7 [6.0 - 8.0] | 13 | +0.480 [+0.407 - +0.519] | 3 / 3 |

Positive margin = CNN better. 'Seeds significantly better' counts seeds whose paired 95% CI excludes zero in the CNN's favour.
