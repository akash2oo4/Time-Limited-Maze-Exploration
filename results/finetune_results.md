# CNN vs flood-fill across 3 training seeds (142 mazes each)

Values: mean over seeds [min - max]. Flood-fill is identical across seeds (same mazes), so its columns are single numbers.

| Budget (s) | Gap closed | CNN failures | FF failures | Score margin (s) | Seeds significantly better |
|---|---|---|---|---|---|
| none | -- | 0.0 [0.0 - 0.0] | 0 | -0.542 [-0.579 - -0.488] | 0 / 3 (worse in 3) |
| 56 | -147% [-173% - -125%] | 0.3 [0.0 - 1.0] | 2 | -0.219 [-0.232 - -0.201] | 0 / 3 (worse in 3) |
| 51 | -37% [-54% - -15%] | 1.0 [0.0 - 2.0] | 2 | -0.158 [-0.220 - -0.078] | 0 / 3 (worse in 2) |
| 45 | -26% [-37% - -18%] | 2.0 [1.0 - 3.0] | 4 | -0.165 [-0.193 - -0.123] | 0 / 3 |
| 38 | -7% [-20% - 3%] | 9.7 [9.0 - 10.0] | 8 | -0.112 [-0.204 - -0.019] | 0 / 3 |
| 31 | -8% [-17% - 3%] | 22.0 [21.0 - 23.0] | 22 | -0.037 [-0.146 - +0.139] | 0 / 3 |

Positive margin = CNN better. 'Seeds significantly better' counts seeds whose paired 95% CI excludes zero in the CNN's favour.
