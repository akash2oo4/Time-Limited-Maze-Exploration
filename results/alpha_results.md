# CNN vs flood-fill across 3 training seeds (142 mazes each)

Values: mean over seeds [min - max]. Flood-fill is identical across seeds (same mazes), so its columns are single numbers.

| Budget (s) | Gap closed | CNN failures | FF failures | Score margin (s) | Seeds significantly better |
|---|---|---|---|---|---|
| none | -- | 0.0 [0.0 - 0.0] | 0 | -0.299 [-0.340 - -0.257] | 0 / 3 (worse in 3) |
| 56 | -68% [-102% - -42%] | 0.3 [0.0 - 1.0] | 2 | -0.130 [-0.185 - -0.091] | 0 / 3 (worse in 1) |
| 51 | -20% [-28% - -4%] | 0.7 [0.0 - 1.0] | 2 | -0.092 [-0.133 - -0.052] | 0 / 3 |
| 45 | -17% [-24% - -8%] | 1.7 [1.0 - 2.0] | 4 | -0.051 [-0.077 - -0.035] | 0 / 3 |
| 38 | -4% [-9% - 4%] | 9.0 [8.0 - 10.0] | 8 | +0.013 [-0.025 - +0.074] | 0 / 3 |
| 31 | -7% [-18% - 1%] | 20.7 [19.0 - 22.0] | 22 | +0.016 [-0.088 - +0.203] | 0 / 3 |

Positive margin = CNN better. 'Seeds significantly better' counts seeds whose paired 95% CI excludes zero in the CNN's favour.
