# CNN vs flood-fill -- model `cnn_contest_s0.pt`, 200 mazes per budget

## Table 1 -- Route quality (exploration gap = seconds slower than the optimal route)

| Budget (s) | Flood-fill mean | CNN mean | Gap closed | Flood-fill p90 | CNN p90 | Flood-fill median | CNN median | Failures FF / CNN |
|---|---|---|---|---|---|---|---|---|
| none | 0.000 | 0.000 | -- | 0.000 | 0.000 | 0.000 | 0.000 | 0 / 0 |
| 22 | 0.218 | 0.101 | 54% | 0.180 | 0.180 | 0.000 | 0.000 | 0 / 0 |
| 20 | 0.266 | 0.104 | 61% | 0.360 | 0.180 | 0.000 | 0.000 | 0 / 0 |
| 18 | 0.374 | 0.152 | 59% | 0.618 | 0.300 | 0.000 | 0.000 | 0 / 0 |
| 15 | 0.367 | 0.258 | 30% | 1.020 | 0.606 | 0.000 | 0.000 | 5 / 0 |
| 12 | 0.488 | 0.347 | 29% | 1.644 | 1.230 | 0.000 | 0.000 | 13 / 4 |

## Table 2 -- Head-to-head on total score (lower score is better)

| Budget (s) | CNN wins | Flood-fill wins | Draws | Mean margin (s) | 95% CI | Significant |
|---|---|---|---|---|---|---|
| none | 29 (14%) | 151 (76%) | 20 | -0.373 | ±0.055 | yes |
| 22 | 66 (33%) | 113 (56%) | 21 | +0.050 | ±0.133 | no |
| 20 | 70 (35%) | 103 (52%) | 27 | +0.109 | ±0.139 | no |
| 18 | 74 (37%) | 101 (50%) | 25 | +0.185 | ±0.209 | no |
| 15 | 76 (38%) | 96 (48%) | 28 | +0.283 | ±0.220 | yes |
| 12 | 81 (40%) | 79 (40%) | 40 | +0.529 | ±0.289 | yes |

## Table 3 -- Search effort

| Budget (s) | Score FF | Score CNN | Search time FF (s) | Search time CNN (s) | Maze sensed FF | Maze sensed CNN |
|---|---|---|---|---|---|---|
| none | 5.233 | 5.606 | 24.2 | 35.4 | 44% | 42% |
| 22 | 5.302 | 5.253 | 19.7 | 21.7 | 41% | 35% |
| 20 | 5.317 | 5.207 | 18.7 | 20.3 | 40% | 35% |
| 18 | 5.389 | 5.203 | 17.6 | 18.7 | 39% | 34% |
| 15 | 5.513 | 5.229 | 15.7 | 16.3 | 38% | 33% |
| 12 | 5.904 | 5.375 | 13.4 | 13.7 | 35% | 32% |

Notes: positive margin = CNN better. Gap statistics are over runs that found a route; failed runs are counted in the Failures column and charged a score of 15 s in Tables 2-3. 'Significant' = the 95% CI of the paired margin excludes zero.
