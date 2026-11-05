# CNN vs flood-fill -- model `cnn_outbound_s1.pt`, 200 mazes per budget

## Table 1 -- Route quality (exploration gap = seconds slower than the optimal route)

| Budget (s) | Flood-fill mean | CNN mean | Gap closed | Flood-fill p90 | CNN p90 | Flood-fill median | CNN median | Failures FF / CNN |
|---|---|---|---|---|---|---|---|---|
| none | 0.000 | 0.000 | -- | 0.000 | 0.000 | 0.000 | 0.000 | 0 / 0 |
| 22 | 0.218 | 0.062 | 72% | 0.180 | 0.000 | 0.000 | 0.000 | 0 / 0 |
| 20 | 0.266 | 0.062 | 77% | 0.360 | 0.012 | 0.000 | 0.000 | 0 / 0 |
| 18 | 0.374 | 0.132 | 65% | 0.618 | 0.180 | 0.000 | 0.000 | 0 / 0 |
| 15 | 0.367 | 0.209 | 43% | 1.020 | 0.408 | 0.000 | 0.000 | 5 / 3 |
| 12 | 0.488 | 0.314 | 36% | 1.644 | 1.248 | 0.000 | 0.000 | 13 / 8 |

## Table 2 -- Head-to-head on total score (lower score is better)

| Budget (s) | CNN wins | Flood-fill wins | Draws | Mean margin (s) | 95% CI | Significant |
|---|---|---|---|---|---|---|
| none | 34 (17%) | 148 (74%) | 18 | -0.373 | ±0.059 | yes |
| 22 | 75 (38%) | 105 (52%) | 20 | +0.103 | ±0.141 | no |
| 20 | 74 (37%) | 107 (54%) | 19 | +0.163 | ±0.151 | yes |
| 18 | 83 (42%) | 93 (46%) | 24 | +0.215 | ±0.214 | yes |
| 15 | 79 (40%) | 94 (47%) | 27 | +0.233 | ±0.235 | no |
| 12 | 89 (44%) | 76 (38%) | 35 | +0.407 | ±0.326 | yes |

## Table 3 -- Search effort

| Budget (s) | Score FF | Score CNN | Search time FF (s) | Search time CNN (s) | Maze sensed FF | Maze sensed CNN |
|---|---|---|---|---|---|---|
| none | 5.233 | 5.607 | 24.2 | 35.4 | 44% | 41% |
| 22 | 5.302 | 5.199 | 19.7 | 21.3 | 41% | 35% |
| 20 | 5.317 | 5.154 | 18.7 | 19.9 | 40% | 34% |
| 18 | 5.389 | 5.174 | 17.6 | 18.5 | 39% | 34% |
| 15 | 5.513 | 5.279 | 15.7 | 16.1 | 38% | 32% |
| 12 | 5.904 | 5.497 | 13.4 | 13.6 | 35% | 31% |

Notes: positive margin = CNN better. Gap statistics are over runs that found a route; failed runs are counted in the Failures column and charged a score of 15 s in Tables 2-3. 'Significant' = the 95% CI of the paired margin excludes zero.
