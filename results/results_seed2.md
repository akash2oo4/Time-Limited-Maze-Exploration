# CNN vs flood-fill -- model `cnn_outbound_s2.pt`, 200 mazes per budget

## Table 1 -- Route quality (exploration gap = seconds slower than the optimal route)

| Budget (s) | Flood-fill mean | CNN mean | Gap closed | Flood-fill p90 | CNN p90 | Flood-fill median | CNN median | Failures FF / CNN |
|---|---|---|---|---|---|---|---|---|
| none | 0.000 | 0.000 | -- | 0.000 | 0.000 | 0.000 | 0.000 | 0 / 0 |
| 22 | 0.218 | 0.056 | 75% | 0.180 | 0.000 | 0.000 | 0.000 | 0 / 0 |
| 20 | 0.266 | 0.073 | 72% | 0.360 | 0.012 | 0.000 | 0.000 | 0 / 0 |
| 18 | 0.374 | 0.092 | 75% | 0.618 | 0.126 | 0.000 | 0.000 | 0 / 0 |
| 15 | 0.367 | 0.217 | 41% | 1.020 | 0.480 | 0.000 | 0.000 | 5 / 3 |
| 12 | 0.488 | 0.292 | 40% | 1.644 | 1.122 | 0.000 | 0.000 | 13 / 6 |

## Table 2 -- Head-to-head on total score (lower score is better)

| Budget (s) | CNN wins | Flood-fill wins | Draws | Mean margin (s) | 95% CI | Significant |
|---|---|---|---|---|---|---|
| none | 37 (18%) | 146 (73%) | 17 | -0.351 | ±0.058 | yes |
| 22 | 81 (40%) | 99 (50%) | 20 | +0.109 | ±0.132 | no |
| 20 | 72 (36%) | 107 (54%) | 21 | +0.152 | ±0.139 | yes |
| 18 | 91 (46%) | 85 (42%) | 24 | +0.254 | ±0.188 | yes |
| 15 | 90 (45%) | 86 (43%) | 24 | +0.226 | ±0.223 | yes |
| 12 | 89 (44%) | 77 (38%) | 34 | +0.519 | ±0.324 | yes |

## Table 3 -- Search effort

| Budget (s) | Score FF | Score CNN | Search time FF (s) | Search time CNN (s) | Maze sensed FF | Maze sensed CNN |
|---|---|---|---|---|---|---|
| none | 5.233 | 5.584 | 24.2 | 34.7 | 44% | 41% |
| 22 | 5.302 | 5.194 | 19.7 | 21.3 | 41% | 34% |
| 20 | 5.317 | 5.165 | 18.7 | 19.9 | 40% | 34% |
| 18 | 5.389 | 5.134 | 17.6 | 18.4 | 39% | 34% |
| 15 | 5.513 | 5.287 | 15.7 | 16.1 | 38% | 33% |
| 12 | 5.904 | 5.385 | 13.4 | 13.6 | 35% | 31% |

Notes: positive margin = CNN better. Gap statistics are over runs that found a route; failed runs are counted in the Failures column and charged a score of 15 s in Tables 2-3. 'Significant' = the 95% CI of the paired margin excludes zero.
