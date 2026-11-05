# CNN vs flood-fill -- model `cnn_heuristic_sortie.pt`, 200 mazes per budget

## Table 1 -- Route quality (exploration gap = seconds slower than the optimal route)

| Budget (s) | Flood-fill mean | CNN mean | Gap closed | Flood-fill p90 | CNN p90 | Flood-fill median | CNN median | Failures FF / CNN |
|---|---|---|---|---|---|---|---|---|
| none | 0.000 | 0.000 | -- | 0.000 | 0.000 | 0.000 | 0.000 | 0 / 0 |
| 22 | 0.218 | 0.089 | 59% | 0.180 | 0.300 | 0.000 | 0.000 | 0 / 0 |
| 20 | 0.266 | 0.116 | 56% | 0.360 | 0.360 | 0.000 | 0.000 | 0 / 0 |
| 18 | 0.374 | 0.190 | 49% | 0.618 | 0.480 | 0.000 | 0.000 | 0 / 0 |
| 15 | 0.367 | 0.284 | 23% | 1.020 | 0.708 | 0.000 | 0.000 | 5 / 3 |
| 12 | 0.488 | 0.377 | 23% | 1.644 | 1.080 | 0.000 | 0.000 | 13 / 7 |

## Table 2 -- Head-to-head on total score (lower score is better)

| Budget (s) | CNN wins | Flood-fill wins | Draws | Mean margin (s) | 95% CI | Significant |
|---|---|---|---|---|---|---|
| none | 24 (12%) | 158 (79%) | 18 | -0.420 | ±0.059 | yes |
| 22 | 64 (32%) | 113 (56%) | 23 | +0.064 | ±0.137 | no |
| 20 | 64 (32%) | 114 (57%) | 22 | +0.101 | ±0.139 | no |
| 18 | 70 (35%) | 108 (54%) | 22 | +0.151 | ±0.207 | no |
| 15 | 67 (34%) | 106 (53%) | 27 | +0.146 | ±0.250 | no |
| 12 | 70 (35%) | 97 (48%) | 33 | +0.397 | ±0.354 | yes |

## Table 3 -- Search effort

| Budget (s) | Score FF | Score CNN | Search time FF (s) | Search time CNN (s) | Maze sensed FF | Maze sensed CNN |
|---|---|---|---|---|---|---|
| none | 5.233 | 5.653 | 24.2 | 36.8 | 44% | 41% |
| 22 | 5.302 | 5.238 | 19.7 | 21.6 | 41% | 34% |
| 20 | 5.317 | 5.216 | 18.7 | 20.2 | 40% | 33% |
| 18 | 5.389 | 5.238 | 17.6 | 18.6 | 39% | 33% |
| 15 | 5.513 | 5.366 | 15.7 | 16.2 | 38% | 32% |
| 12 | 5.904 | 5.507 | 13.4 | 13.7 | 35% | 31% |

Notes: positive margin = CNN better. Gap statistics are over runs that found a route; failed runs are counted in the Failures column and charged a score of 15 s in Tables 2-3. 'Significant' = the 95% CI of the paired margin excludes zero.
