# CNN vs flood-fill -- model `cnn_outbound_s0.pt`, 200 mazes per budget

## Table 1 -- Route quality (exploration gap = seconds slower than the optimal route)

| Budget (s) | Flood-fill mean | CNN mean | Gap closed | Flood-fill p90 | CNN p90 | Flood-fill median | CNN median | Failures FF / CNN |
|---|---|---|---|---|---|---|---|---|
| none | 0.000 | 0.000 | -- | 0.000 | 0.000 | 0.000 | 0.000 | 0 / 0 |
| 22 | 0.218 | 0.049 | 78% | 0.180 | 0.000 | 0.000 | 0.000 | 0 / 0 |
| 20 | 0.266 | 0.072 | 73% | 0.360 | 0.000 | 0.000 | 0.000 | 0 / 0 |
| 18 | 0.374 | 0.129 | 65% | 0.618 | 0.120 | 0.000 | 0.000 | 0 / 0 |
| 15 | 0.367 | 0.209 | 43% | 1.020 | 0.480 | 0.000 | 0.000 | 5 / 2 |
| 12 | 0.488 | 0.300 | 38% | 1.644 | 1.140 | 0.000 | 0.000 | 13 / 6 |

## Table 2 -- Head-to-head on total score (lower score is better)

| Budget (s) | CNN wins | Flood-fill wins | Draws | Mean margin (s) | 95% CI | Significant |
|---|---|---|---|---|---|---|
| none | 32 (16%) | 152 (76%) | 16 | -0.361 | ±0.059 | yes |
| 22 | 74 (37%) | 109 (54%) | 17 | +0.114 | ±0.141 | no |
| 20 | 74 (37%) | 108 (54%) | 18 | +0.151 | ±0.150 | yes |
| 18 | 84 (42%) | 99 (50%) | 17 | +0.215 | ±0.217 | no |
| 15 | 88 (44%) | 87 (44%) | 25 | +0.258 | ±0.228 | yes |
| 12 | 90 (45%) | 76 (38%) | 34 | +0.513 | ±0.335 | yes |

## Table 3 -- Search effort

| Budget (s) | Score FF | Score CNN | Search time FF (s) | Search time CNN (s) | Maze sensed FF | Maze sensed CNN |
|---|---|---|---|---|---|---|
| none | 5.233 | 5.595 | 24.2 | 35.0 | 44% | 42% |
| 22 | 5.302 | 5.189 | 19.7 | 21.4 | 41% | 35% |
| 20 | 5.317 | 5.166 | 18.7 | 20.0 | 40% | 34% |
| 18 | 5.389 | 5.174 | 17.6 | 18.5 | 39% | 34% |
| 15 | 5.513 | 5.254 | 15.7 | 16.1 | 38% | 33% |
| 12 | 5.904 | 5.391 | 13.4 | 13.7 | 35% | 31% |

Notes: positive margin = CNN better. Gap statistics are over runs that found a route; failed runs are counted in the Failures column and charged a score of 15 s in Tables 2-3. 'Significant' = the 95% CI of the paired margin excludes zero.
