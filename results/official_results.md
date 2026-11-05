# CNN vs flood-fill across 3 training seeds (142 mazes each)

Values: mean over seeds [min - max]. Flood-fill is identical across seeds (same mazes), so its columns are single numbers.

| Budget (s) | Gap closed | CNN failures | FF failures | Score margin (s) | Seeds significantly better |
|---|---|---|---|---|---|
| none | -- | 0.0 [0.0 - 0.0] | 0 | -1.046 [-1.090 - -1.010] | 0 / 3 (worse in 3) |
| 56 | -364% [-402% - -319%] | 0.0 [0.0 - 0.0] | 2 | -0.408 [-0.427 - -0.387] | 0 / 3 (worse in 3) |
| 51 | -143% [-156% - -127%] | 0.0 [0.0 - 0.0] | 2 | -0.399 [-0.428 - -0.368] | 0 / 3 (worse in 3) |
| 45 | -95% [-111% - -74%] | 0.0 [0.0 - 0.0] | 4 | -0.409 [-0.469 - -0.327] | 0 / 3 (worse in 3) |
| 38 | -42% [-49% - -34%] | 7.0 [6.0 - 9.0] | 8 | -0.441 [-0.585 - -0.341] | 0 / 3 (worse in 3) |
| 31 | -12% [-17% - -9%] | 21.3 [19.0 - 23.0] | 22 | -0.221 [-0.304 - -0.061] | 0 / 3 |

Positive margin = CNN better. 'Seeds significantly better' counts seeds whose paired 95% CI excludes zero in the CNN's favour.
