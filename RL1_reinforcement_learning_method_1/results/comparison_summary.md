# RL1 vs MILP vs greedy (rl1 vs milp_300s)

560 instances. RL1 gives a valid plan in 502, the MILP in 479 (256 proven optimal). Where both have a valid plan, RL1 costs on average 1.07 % more than the MILP plan (median 0.89 %); against proven optima the mean gap is 0.74 %. RL1 beats the greedy rule in 366/474 instances (mean gap to greedy -2.47 %). RL1 has a valid plan where the MILP has none in 23 instances; the reverse happens in 0. Mean time per instance: RL1 0.12 s, MILP 150.7 s.

Gap = 100 (Z_RL1 - Z_other) / Z_other, over instances where both plans are valid; negative = RL1 cheaper.

## By class

| group | instances | RL1 valid | MILP valid | MILP optimal | RL1 valid, MILP none | MILP valid, RL1 not | gap to MILP mean % | gap to MILP median % | gap to optimum mean % | RL1 < greedy | gap to greedy mean % | RL1 s mean | MILP s mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 90 | 90 | 90 | 64 | 0 | 0 | 1.06 | 1.01 | 0.74 | 59/90 | -1.12 | 0.12 | 118.9 |
| C2 | 80 | 80 | 79 | 50 | 1 | 0 | 1.60 | 1.58 | 1.44 | 50/79 | -1.04 | 0.12 | 135.1 |
| R1 | 120 | 89 | 76 | 36 | 13 | 0 | 0.80 | 0.45 | 0.34 | 69/75 | -1.96 | 0.12 | 147.3 |
| R2 | 110 | 110 | 110 | 64 | 0 | 0 | 1.22 | 0.99 | 0.66 | 83/110 | -2.57 | 0.12 | 139.5 |
| RC1 | 80 | 53 | 44 | 13 | 9 | 0 | 1.13 | 0.93 | 0.75 | 37/41 | -5.94 | 0.12 | 174.4 |
| RC2 | 80 | 80 | 80 | 29 | 0 | 0 | 0.60 | 0.41 | 0.19 | 68/79 | -4.01 | 0.12 | 198.8 |

## By number of EVs

| group | instances | RL1 valid | MILP valid | MILP optimal | RL1 valid, MILP none | MILP valid, RL1 not | gap to MILP mean % | gap to MILP median % | gap to optimum mean % | RL1 < greedy | gap to greedy mean % | RL1 s mean | MILP s mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n = 10 | 56 | 55 | 55 | 55 | 0 | 0 | 0.50 | 0.00 | 0.50 | 19/55 | -1.60 | 0.03 | 0.1 |
| n = 20 | 56 | 55 | 55 | 53 | 0 | 0 | 0.77 | 0.54 | 0.75 | 27/55 | -1.36 | 0.04 | 22.8 |
| n = 30 | 56 | 55 | 55 | 41 | 0 | 0 | 0.69 | 0.55 | 0.53 | 28/55 | -1.03 | 0.06 | 96.5 |
| n = 40 | 56 | 55 | 55 | 29 | 0 | 0 | 0.68 | 0.55 | 0.61 | 45/54 | -2.45 | 0.08 | 153.6 |
| n = 50 | 56 | 53 | 53 | 21 | 0 | 0 | 0.97 | 0.77 | 0.64 | 46/52 | -3.14 | 0.11 | 181.9 |
| n = 60 | 56 | 50 | 50 | 19 | 0 | 0 | 1.52 | 1.41 | 1.22 | 49/49 | -3.17 | 0.13 | 185.9 |
| n = 70 | 56 | 47 | 45 | 14 | 2 | 0 | 1.60 | 1.34 | 1.19 | 43/44 | -3.13 | 0.15 | 191.1 |
| n = 80 | 56 | 47 | 39 | 11 | 8 | 0 | 1.69 | 1.51 | 1.19 | 39/39 | -3.13 | 0.17 | 218.1 |
| n = 90 | 56 | 46 | 37 | 7 | 9 | 0 | 1.59 | 1.58 | 1.41 | 37/37 | -3.35 | 0.19 | 235.5 |
| n = 100 | 56 | 39 | 35 | 6 | 4 | 0 | 1.30 | 1.23 | 0.98 | 33/34 | -3.50 | 0.22 | 221.2 |

## All

| group | instances | RL1 valid | MILP valid | MILP optimal | RL1 valid, MILP none | MILP valid, RL1 not | gap to MILP mean % | gap to MILP median % | gap to optimum mean % | RL1 < greedy | gap to greedy mean % | RL1 s mean | MILP s mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 560 | 502 | 479 | 256 | 23 | 0 | 1.07 | 0.89 | 0.74 | 366/474 | -2.47 | 0.12 | 150.7 |
