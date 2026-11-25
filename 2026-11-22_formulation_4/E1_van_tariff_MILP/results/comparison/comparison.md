# E1: Van tariff p^M (MILP)

**Question.** Who gets vans (per segment and income), and at what cost to the operator and to equity?

p^M = 15 equals the FCS tariff p^F (no van premium); 35 doubles the premium.

Baseline: `milp_300s` in `baseline_MILP/results`. Changes are over paired instances (valid in both). Van shares are pooled over the free-choice EVs of the paired instances.

## Variants against the baseline

| variant | run_id | valid | paired | newly_valid | no_longer_valid | mean_dZ_% | median_dZ_% | mean_d_operator_cost_% | mean_d_customer_cost_% | baseline_share_van_personal_% | share_van_personal_% | baseline_share_van_commercial_% | share_van_commercial_% | vans_used | baseline_vans_used |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1_pM_15 | milp_300s | 471 | 470 | 1 | 4 | -4.189 | -4.145 | 3.686 | -7.8 | 19.78 | 31.37 | 0.73 | 1.38 | 2.4 | 2.17 |
| E1_pM_35 | milp_300s | 477 | 474 | 3 | 0 | 3.914 | 3.876 | -1.386 | 6.532 | 19.92 | 13.52 | 0.73 | 0.76 | 2.12 | 2.19 |

## Headlines

- E1_pM_15 (milp_300s): Z -4.189 % on average over 470 paired instances; van share personal 19.78 -> 31.37 %, commercial 0.73 -> 1.38 %; 1 instance(s) newly solved, 4 lost.
- E1_pM_35 (milp_300s): Z +3.914 % on average over 474 paired instances; van share personal 19.92 -> 13.52 %, commercial 0.73 -> 0.76 %; 3 instance(s) newly solved, 0 lost.

## Van share by segment and income band (all valid plans)

| run | segment | income_band | EVs | free_choice | share_van_free_choice_% | mean_queue_FCS_min | mean_customer_cost_Rs |
|---|---|---|---|---|---|---|---|
| baseline (milp_300s) | personal | 0.0-0.2 | 125 | 125 | 0.0 | 1.45 | 509.15 |
| baseline (milp_300s) | personal | 0.2-0.4 | 690 | 608 | 0.0 | 1.32 | 645.81 |
| baseline (milp_300s) | personal | 0.4-0.6 | 1491 | 1181 | 2.1 | 0.57 | 675.22 |
| baseline (milp_300s) | personal | 0.6-0.8 | 1934 | 1668 | 9.8 | 0.18 | 690.8 |
| baseline (milp_300s) | personal | 0.8-1.0 | 2803 | 2513 | 40.8 | 0.06 | 646.07 |
| baseline (milp_300s) | commercial | 0.0-0.2 | 2532 | 2297 | 1.6 | 2.77 | 607.42 |
| baseline (milp_300s) | commercial | 0.2-0.4 | 7633 | 6363 | 1.0 | 1.76 | 593.67 |
| baseline (milp_300s) | commercial | 0.4-0.6 | 5356 | 4398 | 0.1 | 0.92 | 559.76 |
| baseline (milp_300s) | commercial | 0.6-0.8 | 1186 | 987 | 0.0 | 0.75 | 558.79 |
| baseline (milp_300s) | commercial | 0.8-1.0 | 80 | 80 | 0.0 | 0.0 | 424.95 |
| E1_pM_15 (milp_300s) | personal | 0.0-0.2 | 124 | 124 | 0.8 | 1.8 | 508.21 |
| E1_pM_15 (milp_300s) | personal | 0.2-0.4 | 684 | 604 | 3.1 | 1.19 | 608.05 |
| E1_pM_15 (milp_300s) | personal | 0.4-0.6 | 1480 | 1171 | 11.5 | 0.63 | 623.09 |
| E1_pM_15 (milp_300s) | personal | 0.6-0.8 | 1903 | 1641 | 24.4 | 0.21 | 619.13 |
| E1_pM_15 (milp_300s) | personal | 0.8-1.0 | 2783 | 2494 | 53.4 | 0.11 | 546.59 |
| E1_pM_15 (milp_300s) | commercial | 0.0-0.2 | 2502 | 2270 | 2.2 | 2.26 | 576.22 |
| E1_pM_15 (milp_300s) | commercial | 0.2-0.4 | 7591 | 6324 | 1.4 | 1.58 | 552.8 |
| E1_pM_15 (milp_300s) | commercial | 0.4-0.6 | 5320 | 4370 | 1.1 | 0.78 | 526.97 |
| E1_pM_15 (milp_300s) | commercial | 0.6-0.8 | 1173 | 977 | 1.4 | 0.6 | 525.64 |
| E1_pM_15 (milp_300s) | commercial | 0.8-1.0 | 80 | 80 | 0.0 | 0.24 | 424.65 |
| E1_pM_35 (milp_300s) | personal | 0.0-0.2 | 127 | 127 | 0.0 | 2.98 | 517.97 |
| E1_pM_35 (milp_300s) | personal | 0.2-0.4 | 698 | 614 | 0.2 | 1.31 | 685.39 |
| E1_pM_35 (milp_300s) | personal | 0.4-0.6 | 1505 | 1194 | 1.6 | 0.63 | 712.13 |
| E1_pM_35 (milp_300s) | personal | 0.6-0.8 | 1958 | 1688 | 3.4 | 0.24 | 723.52 |
| E1_pM_35 (milp_300s) | personal | 0.8-1.0 | 2832 | 2538 | 29.9 | 0.07 | 712.49 |
| E1_pM_35 (milp_300s) | commercial | 0.0-0.2 | 2565 | 2328 | 1.4 | 2.89 | 635.87 |
| E1_pM_35 (milp_300s) | commercial | 0.2-0.4 | 7705 | 6427 | 1.2 | 1.85 | 636.27 |
| E1_pM_35 (milp_300s) | commercial | 0.4-0.6 | 5402 | 4435 | 0.2 | 0.95 | 592.89 |
| E1_pM_35 (milp_300s) | commercial | 0.6-0.8 | 1198 | 997 | 0.1 | 0.79 | 595.94 |
| E1_pM_35 (milp_300s) | commercial | 0.8-1.0 | 80 | 80 | 0.0 | 0.0 | 427.85 |

Full columns in comparison.csv and comparison_by_income.csv.
