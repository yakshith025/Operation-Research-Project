# E8: Van battery upper bound B (MILP)

**Question.** Is recharging worth modelling? If Z barely moves, drop it.

B = 200 kWh for every van. If the mean change of Z on paired instances stays below decision_threshold_pct, a larger battery (or recharging) changes little and van recharging need not be modelled.

Baseline: `milp_300s` in `methods/milp/results`. Changes are over paired instances (valid in both). Van shares are pooled over the free-choice EVs of the paired instances.

## Variants against the baseline

| variant | run_id | valid | paired | newly_valid | no_longer_valid | mean_dZ_% | median_dZ_% | mean_d_operator_cost_% | mean_d_customer_cost_% | baseline_share_van_personal_% | share_van_personal_% | baseline_share_van_commercial_% | share_van_commercial_% | vans_used | baseline_vans_used |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E8_battery_200 | milp_300s | 476 | 473 | 3 | 1 | -0.005 | 0.0 | -0.4 | 0.191 | 19.92 | 18.49 | 0.72 | 0.7 | 2.14 | 2.19 |

## Headlines

- E8_battery_200 (milp_300s): Z -0.005 % on average over 473 paired instances; van share personal 19.92 -> 18.49 %, commercial 0.72 -> 0.7 %; 3 instance(s) newly solved, 1 lost.
- Decision rule (E8): |mean change of Z| = 0.01 % is below 1.0 %: a larger battery changes little, so van recharging need not be modelled.

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
| E8_battery_200 (milp_300s) | personal | 0.0-0.2 | 126 | 126 | 0.0 | 2.42 | 513.3 |
| E8_battery_200 (milp_300s) | personal | 0.2-0.4 | 696 | 612 | 0.5 | 1.73 | 653.98 |
| E8_battery_200 (milp_300s) | personal | 0.4-0.6 | 1498 | 1188 | 2.5 | 0.72 | 677.32 |
| E8_battery_200 (milp_300s) | personal | 0.6-0.8 | 1950 | 1682 | 9.0 | 0.27 | 696.22 |
| E8_battery_200 (milp_300s) | personal | 0.8-1.0 | 2824 | 2531 | 37.7 | 0.06 | 654.15 |
| E8_battery_200 (milp_300s) | commercial | 0.0-0.2 | 2555 | 2319 | 1.6 | 2.78 | 609.17 |
| E8_battery_200 (milp_300s) | commercial | 0.2-0.4 | 7684 | 6408 | 1.0 | 1.8 | 595.13 |
| E8_battery_200 (milp_300s) | commercial | 0.4-0.6 | 5385 | 4421 | 0.1 | 0.88 | 561.12 |
| E8_battery_200 (milp_300s) | commercial | 0.6-0.8 | 1192 | 992 | 0.1 | 0.75 | 560.49 |
| E8_battery_200 (milp_300s) | commercial | 0.8-1.0 | 80 | 80 | 0.0 | 0.28 | 426.85 |

Full columns in comparison.csv and comparison_by_income.csv.
