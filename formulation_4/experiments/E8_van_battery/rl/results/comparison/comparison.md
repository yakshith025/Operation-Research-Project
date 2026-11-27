# E8: Van battery upper bound B (RL)

**Question.** Is recharging worth modelling? If Z barely moves, drop it.

B = 200 kWh for every van. If the mean change of Z on paired instances stays below decision_threshold_pct, a larger battery (or recharging) changes little and van recharging need not be modelled.

Baseline: `rl_f4` in `methods/rl/results`. Changes are over paired instances (valid in both). Van shares are pooled over the free-choice EVs of the paired instances.

## Variants against the baseline

| variant | run_id | valid | paired | newly_valid | no_longer_valid | mean_dZ_% | median_dZ_% | mean_d_operator_cost_% | mean_d_customer_cost_% | baseline_share_van_personal_% | share_van_personal_% | baseline_share_van_commercial_% | share_van_commercial_% | vans_used | baseline_vans_used | mean_gap_to_milp_% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E8_battery_200 | rl_f4_zero_shot | 501 | 500 | 1 | 0 | -0.018 | 0.0 | -0.165 | 0.05 | 6.73 | 6.51 | 0.94 | 0.89 | 2.19 | 2.21 | 0.835 |
| E8_battery_200 | rl_f4_retrained | 500 | 499 | 1 | 1 | -0.075 | 0.0 | -0.059 | -0.07 | 6.74 | 8.16 | 0.91 | 0.65 | 2.18 | 2.2 | 0.77 |

## Headlines

- E8_battery_200 (rl_f4_zero_shot): Z -0.018 % on average over 500 paired instances; van share personal 6.73 -> 6.51 %, commercial 0.94 -> 0.89 %; 1 instance(s) newly solved, 0 lost.
- E8_battery_200 (rl_f4_retrained): Z -0.075 % on average over 499 paired instances; van share personal 6.74 -> 8.16 %, commercial 0.91 -> 0.65 %; 1 instance(s) newly solved, 1 lost.
- Decision rule (E8): |mean change of Z| = 0.02 % is below 1.0 %: a larger battery changes little, so van recharging need not be modelled.
- Decision rule (E8): |mean change of Z| = 0.07 % is below 1.0 %: a larger battery changes little, so van recharging need not be modelled.

## Van share by segment and income band (all valid plans)

| run | segment | income_band | EVs | free_choice | share_van_free_choice_% | mean_queue_FCS_min | mean_customer_cost_Rs |
|---|---|---|---|---|---|---|---|
| baseline (rl_f4) | personal | 0.0-0.2 | 141 | 141 | 0.0 | 2.47 | 531.45 |
| baseline (rl_f4) | personal | 0.2-0.4 | 745 | 652 | 0.0 | 1.58 | 645.53 |
| baseline (rl_f4) | personal | 0.4-0.6 | 1650 | 1322 | 1.4 | 1.03 | 678.46 |
| baseline (rl_f4) | personal | 0.6-0.8 | 2201 | 1896 | 4.7 | 0.98 | 713.0 |
| baseline (rl_f4) | personal | 0.8-1.0 | 3031 | 2716 | 12.7 | 0.19 | 721.92 |
| baseline (rl_f4) | commercial | 0.0-0.2 | 2832 | 2572 | 0.5 | 5.18 | 614.32 |
| baseline (rl_f4) | commercial | 0.2-0.4 | 8237 | 6901 | 1.3 | 3.57 | 608.07 |
| baseline (rl_f4) | commercial | 0.4-0.6 | 5798 | 4754 | 0.7 | 2.4 | 581.04 |
| baseline (rl_f4) | commercial | 0.6-0.8 | 1334 | 1110 | 0.5 | 2.11 | 575.96 |
| baseline (rl_f4) | commercial | 0.8-1.0 | 81 | 81 | 0.0 | 1.01 | 442.35 |
| E8_battery_200 (rl_f4_zero_shot) | personal | 0.0-0.2 | 142 | 142 | 0.0 | 2.48 | 531.41 |
| E8_battery_200 (rl_f4_zero_shot) | personal | 0.2-0.4 | 746 | 653 | 0.0 | 1.78 | 647.45 |
| E8_battery_200 (rl_f4_zero_shot) | personal | 0.4-0.6 | 1657 | 1328 | 1.4 | 0.92 | 676.04 |
| E8_battery_200 (rl_f4_zero_shot) | personal | 0.6-0.8 | 2210 | 1903 | 4.5 | 0.91 | 713.77 |
| E8_battery_200 (rl_f4_zero_shot) | personal | 0.8-1.0 | 3044 | 2728 | 12.4 | 0.21 | 724.88 |
| E8_battery_200 (rl_f4_zero_shot) | commercial | 0.0-0.2 | 2841 | 2580 | 0.7 | 5.22 | 616.24 |
| E8_battery_200 (rl_f4_zero_shot) | commercial | 0.2-0.4 | 8261 | 6919 | 1.2 | 3.62 | 608.94 |
| E8_battery_200 (rl_f4_zero_shot) | commercial | 0.4-0.6 | 5830 | 4783 | 0.8 | 2.45 | 582.77 |
| E8_battery_200 (rl_f4_zero_shot) | commercial | 0.6-0.8 | 1338 | 1113 | 0.7 | 2.04 | 580.14 |
| E8_battery_200 (rl_f4_zero_shot) | commercial | 0.8-1.0 | 81 | 81 | 0.0 | 0.88 | 440.17 |
| E8_battery_200 (rl_f4_retrained) | personal | 0.0-0.2 | 141 | 141 | 0.0 | 5.22 | 542.25 |
| E8_battery_200 (rl_f4_retrained) | personal | 0.2-0.4 | 744 | 651 | 0.0 | 3.29 | 667.17 |
| E8_battery_200 (rl_f4_retrained) | personal | 0.4-0.6 | 1649 | 1321 | 1.9 | 1.56 | 689.61 |
| E8_battery_200 (rl_f4_retrained) | personal | 0.6-0.8 | 2201 | 1896 | 6.2 | 0.99 | 719.7 |
| E8_battery_200 (rl_f4_retrained) | personal | 0.8-1.0 | 3036 | 2721 | 15.0 | 0.19 | 726.34 |
| E8_battery_200 (rl_f4_retrained) | commercial | 0.0-0.2 | 2831 | 2571 | 0.6 | 5.15 | 613.56 |
| E8_battery_200 (rl_f4_retrained) | commercial | 0.2-0.4 | 8236 | 6897 | 0.8 | 3.43 | 603.9 |
| E8_battery_200 (rl_f4_retrained) | commercial | 0.4-0.6 | 5810 | 4766 | 0.7 | 2.34 | 577.96 |
| E8_battery_200 (rl_f4_retrained) | commercial | 0.6-0.8 | 1331 | 1107 | 0.3 | 1.8 | 571.65 |
| E8_battery_200 (rl_f4_retrained) | commercial | 0.8-1.0 | 81 | 81 | 0.0 | 0.23 | 424.1 |

Full columns in comparison.csv and comparison_by_income.csv.
