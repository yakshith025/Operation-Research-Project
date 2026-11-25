# E5: Value of charging time at theta = 1 (MILP)

**Question.** How sensitive is the van share (per segment) to the most influential assumed value?

Personal: c^{C,1} (value of a minute of FCS charging at the top income), the most influential value in formulation 3. Commercial: the downtime rate at theta = 1 (D = C = V, and Q with it), the commercial counterpart; it is scaled as a whole so that D = C = V still holds. +-25 %.

Baseline: `milp_300s` in `baseline_MILP/results`. Changes are over paired instances (valid in both). Van shares are pooled over the free-choice EVs of the paired instances.

## Variants against the baseline

| variant | run_id | valid | paired | newly_valid | no_longer_valid | mean_dZ_% | median_dZ_% | mean_d_operator_cost_% | mean_d_customer_cost_% | baseline_share_van_personal_% | share_van_personal_% | baseline_share_van_commercial_% | share_van_commercial_% | vans_used | baseline_vans_used |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E5_cC1_personal_7.5 | milp_300s | 473 | 472 | 1 | 2 | -2.212 | -2.195 | -1.147 | -2.593 | 19.91 | 14.05 | 0.72 | 0.82 | 2.13 | 2.18 |
| E5_cC1_personal_12.5 | milp_300s | 473 | 472 | 1 | 2 | 2.078 | 1.994 | 2.746 | 1.837 | 19.88 | 29.36 | 0.73 | 0.67 | 2.37 | 2.18 |
| E5_omega1_commercial_4.5 | milp_300s | 472 | 471 | 1 | 3 | -2.666 | -2.662 | -0.17 | -3.799 | 19.88 | 19.61 | 0.72 | 0.67 | 2.17 | 2.18 |
| E5_omega1_commercial_7.5 | milp_300s | 475 | 472 | 3 | 2 | 2.662 | 2.651 | 0.26 | 3.815 | 19.88 | 19.57 | 0.73 | 0.68 | 2.2 | 2.18 |

## Headlines

- E5_cC1_personal_7.5 (milp_300s): Z -2.212 % on average over 472 paired instances; van share personal 19.91 -> 14.05 %, commercial 0.72 -> 0.82 %; 1 instance(s) newly solved, 2 lost.
- E5_cC1_personal_12.5 (milp_300s): Z +2.078 % on average over 472 paired instances; van share personal 19.88 -> 29.36 %, commercial 0.73 -> 0.67 %; 1 instance(s) newly solved, 2 lost.
- E5_omega1_commercial_4.5 (milp_300s): Z -2.666 % on average over 471 paired instances; van share personal 19.88 -> 19.61 %, commercial 0.72 -> 0.67 %; 1 instance(s) newly solved, 3 lost.
- E5_omega1_commercial_7.5 (milp_300s): Z +2.662 % on average over 472 paired instances; van share personal 19.88 -> 19.57 %, commercial 0.73 -> 0.68 %; 3 instance(s) newly solved, 2 lost.

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
| E5_cC1_personal_7.5 (milp_300s) | personal | 0.0-0.2 | 124 | 124 | 0.0 | 2.11 | 497.65 |
| E5_cC1_personal_7.5 (milp_300s) | personal | 0.2-0.4 | 688 | 606 | 0.3 | 1.16 | 614.17 |
| E5_cC1_personal_7.5 (milp_300s) | personal | 0.4-0.6 | 1484 | 1175 | 1.8 | 0.59 | 619.54 |
| E5_cC1_personal_7.5 (milp_300s) | personal | 0.6-0.8 | 1926 | 1662 | 5.1 | 0.26 | 620.59 |
| E5_cC1_personal_7.5 (milp_300s) | personal | 0.8-1.0 | 2795 | 2506 | 29.8 | 0.05 | 594.55 |
| E5_cC1_personal_7.5 (milp_300s) | commercial | 0.0-0.2 | 2522 | 2288 | 1.8 | 2.8 | 608.6 |
| E5_cC1_personal_7.5 (milp_300s) | commercial | 0.2-0.4 | 7612 | 6344 | 1.1 | 1.8 | 594.15 |
| E5_cC1_personal_7.5 (milp_300s) | commercial | 0.4-0.6 | 5339 | 4384 | 0.2 | 0.91 | 560.85 |
| E5_cC1_personal_7.5 (milp_300s) | commercial | 0.6-0.8 | 1180 | 982 | 0.0 | 0.7 | 559.34 |
| E5_cC1_personal_7.5 (milp_300s) | commercial | 0.8-1.0 | 80 | 80 | 0.0 | 0.37 | 432.49 |
| E5_cC1_personal_12.5 (milp_300s) | personal | 0.0-0.2 | 124 | 124 | 0.0 | 2.32 | 520.74 |
| E5_cC1_personal_12.5 (milp_300s) | personal | 0.2-0.4 | 688 | 606 | 0.0 | 1.14 | 683.07 |
| E5_cC1_personal_12.5 (milp_300s) | personal | 0.4-0.6 | 1484 | 1175 | 3.4 | 0.61 | 732.55 |
| E5_cC1_personal_12.5 (milp_300s) | personal | 0.6-0.8 | 1925 | 1661 | 18.5 | 0.19 | 748.06 |
| E5_cC1_personal_12.5 (milp_300s) | personal | 0.8-1.0 | 2795 | 2506 | 57.3 | 0.09 | 652.68 |
| E5_cC1_personal_12.5 (milp_300s) | commercial | 0.0-0.2 | 2522 | 2288 | 1.6 | 2.56 | 606.5 |
| E5_cC1_personal_12.5 (milp_300s) | commercial | 0.2-0.4 | 7609 | 6342 | 0.8 | 1.57 | 592.46 |
| E5_cC1_personal_12.5 (milp_300s) | commercial | 0.4-0.6 | 5334 | 4380 | 0.2 | 0.83 | 559.68 |
| E5_cC1_personal_12.5 (milp_300s) | commercial | 0.6-0.8 | 1179 | 981 | 0.0 | 0.61 | 559.3 |
| E5_cC1_personal_12.5 (milp_300s) | commercial | 0.8-1.0 | 80 | 80 | 0.0 | 0.24 | 418.84 |
| E5_omega1_commercial_4.5 (milp_300s) | personal | 0.0-0.2 | 124 | 124 | 0.0 | 2.78 | 512.09 |
| E5_omega1_commercial_4.5 (milp_300s) | personal | 0.2-0.4 | 686 | 605 | 0.3 | 1.24 | 642.99 |
| E5_omega1_commercial_4.5 (milp_300s) | personal | 0.4-0.6 | 1482 | 1173 | 2.0 | 0.55 | 674.53 |
| E5_omega1_commercial_4.5 (milp_300s) | personal | 0.6-0.8 | 1915 | 1652 | 9.9 | 0.16 | 688.71 |
| E5_omega1_commercial_4.5 (milp_300s) | personal | 0.8-1.0 | 2789 | 2500 | 39.9 | 0.03 | 645.69 |
| E5_omega1_commercial_4.5 (milp_300s) | commercial | 0.0-0.2 | 2512 | 2279 | 1.6 | 2.67 | 590.59 |
| E5_omega1_commercial_4.5 (milp_300s) | commercial | 0.2-0.4 | 7603 | 6335 | 0.9 | 1.86 | 566.14 |
| E5_omega1_commercial_4.5 (milp_300s) | commercial | 0.4-0.6 | 5332 | 4379 | 0.1 | 0.88 | 515.89 |
| E5_omega1_commercial_4.5 (milp_300s) | commercial | 0.6-0.8 | 1177 | 980 | 0.1 | 0.81 | 501.81 |
| E5_omega1_commercial_4.5 (milp_300s) | commercial | 0.8-1.0 | 80 | 80 | 0.0 | 0.09 | 366.7 |
| E5_omega1_commercial_7.5 (milp_300s) | personal | 0.0-0.2 | 125 | 125 | 0.0 | 2.69 | 516.71 |
| E5_omega1_commercial_7.5 (milp_300s) | personal | 0.2-0.4 | 692 | 609 | 0.2 | 1.34 | 651.2 |
| E5_omega1_commercial_7.5 (milp_300s) | personal | 0.4-0.6 | 1494 | 1184 | 2.5 | 0.6 | 678.81 |
| E5_omega1_commercial_7.5 (milp_300s) | personal | 0.6-0.8 | 1946 | 1679 | 9.3 | 0.34 | 697.94 |
| E5_omega1_commercial_7.5 (milp_300s) | personal | 0.8-1.0 | 2810 | 2520 | 40.0 | 0.1 | 651.11 |
| E5_omega1_commercial_7.5 (milp_300s) | commercial | 0.0-0.2 | 2545 | 2309 | 1.3 | 2.66 | 621.06 |
| E5_omega1_commercial_7.5 (milp_300s) | commercial | 0.2-0.4 | 7646 | 6376 | 1.0 | 1.64 | 621.38 |
| E5_omega1_commercial_7.5 (milp_300s) | commercial | 0.4-0.6 | 5363 | 4403 | 0.3 | 0.8 | 604.14 |
| E5_omega1_commercial_7.5 (milp_300s) | commercial | 0.6-0.8 | 1189 | 989 | 0.0 | 0.59 | 616.91 |
| E5_omega1_commercial_7.5 (milp_300s) | commercial | 0.8-1.0 | 80 | 80 | 0.0 | 0.0 | 487.48 |

Full columns in comparison.csv and comparison_by_income.csv.
