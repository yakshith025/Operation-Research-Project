# E5: Value of charging time at theta = 1 (RL)

**Question.** How sensitive is the van share (per segment) to the most influential assumed value?

Personal: c^{C,1} (value of a minute of FCS charging at the top income), the most influential value in formulation 3. Commercial: the downtime rate at theta = 1 (D = C = V, and Q with it), the commercial counterpart; it is scaled as a whole so that D = C = V still holds. +-25 %.

Baseline: `rl_f4` in `baseline_RL/results`. Changes are over paired instances (valid in both). Van shares are pooled over the free-choice EVs of the paired instances.

## Variants against the baseline

| variant | run_id | valid | paired | newly_valid | no_longer_valid | mean_dZ_% | median_dZ_% | mean_d_operator_cost_% | mean_d_customer_cost_% | baseline_share_van_personal_% | share_van_personal_% | baseline_share_van_commercial_% | share_van_commercial_% | vans_used | baseline_vans_used | mean_gap_to_milp_% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E5_cC1_personal_7.5 | rl_f4_zero_shot | 499 | 499 | 0 | 1 | -2.486 | -2.541 | -0.212 | -3.449 | 6.73 | 5.46 | 0.93 | 0.92 | 2.2 | 2.2 | 0.555 |
| E5_cC1_personal_7.5 | rl_f4_retrained | 500 | 499 | 1 | 1 | -2.459 | -2.561 | -0.587 | -3.238 | 6.73 | 3.85 | 0.93 | 0.83 | 2.19 | 2.2 | 0.559 |
| E5_cC1_personal_12.5 | rl_f4_zero_shot | 502 | 500 | 2 | 0 | 2.448 | 2.49 | 0.315 | 3.365 | 6.73 | 7.95 | 0.94 | 0.84 | 2.23 | 2.21 | 1.217 |
| E5_cC1_personal_12.5 | rl_f4_retrained | 500 | 500 | 0 | 0 | 2.38 | 2.461 | 0.928 | 3.022 | 6.73 | 11.73 | 0.94 | 0.56 | 2.26 | 2.21 | 1.14 |
| E5_omega1_commercial_4.5 | rl_f4_zero_shot | 500 | 499 | 1 | 1 | -2.722 | -2.706 | -0.13 | -3.858 | 6.73 | 6.49 | 0.93 | 0.85 | 2.19 | 2.2 | 0.822 |
| E5_omega1_commercial_4.5 | rl_f4_retrained | 500 | 499 | 1 | 1 | -2.521 | -2.562 | 0.17 | -3.695 | 6.74 | 7.58 | 0.91 | 0.57 | 2.23 | 2.2 | 0.979 |
| E5_omega1_commercial_7.5 | rl_f4_zero_shot | 500 | 500 | 0 | 0 | 2.722 | 2.705 | 0.259 | 3.824 | 6.73 | 6.93 | 0.94 | 0.93 | 2.24 | 2.21 | 0.833 |
| E5_omega1_commercial_7.5 | rl_f4_retrained | 499 | 499 | 0 | 1 | 2.752 | 2.687 | 0.14 | 3.917 | 6.74 | 7.04 | 0.91 | 0.79 | 2.22 | 2.2 | 0.857 |

## Headlines

- E5_cC1_personal_7.5 (rl_f4_zero_shot): Z -2.486 % on average over 499 paired instances; van share personal 6.73 -> 5.46 %, commercial 0.93 -> 0.92 %; 0 instance(s) newly solved, 1 lost.
- E5_cC1_personal_7.5 (rl_f4_retrained): Z -2.459 % on average over 499 paired instances; van share personal 6.73 -> 3.85 %, commercial 0.93 -> 0.83 %; 1 instance(s) newly solved, 1 lost.
- E5_cC1_personal_12.5 (rl_f4_zero_shot): Z +2.448 % on average over 500 paired instances; van share personal 6.73 -> 7.95 %, commercial 0.94 -> 0.84 %; 2 instance(s) newly solved, 0 lost.
- E5_cC1_personal_12.5 (rl_f4_retrained): Z +2.38 % on average over 500 paired instances; van share personal 6.73 -> 11.73 %, commercial 0.94 -> 0.56 %; 0 instance(s) newly solved, 0 lost.
- E5_omega1_commercial_4.5 (rl_f4_zero_shot): Z -2.722 % on average over 499 paired instances; van share personal 6.73 -> 6.49 %, commercial 0.93 -> 0.85 %; 1 instance(s) newly solved, 1 lost.
- E5_omega1_commercial_4.5 (rl_f4_retrained): Z -2.521 % on average over 499 paired instances; van share personal 6.74 -> 7.58 %, commercial 0.91 -> 0.57 %; 1 instance(s) newly solved, 1 lost.
- E5_omega1_commercial_7.5 (rl_f4_zero_shot): Z +2.722 % on average over 500 paired instances; van share personal 6.73 -> 6.93 %, commercial 0.94 -> 0.93 %; 0 instance(s) newly solved, 0 lost.
- E5_omega1_commercial_7.5 (rl_f4_retrained): Z +2.752 % on average over 499 paired instances; van share personal 6.74 -> 7.04 %, commercial 0.91 -> 0.79 %; 0 instance(s) newly solved, 1 lost.

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
| E5_cC1_personal_7.5 (rl_f4_zero_shot) | personal | 0.0-0.2 | 141 | 141 | 0.0 | 2.54 | 520.98 |
| E5_cC1_personal_7.5 (rl_f4_zero_shot) | personal | 0.2-0.4 | 743 | 651 | 0.0 | 1.78 | 613.66 |
| E5_cC1_personal_7.5 (rl_f4_zero_shot) | personal | 0.4-0.6 | 1646 | 1318 | 1.3 | 0.97 | 619.1 |
| E5_cC1_personal_7.5 (rl_f4_zero_shot) | personal | 0.6-0.8 | 2187 | 1883 | 3.3 | 0.72 | 630.96 |
| E5_cC1_personal_7.5 (rl_f4_zero_shot) | personal | 0.8-1.0 | 3020 | 2706 | 10.6 | 0.19 | 631.62 |
| E5_cC1_personal_7.5 (rl_f4_zero_shot) | commercial | 0.0-0.2 | 2817 | 2558 | 0.6 | 5.02 | 613.04 |
| E5_cC1_personal_7.5 (rl_f4_zero_shot) | commercial | 0.2-0.4 | 8213 | 6880 | 1.0 | 3.57 | 607.11 |
| E5_cC1_personal_7.5 (rl_f4_zero_shot) | commercial | 0.4-0.6 | 5784 | 4743 | 0.9 | 2.37 | 583.22 |
| E5_cC1_personal_7.5 (rl_f4_zero_shot) | commercial | 0.6-0.8 | 1328 | 1105 | 0.7 | 2.16 | 577.57 |
| E5_cC1_personal_7.5 (rl_f4_zero_shot) | commercial | 0.8-1.0 | 81 | 81 | 0.0 | 0.56 | 443.08 |
| E5_cC1_personal_7.5 (rl_f4_retrained) | personal | 0.0-0.2 | 142 | 142 | 0.0 | 5.76 | 538.25 |
| E5_cC1_personal_7.5 (rl_f4_retrained) | personal | 0.2-0.4 | 744 | 652 | 0.5 | 3.76 | 633.18 |
| E5_cC1_personal_7.5 (rl_f4_retrained) | personal | 0.4-0.6 | 1653 | 1324 | 1.8 | 2.07 | 638.54 |
| E5_cC1_personal_7.5 (rl_f4_retrained) | personal | 0.6-0.8 | 2196 | 1890 | 2.1 | 1.09 | 645.9 |
| E5_cC1_personal_7.5 (rl_f4_retrained) | personal | 0.8-1.0 | 3033 | 2718 | 7.1 | 0.24 | 640.78 |
| E5_cC1_personal_7.5 (rl_f4_retrained) | commercial | 0.0-0.2 | 2826 | 2566 | 0.7 | 5.3 | 615.85 |
| E5_cC1_personal_7.5 (rl_f4_retrained) | commercial | 0.2-0.4 | 8237 | 6898 | 1.1 | 3.5 | 606.03 |
| E5_cC1_personal_7.5 (rl_f4_retrained) | commercial | 0.4-0.6 | 5816 | 4772 | 0.7 | 2.21 | 578.71 |
| E5_cC1_personal_7.5 (rl_f4_retrained) | commercial | 0.6-0.8 | 1332 | 1108 | 0.6 | 1.45 | 566.92 |
| E5_cC1_personal_7.5 (rl_f4_retrained) | commercial | 0.8-1.0 | 81 | 81 | 0.0 | 0.0 | 421.49 |
| E5_cC1_personal_12.5 (rl_f4_zero_shot) | personal | 0.0-0.2 | 142 | 142 | 0.0 | 2.43 | 543.97 |
| E5_cC1_personal_12.5 (rl_f4_zero_shot) | personal | 0.2-0.4 | 748 | 654 | 0.0 | 1.87 | 681.85 |
| E5_cC1_personal_12.5 (rl_f4_zero_shot) | personal | 0.4-0.6 | 1661 | 1332 | 1.5 | 1.09 | 735.41 |
| E5_cC1_personal_12.5 (rl_f4_zero_shot) | personal | 0.6-0.8 | 2224 | 1916 | 5.2 | 1.05 | 792.55 |
| E5_cC1_personal_12.5 (rl_f4_zero_shot) | personal | 0.8-1.0 | 3055 | 2738 | 15.2 | 0.23 | 809.98 |
| E5_cC1_personal_12.5 (rl_f4_zero_shot) | commercial | 0.0-0.2 | 2856 | 2594 | 0.5 | 5.24 | 614.29 |
| E5_cC1_personal_12.5 (rl_f4_zero_shot) | commercial | 0.2-0.4 | 8285 | 6940 | 1.1 | 3.72 | 607.88 |
| E5_cC1_personal_12.5 (rl_f4_zero_shot) | commercial | 0.4-0.6 | 5844 | 4794 | 0.9 | 2.4 | 583.58 |
| E5_cC1_personal_12.5 (rl_f4_zero_shot) | commercial | 0.6-0.8 | 1344 | 1118 | 0.7 | 1.96 | 578.42 |
| E5_cC1_personal_12.5 (rl_f4_zero_shot) | commercial | 0.8-1.0 | 81 | 81 | 1.2 | 0.47 | 448.67 |
| E5_cC1_personal_12.5 (rl_f4_retrained) | personal | 0.0-0.2 | 141 | 141 | 0.0 | 6.31 | 565.93 |
| E5_cC1_personal_12.5 (rl_f4_retrained) | personal | 0.2-0.4 | 745 | 652 | 0.0 | 3.56 | 698.59 |
| E5_cC1_personal_12.5 (rl_f4_retrained) | personal | 0.4-0.6 | 1650 | 1322 | 2.8 | 1.79 | 749.86 |
| E5_cC1_personal_12.5 (rl_f4_retrained) | personal | 0.6-0.8 | 2201 | 1896 | 9.9 | 1.16 | 792.56 |
| E5_cC1_personal_12.5 (rl_f4_retrained) | personal | 0.8-1.0 | 3031 | 2716 | 20.8 | 0.3 | 801.44 |
| E5_cC1_personal_12.5 (rl_f4_retrained) | commercial | 0.0-0.2 | 2832 | 2572 | 0.5 | 4.8 | 610.41 |
| E5_cC1_personal_12.5 (rl_f4_retrained) | commercial | 0.2-0.4 | 8237 | 6901 | 0.7 | 3.38 | 601.29 |
| E5_cC1_personal_12.5 (rl_f4_retrained) | commercial | 0.4-0.6 | 5798 | 4754 | 0.6 | 2.12 | 575.15 |
| E5_cC1_personal_12.5 (rl_f4_retrained) | commercial | 0.6-0.8 | 1334 | 1110 | 0.1 | 1.76 | 568.48 |
| E5_cC1_personal_12.5 (rl_f4_retrained) | commercial | 0.8-1.0 | 81 | 81 | 0.0 | 0.0 | 417.53 |
| E5_omega1_commercial_4.5 (rl_f4_zero_shot) | personal | 0.0-0.2 | 142 | 142 | 0.0 | 2.69 | 532.27 |
| E5_omega1_commercial_4.5 (rl_f4_zero_shot) | personal | 0.2-0.4 | 744 | 652 | 0.0 | 1.34 | 643.58 |
| E5_omega1_commercial_4.5 (rl_f4_zero_shot) | personal | 0.4-0.6 | 1653 | 1324 | 1.4 | 1.13 | 676.95 |
| E5_omega1_commercial_4.5 (rl_f4_zero_shot) | personal | 0.6-0.8 | 2196 | 1890 | 4.1 | 0.77 | 711.05 |
| E5_omega1_commercial_4.5 (rl_f4_zero_shot) | personal | 0.8-1.0 | 3033 | 2718 | 12.5 | 0.18 | 721.96 |
| E5_omega1_commercial_4.5 (rl_f4_zero_shot) | commercial | 0.0-0.2 | 2826 | 2566 | 0.4 | 5.34 | 597.72 |
| E5_omega1_commercial_4.5 (rl_f4_zero_shot) | commercial | 0.2-0.4 | 8237 | 6898 | 1.2 | 3.65 | 579.09 |
| E5_omega1_commercial_4.5 (rl_f4_zero_shot) | commercial | 0.4-0.6 | 5816 | 4772 | 0.9 | 2.42 | 537.25 |
| E5_omega1_commercial_4.5 (rl_f4_zero_shot) | commercial | 0.6-0.8 | 1332 | 1108 | 0.5 | 1.89 | 513.7 |
| E5_omega1_commercial_4.5 (rl_f4_zero_shot) | commercial | 0.8-1.0 | 81 | 81 | 0.0 | 1.01 | 377.91 |
| E5_omega1_commercial_4.5 (rl_f4_retrained) | personal | 0.0-0.2 | 140 | 140 | 0.0 | 3.68 | 537.6 |
| E5_omega1_commercial_4.5 (rl_f4_retrained) | personal | 0.2-0.4 | 745 | 651 | 0.3 | 2.68 | 658.1 |
| E5_omega1_commercial_4.5 (rl_f4_retrained) | personal | 0.4-0.6 | 1646 | 1319 | 2.4 | 1.62 | 692.63 |
| E5_omega1_commercial_4.5 (rl_f4_retrained) | personal | 0.6-0.8 | 2206 | 1902 | 5.6 | 1.2 | 726.2 |
| E5_omega1_commercial_4.5 (rl_f4_retrained) | personal | 0.8-1.0 | 3034 | 2719 | 13.7 | 0.36 | 737.14 |
| E5_omega1_commercial_4.5 (rl_f4_retrained) | commercial | 0.0-0.2 | 2837 | 2577 | 0.2 | 4.93 | 594.09 |
| E5_omega1_commercial_4.5 (rl_f4_retrained) | commercial | 0.2-0.4 | 8236 | 6900 | 0.7 | 3.38 | 571.74 |
| E5_omega1_commercial_4.5 (rl_f4_retrained) | commercial | 0.4-0.6 | 5792 | 4748 | 0.7 | 2.29 | 531.0 |
| E5_omega1_commercial_4.5 (rl_f4_retrained) | commercial | 0.6-0.8 | 1333 | 1109 | 0.6 | 1.98 | 514.84 |
| E5_omega1_commercial_4.5 (rl_f4_retrained) | commercial | 0.8-1.0 | 81 | 81 | 0.0 | 0.42 | 361.25 |
| E5_omega1_commercial_7.5 (rl_f4_zero_shot) | personal | 0.0-0.2 | 141 | 141 | 0.0 | 2.59 | 532.3 |
| E5_omega1_commercial_7.5 (rl_f4_zero_shot) | personal | 0.2-0.4 | 745 | 652 | 0.0 | 1.35 | 643.46 |
| E5_omega1_commercial_7.5 (rl_f4_zero_shot) | personal | 0.4-0.6 | 1650 | 1322 | 1.4 | 1.07 | 678.98 |
| E5_omega1_commercial_7.5 (rl_f4_zero_shot) | personal | 0.6-0.8 | 2201 | 1896 | 5.4 | 0.9 | 711.54 |
| E5_omega1_commercial_7.5 (rl_f4_zero_shot) | personal | 0.8-1.0 | 3031 | 2716 | 12.7 | 0.22 | 724.0 |
| E5_omega1_commercial_7.5 (rl_f4_zero_shot) | commercial | 0.0-0.2 | 2832 | 2572 | 0.4 | 5.1 | 629.02 |
| E5_omega1_commercial_7.5 (rl_f4_zero_shot) | commercial | 0.2-0.4 | 8237 | 6901 | 1.1 | 3.62 | 637.49 |
| E5_omega1_commercial_7.5 (rl_f4_zero_shot) | commercial | 0.4-0.6 | 5798 | 4754 | 0.9 | 2.26 | 628.29 |
| E5_omega1_commercial_7.5 (rl_f4_zero_shot) | commercial | 0.6-0.8 | 1334 | 1110 | 0.9 | 2.15 | 640.04 |
| E5_omega1_commercial_7.5 (rl_f4_zero_shot) | commercial | 0.8-1.0 | 81 | 81 | 1.2 | 0.85 | 505.07 |
| E5_omega1_commercial_7.5 (rl_f4_retrained) | personal | 0.0-0.2 | 140 | 140 | 0.0 | 4.51 | 542.83 |
| E5_omega1_commercial_7.5 (rl_f4_retrained) | personal | 0.2-0.4 | 743 | 650 | 0.2 | 3.01 | 663.64 |
| E5_omega1_commercial_7.5 (rl_f4_retrained) | personal | 0.4-0.6 | 1642 | 1315 | 1.6 | 1.72 | 692.81 |
| E5_omega1_commercial_7.5 (rl_f4_retrained) | personal | 0.6-0.8 | 2192 | 1889 | 5.4 | 1.11 | 723.39 |
| E5_omega1_commercial_7.5 (rl_f4_retrained) | personal | 0.8-1.0 | 3023 | 2709 | 12.8 | 0.27 | 733.57 |
| E5_omega1_commercial_7.5 (rl_f4_retrained) | commercial | 0.0-0.2 | 2822 | 2563 | 0.7 | 4.91 | 629.19 |
| E5_omega1_commercial_7.5 (rl_f4_retrained) | commercial | 0.2-0.4 | 8212 | 6879 | 1.0 | 3.43 | 633.36 |
| E5_omega1_commercial_7.5 (rl_f4_retrained) | commercial | 0.4-0.6 | 5778 | 4737 | 0.6 | 2.09 | 622.9 |
| E5_omega1_commercial_7.5 (rl_f4_retrained) | commercial | 0.6-0.8 | 1327 | 1104 | 0.5 | 1.57 | 627.0 |
| E5_omega1_commercial_7.5 (rl_f4_retrained) | commercial | 0.8-1.0 | 81 | 81 | 0.0 | 0.0 | 485.96 |

Full columns in comparison.csv and comparison_by_income.csv.
