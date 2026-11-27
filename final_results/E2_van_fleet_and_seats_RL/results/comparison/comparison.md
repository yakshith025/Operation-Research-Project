# E2: Van fleet and seats (one at a time) (RL)

**Question.** Where does another van or seat pay off, and where is capacity idle?

Number of vans |M| (config n_vans) and EVs per van Q (config van_max_evs) for every Solomon file.

Baseline: `rl_f4` in `baseline_RL/results`. Changes are over paired instances (valid in both). Van shares are pooled over the free-choice EVs of the paired instances.

## Variants against the baseline

| variant | run_id | valid | paired | newly_valid | no_longer_valid | mean_dZ_% | median_dZ_% | mean_d_operator_cost_% | mean_d_customer_cost_% | baseline_share_van_personal_% | share_van_personal_% | baseline_share_van_commercial_% | share_van_commercial_% | vans_used | baseline_vans_used | mean_gap_to_milp_% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E2_vans_3 | rl_f4_zero_shot | 465 | 465 | 0 | 35 | 0.024 | 0.0 | -0.266 | 0.153 | 6.38 | 5.74 | 0.31 | 0.23 | 2.0 | 2.03 | 0.911 |
| E2_vans_3 | rl_f4_retrained | 466 | 466 | 0 | 34 | 0.086 | 0.0 | -0.12 | 0.178 | 6.37 | 5.87 | 0.39 | 0.23 | 2.02 | 2.03 | 0.957 |
| E2_vans_7 | rl_f4_zero_shot | 506 | 500 | 6 | 0 | 0.045 | 0.0 | 0.133 | 0.025 | 6.73 | 6.72 | 0.94 | 1.01 | 2.23 | 2.21 | 0.836 |
| E2_vans_7 | rl_f4_retrained | 506 | 500 | 6 | 0 | 0.004 | 0.0 | 0.057 | -0.002 | 6.73 | 7.03 | 0.94 | 0.68 | 2.23 | 2.21 | 0.806 |
| E2_seats_4 | rl_f4_zero_shot | 493 | 493 | 0 | 7 | 0.909 | 0.187 | 3.873 | -0.348 | 6.72 | 5.77 | 0.46 | 0.37 | 2.54 | 2.17 | 0.693 |
| E2_seats_4 | rl_f4_retrained | 494 | 494 | 0 | 6 | 0.878 | 0.336 | 4.043 | -0.46 | 6.75 | 7.25 | 0.52 | 0.33 | 2.55 | 2.18 | 0.651 |
| E2_seats_8 | rl_f4_zero_shot | 500 | 499 | 1 | 1 | -0.083 | 0.0 | -0.613 | 0.16 | 6.73 | 6.6 | 0.93 | 0.92 | 2.13 | 2.2 | 0.855 |
| E2_seats_8 | rl_f4_retrained | 499 | 499 | 0 | 1 | 0.066 | 0.013 | -0.308 | 0.247 | 6.74 | 7.19 | 0.91 | 0.94 | 2.16 | 2.2 | 0.979 |

## Headlines

- E2_vans_3 (rl_f4_zero_shot): Z +0.024 % on average over 465 paired instances; van share personal 6.38 -> 5.74 %, commercial 0.31 -> 0.23 %; 0 instance(s) newly solved, 35 lost.
- E2_vans_3 (rl_f4_retrained): Z +0.086 % on average over 466 paired instances; van share personal 6.37 -> 5.87 %, commercial 0.39 -> 0.23 %; 0 instance(s) newly solved, 34 lost.
- E2_vans_7 (rl_f4_zero_shot): Z +0.045 % on average over 500 paired instances; van share personal 6.73 -> 6.72 %, commercial 0.94 -> 1.01 %; 6 instance(s) newly solved, 0 lost.
- E2_vans_7 (rl_f4_retrained): Z +0.004 % on average over 500 paired instances; van share personal 6.73 -> 7.03 %, commercial 0.94 -> 0.68 %; 6 instance(s) newly solved, 0 lost.
- E2_seats_4 (rl_f4_zero_shot): Z +0.909 % on average over 493 paired instances; van share personal 6.72 -> 5.77 %, commercial 0.46 -> 0.37 %; 0 instance(s) newly solved, 7 lost.
- E2_seats_4 (rl_f4_retrained): Z +0.878 % on average over 494 paired instances; van share personal 6.75 -> 7.25 %, commercial 0.52 -> 0.33 %; 0 instance(s) newly solved, 6 lost.
- E2_seats_8 (rl_f4_zero_shot): Z -0.083 % on average over 499 paired instances; van share personal 6.73 -> 6.6 %, commercial 0.93 -> 0.92 %; 1 instance(s) newly solved, 1 lost.
- E2_seats_8 (rl_f4_retrained): Z +0.066 % on average over 499 paired instances; van share personal 6.74 -> 7.19 %, commercial 0.91 -> 0.94 %; 0 instance(s) newly solved, 1 lost.

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
| E2_vans_3 (rl_f4_zero_shot) | personal | 0.0-0.2 | 117 | 117 | 0.0 | 1.39 | 503.41 |
| E2_vans_3 (rl_f4_zero_shot) | personal | 0.2-0.4 | 672 | 587 | 0.0 | 1.27 | 637.6 |
| E2_vans_3 (rl_f4_zero_shot) | personal | 0.4-0.6 | 1420 | 1124 | 0.4 | 0.59 | 674.77 |
| E2_vans_3 (rl_f4_zero_shot) | personal | 0.6-0.8 | 1882 | 1631 | 3.2 | 0.47 | 707.97 |
| E2_vans_3 (rl_f4_zero_shot) | personal | 0.8-1.0 | 2731 | 2448 | 11.5 | 0.09 | 720.01 |
| E2_vans_3 (rl_f4_zero_shot) | commercial | 0.0-0.2 | 2452 | 2228 | 0.0 | 4.08 | 608.71 |
| E2_vans_3 (rl_f4_zero_shot) | commercial | 0.2-0.4 | 7390 | 6156 | 0.3 | 2.54 | 600.03 |
| E2_vans_3 (rl_f4_zero_shot) | commercial | 0.4-0.6 | 5175 | 4249 | 0.2 | 1.48 | 570.3 |
| E2_vans_3 (rl_f4_zero_shot) | commercial | 0.6-0.8 | 1142 | 949 | 0.4 | 1.16 | 569.63 |
| E2_vans_3 (rl_f4_zero_shot) | commercial | 0.8-1.0 | 79 | 79 | 0.0 | 0.33 | 434.6 |
| E2_vans_3 (rl_f4_retrained) | personal | 0.0-0.2 | 118 | 118 | 0.0 | 2.26 | 512.18 |
| E2_vans_3 (rl_f4_retrained) | personal | 0.2-0.4 | 675 | 591 | 0.0 | 1.33 | 638.7 |
| E2_vans_3 (rl_f4_retrained) | personal | 0.4-0.6 | 1432 | 1133 | 0.5 | 0.62 | 676.99 |
| E2_vans_3 (rl_f4_retrained) | personal | 0.6-0.8 | 1885 | 1632 | 3.9 | 0.66 | 712.93 |
| E2_vans_3 (rl_f4_retrained) | personal | 0.8-1.0 | 2739 | 2456 | 11.4 | 0.11 | 721.64 |
| E2_vans_3 (rl_f4_retrained) | commercial | 0.0-0.2 | 2461 | 2236 | 0.1 | 3.96 | 608.34 |
| E2_vans_3 (rl_f4_retrained) | commercial | 0.2-0.4 | 7433 | 6192 | 0.3 | 2.73 | 600.03 |
| E2_vans_3 (rl_f4_retrained) | commercial | 0.4-0.6 | 5200 | 4269 | 0.2 | 1.43 | 569.34 |
| E2_vans_3 (rl_f4_retrained) | commercial | 0.6-0.8 | 1148 | 955 | 0.1 | 1.13 | 565.57 |
| E2_vans_3 (rl_f4_retrained) | commercial | 0.8-1.0 | 79 | 79 | 0.0 | 0.0 | 429.33 |
| E2_vans_7 (rl_f4_zero_shot) | personal | 0.0-0.2 | 144 | 144 | 0.0 | 3.35 | 536.99 |
| E2_vans_7 (rl_f4_zero_shot) | personal | 0.2-0.4 | 758 | 662 | 0.0 | 2.18 | 651.11 |
| E2_vans_7 (rl_f4_zero_shot) | personal | 0.4-0.6 | 1687 | 1354 | 1.3 | 1.27 | 680.8 |
| E2_vans_7 (rl_f4_zero_shot) | personal | 0.6-0.8 | 2272 | 1957 | 4.8 | 1.0 | 716.25 |
| E2_vans_7 (rl_f4_zero_shot) | personal | 0.8-1.0 | 3094 | 2773 | 12.9 | 0.22 | 723.63 |
| E2_vans_7 (rl_f4_zero_shot) | commercial | 0.0-0.2 | 2911 | 2644 | 1.1 | 5.67 | 619.05 |
| E2_vans_7 (rl_f4_zero_shot) | commercial | 0.2-0.4 | 8394 | 7036 | 1.2 | 3.8 | 608.99 |
| E2_vans_7 (rl_f4_zero_shot) | commercial | 0.4-0.6 | 5919 | 4852 | 1.4 | 2.5 | 587.06 |
| E2_vans_7 (rl_f4_zero_shot) | commercial | 0.6-0.8 | 1370 | 1140 | 1.1 | 2.15 | 579.09 |
| E2_vans_7 (rl_f4_zero_shot) | commercial | 0.8-1.0 | 81 | 81 | 0.0 | 0.46 | 438.32 |
| E2_vans_7 (rl_f4_retrained) | personal | 0.0-0.2 | 144 | 144 | 0.0 | 5.01 | 546.26 |
| E2_vans_7 (rl_f4_retrained) | personal | 0.2-0.4 | 758 | 662 | 0.3 | 3.69 | 668.1 |
| E2_vans_7 (rl_f4_retrained) | personal | 0.4-0.6 | 1687 | 1354 | 2.3 | 2.0 | 696.5 |
| E2_vans_7 (rl_f4_retrained) | personal | 0.6-0.8 | 2272 | 1957 | 5.3 | 1.02 | 717.89 |
| E2_vans_7 (rl_f4_retrained) | personal | 0.8-1.0 | 3094 | 2773 | 13.4 | 0.18 | 724.54 |
| E2_vans_7 (rl_f4_retrained) | commercial | 0.0-0.2 | 2911 | 2644 | 0.8 | 5.68 | 619.48 |
| E2_vans_7 (rl_f4_retrained) | commercial | 0.2-0.4 | 8394 | 7036 | 1.1 | 3.82 | 607.02 |
| E2_vans_7 (rl_f4_retrained) | commercial | 0.4-0.6 | 5919 | 4852 | 0.9 | 2.4 | 580.74 |
| E2_vans_7 (rl_f4_retrained) | commercial | 0.6-0.8 | 1370 | 1140 | 0.2 | 1.8 | 568.38 |
| E2_vans_7 (rl_f4_retrained) | commercial | 0.8-1.0 | 81 | 81 | 0.0 | 0.24 | 428.81 |
| E2_seats_4 (rl_f4_zero_shot) | personal | 0.0-0.2 | 137 | 137 | 0.0 | 2.9 | 536.73 |
| E2_seats_4 (rl_f4_zero_shot) | personal | 0.2-0.4 | 732 | 640 | 0.2 | 1.61 | 645.24 |
| E2_seats_4 (rl_f4_zero_shot) | personal | 0.4-0.6 | 1603 | 1283 | 1.2 | 0.82 | 673.16 |
| E2_seats_4 (rl_f4_zero_shot) | personal | 0.6-0.8 | 2154 | 1855 | 3.6 | 0.87 | 709.95 |
| E2_seats_4 (rl_f4_zero_shot) | personal | 0.8-1.0 | 2953 | 2649 | 11.2 | 0.18 | 720.95 |
| E2_seats_4 (rl_f4_zero_shot) | commercial | 0.0-0.2 | 2762 | 2510 | 0.3 | 4.93 | 610.82 |
| E2_seats_4 (rl_f4_zero_shot) | commercial | 0.2-0.4 | 8021 | 6718 | 0.5 | 3.38 | 597.89 |
| E2_seats_4 (rl_f4_zero_shot) | commercial | 0.4-0.6 | 5637 | 4621 | 0.3 | 2.15 | 574.19 |
| E2_seats_4 (rl_f4_zero_shot) | commercial | 0.6-0.8 | 1302 | 1082 | 0.4 | 2.17 | 573.85 |
| E2_seats_4 (rl_f4_zero_shot) | commercial | 0.8-1.0 | 79 | 79 | 0.0 | 0.59 | 434.66 |
| E2_seats_4 (rl_f4_retrained) | personal | 0.0-0.2 | 138 | 138 | 0.0 | 6.48 | 554.28 |
| E2_seats_4 (rl_f4_retrained) | personal | 0.2-0.4 | 733 | 641 | 0.3 | 3.23 | 663.78 |
| E2_seats_4 (rl_f4_retrained) | personal | 0.4-0.6 | 1609 | 1288 | 2.1 | 1.63 | 687.99 |
| E2_seats_4 (rl_f4_retrained) | personal | 0.6-0.8 | 2162 | 1862 | 4.8 | 0.99 | 715.54 |
| E2_seats_4 (rl_f4_retrained) | personal | 0.8-1.0 | 2965 | 2660 | 13.5 | 0.21 | 719.74 |
| E2_seats_4 (rl_f4_retrained) | commercial | 0.0-0.2 | 2771 | 2518 | 0.3 | 4.95 | 609.53 |
| E2_seats_4 (rl_f4_retrained) | commercial | 0.2-0.4 | 8042 | 6734 | 0.4 | 3.15 | 595.37 |
| E2_seats_4 (rl_f4_retrained) | commercial | 0.4-0.6 | 5665 | 4646 | 0.3 | 1.98 | 570.62 |
| E2_seats_4 (rl_f4_retrained) | commercial | 0.6-0.8 | 1306 | 1085 | 0.2 | 1.7 | 567.45 |
| E2_seats_4 (rl_f4_retrained) | commercial | 0.8-1.0 | 79 | 79 | 0.0 | 0.0 | 424.84 |
| E2_seats_8 (rl_f4_zero_shot) | personal | 0.0-0.2 | 142 | 142 | 0.0 | 2.53 | 529.51 |
| E2_seats_8 (rl_f4_zero_shot) | personal | 0.2-0.4 | 744 | 652 | 0.0 | 1.84 | 647.32 |
| E2_seats_8 (rl_f4_zero_shot) | personal | 0.4-0.6 | 1653 | 1324 | 1.6 | 1.12 | 678.57 |
| E2_seats_8 (rl_f4_zero_shot) | personal | 0.6-0.8 | 2196 | 1890 | 4.9 | 0.78 | 711.03 |
| E2_seats_8 (rl_f4_zero_shot) | personal | 0.8-1.0 | 3033 | 2718 | 12.1 | 0.19 | 724.21 |
| E2_seats_8 (rl_f4_zero_shot) | commercial | 0.0-0.2 | 2826 | 2566 | 0.6 | 5.27 | 614.65 |
| E2_seats_8 (rl_f4_zero_shot) | commercial | 0.2-0.4 | 8237 | 6898 | 1.2 | 3.5 | 609.36 |
| E2_seats_8 (rl_f4_zero_shot) | commercial | 0.4-0.6 | 5816 | 4772 | 0.9 | 2.4 | 584.41 |
| E2_seats_8 (rl_f4_zero_shot) | commercial | 0.6-0.8 | 1332 | 1108 | 0.8 | 2.05 | 579.46 |
| E2_seats_8 (rl_f4_zero_shot) | commercial | 0.8-1.0 | 81 | 81 | 1.2 | 0.35 | 444.15 |
| E2_seats_8 (rl_f4_retrained) | personal | 0.0-0.2 | 140 | 140 | 0.0 | 3.39 | 536.33 |
| E2_seats_8 (rl_f4_retrained) | personal | 0.2-0.4 | 743 | 650 | 0.0 | 2.0 | 650.66 |
| E2_seats_8 (rl_f4_retrained) | personal | 0.4-0.6 | 1642 | 1315 | 1.5 | 1.43 | 684.67 |
| E2_seats_8 (rl_f4_retrained) | personal | 0.6-0.8 | 2192 | 1889 | 5.2 | 1.02 | 719.09 |
| E2_seats_8 (rl_f4_retrained) | personal | 0.8-1.0 | 3023 | 2709 | 13.4 | 0.22 | 722.86 |
| E2_seats_8 (rl_f4_retrained) | commercial | 0.0-0.2 | 2822 | 2563 | 0.7 | 5.16 | 613.82 |
| E2_seats_8 (rl_f4_retrained) | commercial | 0.2-0.4 | 8212 | 6879 | 1.2 | 3.42 | 609.11 |
| E2_seats_8 (rl_f4_retrained) | commercial | 0.4-0.6 | 5778 | 4737 | 0.8 | 2.29 | 580.98 |
| E2_seats_8 (rl_f4_retrained) | commercial | 0.6-0.8 | 1327 | 1104 | 0.4 | 1.94 | 576.06 |
| E2_seats_8 (rl_f4_retrained) | commercial | 0.8-1.0 | 81 | 81 | 0.0 | 0.0 | 431.88 |

Full columns in comparison.csv and comparison_by_income.csv.
