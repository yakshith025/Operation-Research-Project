# E1: Van tariff p^M (RL)

**Question.** Who gets vans (per segment and income), and at what cost to the operator and to equity?

p^M = 15 equals the FCS tariff p^F (no van premium); 35 doubles the premium.

Baseline: `rl_f4` in `methods/rl/results`. Changes are over paired instances (valid in both). Van shares are pooled over the free-choice EVs of the paired instances.

## Variants against the baseline

| variant | run_id | valid | paired | newly_valid | no_longer_valid | mean_dZ_% | median_dZ_% | mean_d_operator_cost_% | mean_d_customer_cost_% | baseline_share_van_personal_% | share_van_personal_% | baseline_share_van_commercial_% | share_van_commercial_% | vans_used | baseline_vans_used | mean_gap_to_milp_% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1_pM_15 | rl_f4_zero_shot | 500 | 499 | 1 | 1 | -3.736 | -3.561 | 0.446 | -5.661 | 6.74 | 8.38 | 0.91 | 1.02 | 2.22 | 2.2 | 1.307 |
| E1_pM_15 | rl_f4_retrained | 499 | 499 | 0 | 1 | -3.777 | -3.612 | 1.038 | -5.973 | 6.74 | 11.31 | 0.91 | 1.03 | 2.26 | 2.2 | 1.248 |
| E1_pM_35 | rl_f4_zero_shot | 499 | 498 | 1 | 2 | 3.638 | 3.516 | -0.361 | 5.489 | 6.74 | 5.11 | 0.9 | 0.75 | 2.18 | 2.2 | 0.583 |
| E1_pM_35 | rl_f4_retrained | 500 | 499 | 1 | 1 | 3.727 | 3.571 | -0.763 | 5.8 | 6.74 | 1.1 | 0.91 | 0.92 | 2.18 | 2.2 | 0.635 |

## Headlines

- E1_pM_15 (rl_f4_zero_shot): Z -3.736 % on average over 499 paired instances; van share personal 6.74 -> 8.38 %, commercial 0.91 -> 1.02 %; 1 instance(s) newly solved, 1 lost.
- E1_pM_15 (rl_f4_retrained): Z -3.777 % on average over 499 paired instances; van share personal 6.74 -> 11.31 %, commercial 0.91 -> 1.03 %; 0 instance(s) newly solved, 1 lost.
- E1_pM_35 (rl_f4_zero_shot): Z +3.638 % on average over 498 paired instances; van share personal 6.74 -> 5.11 %, commercial 0.9 -> 0.75 %; 1 instance(s) newly solved, 2 lost.
- E1_pM_35 (rl_f4_retrained): Z +3.727 % on average over 499 paired instances; van share personal 6.74 -> 1.1 %, commercial 0.91 -> 0.92 %; 1 instance(s) newly solved, 1 lost.

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
| E1_pM_15 (rl_f4_zero_shot) | personal | 0.0-0.2 | 141 | 141 | 0.0 | 2.68 | 533.67 |
| E1_pM_15 (rl_f4_zero_shot) | personal | 0.2-0.4 | 744 | 651 | 0.0 | 1.37 | 608.28 |
| E1_pM_15 (rl_f4_zero_shot) | personal | 0.4-0.6 | 1649 | 1321 | 1.5 | 0.97 | 642.79 |
| E1_pM_15 (rl_f4_zero_shot) | personal | 0.6-0.8 | 2201 | 1896 | 6.6 | 0.77 | 679.95 |
| E1_pM_15 (rl_f4_zero_shot) | personal | 0.8-1.0 | 3036 | 2721 | 15.4 | 0.21 | 690.67 |
| E1_pM_15 (rl_f4_zero_shot) | commercial | 0.0-0.2 | 2831 | 2571 | 0.7 | 5.18 | 589.13 |
| E1_pM_15 (rl_f4_zero_shot) | commercial | 0.2-0.4 | 8236 | 6897 | 1.1 | 3.56 | 566.78 |
| E1_pM_15 (rl_f4_zero_shot) | commercial | 0.4-0.6 | 5810 | 4766 | 1.2 | 2.33 | 550.29 |
| E1_pM_15 (rl_f4_zero_shot) | commercial | 0.6-0.8 | 1331 | 1107 | 0.9 | 2.07 | 545.79 |
| E1_pM_15 (rl_f4_zero_shot) | commercial | 0.8-1.0 | 81 | 81 | 0.0 | 1.33 | 442.62 |
| E1_pM_15 (rl_f4_retrained) | personal | 0.0-0.2 | 140 | 140 | 0.7 | 6.03 | 550.76 |
| E1_pM_15 (rl_f4_retrained) | personal | 0.2-0.4 | 743 | 650 | 1.1 | 3.82 | 629.82 |
| E1_pM_15 (rl_f4_retrained) | personal | 0.4-0.6 | 1642 | 1315 | 3.2 | 1.6 | 656.14 |
| E1_pM_15 (rl_f4_retrained) | personal | 0.6-0.8 | 2192 | 1889 | 11.5 | 0.94 | 676.97 |
| E1_pM_15 (rl_f4_retrained) | personal | 0.8-1.0 | 3023 | 2709 | 18.1 | 0.19 | 678.39 |
| E1_pM_15 (rl_f4_retrained) | commercial | 0.0-0.2 | 2822 | 2563 | 1.1 | 5.25 | 588.55 |
| E1_pM_15 (rl_f4_retrained) | commercial | 0.2-0.4 | 8212 | 6879 | 1.3 | 3.54 | 563.82 |
| E1_pM_15 (rl_f4_retrained) | commercial | 0.4-0.6 | 5778 | 4737 | 0.8 | 2.0 | 543.4 |
| E1_pM_15 (rl_f4_retrained) | commercial | 0.6-0.8 | 1327 | 1104 | 0.7 | 1.58 | 532.75 |
| E1_pM_15 (rl_f4_retrained) | commercial | 0.8-1.0 | 81 | 81 | 0.0 | 0.0 | 425.45 |
| E1_pM_35 (rl_f4_zero_shot) | personal | 0.0-0.2 | 141 | 141 | 0.0 | 2.78 | 534.45 |
| E1_pM_35 (rl_f4_zero_shot) | personal | 0.2-0.4 | 742 | 650 | 0.0 | 1.87 | 685.08 |
| E1_pM_35 (rl_f4_zero_shot) | personal | 0.4-0.6 | 1645 | 1317 | 1.4 | 1.31 | 712.7 |
| E1_pM_35 (rl_f4_zero_shot) | personal | 0.6-0.8 | 2187 | 1883 | 2.4 | 0.83 | 737.28 |
| E1_pM_35 (rl_f4_zero_shot) | personal | 0.8-1.0 | 3025 | 2711 | 10.3 | 0.18 | 753.59 |
| E1_pM_35 (rl_f4_zero_shot) | commercial | 0.0-0.2 | 2816 | 2557 | 0.4 | 5.09 | 638.77 |
| E1_pM_35 (rl_f4_zero_shot) | commercial | 0.2-0.4 | 8212 | 6876 | 1.0 | 3.6 | 650.35 |
| E1_pM_35 (rl_f4_zero_shot) | commercial | 0.4-0.6 | 5796 | 4755 | 0.8 | 2.4 | 620.06 |
| E1_pM_35 (rl_f4_zero_shot) | commercial | 0.6-0.8 | 1325 | 1102 | 0.6 | 1.89 | 611.32 |
| E1_pM_35 (rl_f4_zero_shot) | commercial | 0.8-1.0 | 81 | 81 | 0.0 | 0.42 | 439.67 |
| E1_pM_35 (rl_f4_retrained) | personal | 0.0-0.2 | 141 | 141 | 0.0 | 5.53 | 547.11 |
| E1_pM_35 (rl_f4_retrained) | personal | 0.2-0.4 | 744 | 651 | 0.0 | 2.69 | 693.44 |
| E1_pM_35 (rl_f4_retrained) | personal | 0.4-0.6 | 1649 | 1321 | 1.5 | 1.77 | 721.57 |
| E1_pM_35 (rl_f4_retrained) | personal | 0.6-0.8 | 2201 | 1896 | 0.5 | 1.07 | 744.23 |
| E1_pM_35 (rl_f4_retrained) | personal | 0.8-1.0 | 3036 | 2721 | 1.6 | 0.25 | 765.64 |
| E1_pM_35 (rl_f4_retrained) | commercial | 0.0-0.2 | 2831 | 2571 | 1.1 | 5.46 | 645.4 |
| E1_pM_35 (rl_f4_retrained) | commercial | 0.2-0.4 | 8236 | 6897 | 1.2 | 3.59 | 649.83 |
| E1_pM_35 (rl_f4_retrained) | commercial | 0.4-0.6 | 5810 | 4766 | 0.9 | 2.31 | 612.95 |
| E1_pM_35 (rl_f4_retrained) | commercial | 0.6-0.8 | 1331 | 1107 | 0.3 | 1.76 | 607.62 |
| E1_pM_35 (rl_f4_retrained) | commercial | 0.8-1.0 | 81 | 81 | 0.0 | 0.0 | 431.22 |

Full columns in comparison.csv and comparison_by_income.csv.
