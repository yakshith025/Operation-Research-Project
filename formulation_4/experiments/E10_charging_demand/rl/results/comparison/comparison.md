# E10: Charging demand (energy per EV) (RL)

**Question.** How does a change in charging demand -- the energy each EV needs -- affect the optimal solution? (The demand LEVEL, i.e. the number of EVs, is the n = 10..file-size sweep itself and is reported separately.)

Rubric factor 'charging demand'. The energy need E_i of every EV is multiplied by config.energy_scale (baseline 1.0). This scales both charging times (tau^F = 60 E / P^F at a station, tau^M by a van) and the energy drawn, so more energy per EV lengthens every session, tightens both the stations and the vans, and should raise cost and queues; less does the reverse. +-25 %. The RL generator applies the same scale when a variant model is retrained (rl_f4_generator.energy_scale).

Baseline: `rl_f4` in `methods/rl/results`. Changes are over paired instances (valid in both). Van shares are pooled over the free-choice EVs of the paired instances.

## Variants against the baseline

| variant | run_id | valid | paired | newly_valid | no_longer_valid | mean_dZ_% | median_dZ_% | mean_d_operator_cost_% | mean_d_customer_cost_% | baseline_share_van_personal_% | share_van_personal_% | baseline_share_van_commercial_% | share_van_commercial_% | vans_used | baseline_vans_used | mean_gap_to_milp_% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E10_demand_0.75 | rl_f4_retrained | 539 | 500 | 39 | 0 | -21.819 | -21.789 | -18.586 | -23.351 | 6.73 | 6.12 | 0.94 | 0.12 | 2.0 | 2.21 |  |
| E10_demand_1.25 | rl_f4_retrained | 421 | 421 | 0 | 79 | 22.188 | 21.956 | 19.326 | 23.575 | 6.28 | 5.48 | 0.4 | 0.77 | 2.19 | 1.99 |  |

## Headlines

- E10_demand_0.75 (rl_f4_retrained): Z -21.819 % on average over 500 paired instances; van share personal 6.73 -> 6.12 %, commercial 0.94 -> 0.12 %; 39 instance(s) newly solved, 0 lost.
- E10_demand_1.25 (rl_f4_retrained): Z +22.188 % on average over 421 paired instances; van share personal 6.28 -> 5.48 %, commercial 0.4 -> 0.77 %; 0 instance(s) newly solved, 79 lost.

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
| E10_demand_0.75 (rl_f4_retrained) | personal | 0.0-0.2 | 152 | 152 | 0.0 | 2.89 | 416.57 |
| E10_demand_0.75 (rl_f4_retrained) | personal | 0.2-0.4 | 824 | 704 | 0.0 | 2.18 | 511.17 |
| E10_demand_0.75 (rl_f4_retrained) | personal | 0.4-0.6 | 1823 | 1481 | 1.6 | 0.98 | 531.37 |
| E10_demand_0.75 (rl_f4_retrained) | personal | 0.6-0.8 | 2660 | 2302 | 4.7 | 1.07 | 566.58 |
| E10_demand_0.75 (rl_f4_retrained) | personal | 0.8-1.0 | 3370 | 3029 | 12.3 | 0.2 | 572.57 |
| E10_demand_0.75 (rl_f4_retrained) | commercial | 0.0-0.2 | 3314 | 3009 | 0.2 | 6.67 | 480.53 |
| E10_demand_0.75 (rl_f4_retrained) | commercial | 0.2-0.4 | 9041 | 7611 | 0.2 | 4.03 | 461.49 |
| E10_demand_0.75 (rl_f4_retrained) | commercial | 0.4-0.6 | 6358 | 5195 | 0.2 | 2.33 | 443.13 |
| E10_demand_0.75 (rl_f4_retrained) | commercial | 0.6-0.8 | 1527 | 1266 | 0.1 | 1.46 | 443.66 |
| E10_demand_0.75 (rl_f4_retrained) | commercial | 0.8-1.0 | 81 | 81 | 0.0 | 0.3 | 334.44 |
| E10_demand_1.25 (rl_f4_retrained) | personal | 0.0-0.2 | 104 | 104 | 0.0 | 4.11 | 603.35 |
| E10_demand_1.25 (rl_f4_retrained) | personal | 0.2-0.4 | 600 | 542 | 0.6 | 2.45 | 780.12 |
| E10_demand_1.25 (rl_f4_retrained) | personal | 0.4-0.6 | 1327 | 1032 | 0.8 | 0.81 | 837.09 |
| E10_demand_1.25 (rl_f4_retrained) | personal | 0.6-0.8 | 1542 | 1323 | 3.7 | 0.52 | 866.61 |
| E10_demand_1.25 (rl_f4_retrained) | personal | 0.8-1.0 | 2533 | 2251 | 10.1 | 0.11 | 879.87 |
| E10_demand_1.25 (rl_f4_retrained) | commercial | 0.0-0.2 | 2102 | 1912 | 0.5 | 3.08 | 733.28 |
| E10_demand_1.25 (rl_f4_retrained) | commercial | 0.2-0.4 | 7006 | 5795 | 0.9 | 2.42 | 742.86 |
| E10_demand_1.25 (rl_f4_retrained) | commercial | 0.4-0.6 | 4877 | 4000 | 0.8 | 1.21 | 697.04 |
| E10_demand_1.25 (rl_f4_retrained) | commercial | 0.6-0.8 | 1040 | 882 | 0.6 | 0.91 | 687.79 |
| E10_demand_1.25 (rl_f4_retrained) | commercial | 0.8-1.0 | 79 | 79 | 0.0 | 0.0 | 510.59 |

Full columns in comparison.csv and comparison_by_income.csv.
