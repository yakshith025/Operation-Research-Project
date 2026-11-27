# E9: Station capacity (charging points and power) (RL)

**Question.** How do changes in station capacity -- the number of charging points per station and the power per point -- affect the optimal solution (cost, queues, van use)?

Rubric factor 'station capacities'. Two levers, one change per variant: the number of guns per fixed charging station (config.guns_per_station, baseline 2) and the charging power per gun in kW (config.station_power_kW, baseline 22). Applied in memory with --set (no data copy). Fewer or weaker guns make the fixed stations a tighter bottleneck (longer queues, slower charging), which should push more EVs onto vans and raise Z; more or stronger guns do the reverse. Power enters tau^F = 60 E / P^F, so it shortens every FCS session without adding parallel capacity.

Baseline: `rl_f4` in `baseline_RL/results`. Changes are over paired instances (valid in both). Van shares are pooled over the free-choice EVs of the paired instances.

## Variants against the baseline

| variant | run_id | valid | paired | newly_valid | no_longer_valid | mean_dZ_% | median_dZ_% | mean_d_operator_cost_% | mean_d_customer_cost_% | baseline_share_van_personal_% | share_van_personal_% | baseline_share_van_commercial_% | share_van_commercial_% | vans_used | baseline_vans_used | mean_gap_to_milp_% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E9_guns_1 | rl_f4_retrained | 432 | 432 | 0 | 68 | 4.762 | 2.216 | 5.397 | 4.555 | 6.18 | 16.74 | 0.04 | 1.83 | 2.29 | 1.92 |  |
| E9_guns_4 | rl_f4_retrained | 509 | 500 | 9 | 0 | -2.215 | -0.565 | -0.903 | -2.723 | 6.73 | 5.29 | 0.94 | 0.01 | 2.13 | 2.21 |  |
| E9_power_11 | rl_f4_retrained | 398 | 398 | 0 | 102 | 20.126 | 19.035 | 25.613 | 18.425 | 6.3 | 58.74 | 0.15 | 2.27 | 3.25 | 1.93 |  |
| E9_power_44 | rl_f4_retrained | 532 | 500 | 32 | 0 | -12.296 | -11.789 | -1.543 | -16.973 | 6.73 | 0.0 | 0.94 | 0.01 | 2.12 | 2.21 |  |

## Headlines

- E9_guns_1 (rl_f4_retrained): Z +4.762 % on average over 432 paired instances; van share personal 6.18 -> 16.74 %, commercial 0.04 -> 1.83 %; 0 instance(s) newly solved, 68 lost.
- E9_guns_4 (rl_f4_retrained): Z -2.215 % on average over 500 paired instances; van share personal 6.73 -> 5.29 %, commercial 0.94 -> 0.01 %; 9 instance(s) newly solved, 0 lost.
- E9_power_11 (rl_f4_retrained): Z +20.126 % on average over 398 paired instances; van share personal 6.3 -> 58.74 %, commercial 0.15 -> 2.27 %; 0 instance(s) newly solved, 102 lost.
- E9_power_44 (rl_f4_retrained): Z -12.296 % on average over 500 paired instances; van share personal 6.73 -> 0.0 %, commercial 0.94 -> 0.01 %; 32 instance(s) newly solved, 0 lost.

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
| E9_guns_1 (rl_f4_retrained) | personal | 0.0-0.2 | 105 | 105 | 0.0 | 7.56 | 529.3 |
| E9_guns_1 (rl_f4_retrained) | personal | 0.2-0.4 | 605 | 537 | 2.2 | 4.25 | 668.81 |
| E9_guns_1 (rl_f4_retrained) | personal | 0.4-0.6 | 1276 | 997 | 5.1 | 1.41 | 702.63 |
| E9_guns_1 (rl_f4_retrained) | personal | 0.6-0.8 | 1585 | 1372 | 14.4 | 0.93 | 731.57 |
| E9_guns_1 (rl_f4_retrained) | personal | 0.8-1.0 | 2487 | 2217 | 27.7 | 0.3 | 713.68 |
| E9_guns_1 (rl_f4_retrained) | commercial | 0.0-0.2 | 2112 | 1918 | 1.6 | 6.2 | 626.96 |
| E9_guns_1 (rl_f4_retrained) | commercial | 0.2-0.4 | 6785 | 5622 | 2.4 | 4.44 | 626.97 |
| E9_guns_1 (rl_f4_retrained) | commercial | 0.4-0.6 | 4739 | 3902 | 1.3 | 2.43 | 594.08 |
| E9_guns_1 (rl_f4_retrained) | commercial | 0.6-0.8 | 1019 | 853 | 0.9 | 1.78 | 577.06 |
| E9_guns_1 (rl_f4_retrained) | commercial | 0.8-1.0 | 77 | 77 | 0.0 | 1.18 | 427.62 |
| E9_guns_4 (rl_f4_retrained) | personal | 0.0-0.2 | 144 | 144 | 0.0 | 1.47 | 527.37 |
| E9_guns_4 (rl_f4_retrained) | personal | 0.2-0.4 | 764 | 665 | 0.0 | 0.46 | 639.31 |
| E9_guns_4 (rl_f4_retrained) | personal | 0.4-0.6 | 1699 | 1366 | 1.3 | 0.08 | 660.4 |
| E9_guns_4 (rl_f4_retrained) | personal | 0.6-0.8 | 2320 | 1999 | 3.1 | 0.1 | 689.89 |
| E9_guns_4 (rl_f4_retrained) | personal | 0.8-1.0 | 3130 | 2806 | 10.0 | 0.0 | 718.88 |
| E9_guns_4 (rl_f4_retrained) | commercial | 0.0-0.2 | 2959 | 2686 | 0.0 | 1.88 | 592.36 |
| E9_guns_4 (rl_f4_retrained) | commercial | 0.2-0.4 | 8475 | 7105 | 0.0 | 1.03 | 579.1 |
| E9_guns_4 (rl_f4_retrained) | commercial | 0.4-0.6 | 5970 | 4891 | 0.0 | 0.37 | 553.97 |
| E9_guns_4 (rl_f4_retrained) | commercial | 0.6-0.8 | 1388 | 1155 | 0.0 | 0.24 | 549.97 |
| E9_guns_4 (rl_f4_retrained) | commercial | 0.8-1.0 | 81 | 81 | 0.0 | 0.0 | 405.35 |
| E9_power_11 (rl_f4_retrained) | personal | 0.0-0.2 | 93 | 93 | 5.4 | 1.8 | 563.69 |
| E9_power_11 (rl_f4_retrained) | personal | 0.2-0.4 | 553 | 503 | 6.0 | 0.74 | 781.25 |
| E9_power_11 (rl_f4_retrained) | personal | 0.4-0.6 | 1254 | 967 | 29.0 | 0.25 | 814.27 |
| E9_power_11 (rl_f4_retrained) | personal | 0.6-0.8 | 1386 | 1185 | 58.4 | 0.15 | 752.25 |
| E9_power_11 (rl_f4_retrained) | personal | 0.8-1.0 | 2380 | 2107 | 87.6 | 0.05 | 606.22 |
| E9_power_11 (rl_f4_retrained) | commercial | 0.0-0.2 | 1890 | 1723 | 2.8 | 1.81 | 705.34 |
| E9_power_11 (rl_f4_retrained) | commercial | 0.2-0.4 | 6608 | 5456 | 2.4 | 1.39 | 734.11 |
| E9_power_11 (rl_f4_retrained) | commercial | 0.4-0.6 | 4592 | 3766 | 2.0 | 0.38 | 723.73 |
| E9_power_11 (rl_f4_retrained) | commercial | 0.6-0.8 | 956 | 819 | 1.7 | 0.29 | 740.18 |
| E9_power_11 (rl_f4_retrained) | commercial | 0.8-1.0 | 78 | 78 | 1.3 | 0.81 | 630.83 |
| E9_power_44 (rl_f4_retrained) | personal | 0.0-0.2 | 146 | 146 | 0.0 | 2.57 | 486.57 |
| E9_power_44 (rl_f4_retrained) | personal | 0.2-0.4 | 809 | 690 | 0.0 | 1.42 | 570.11 |
| E9_power_44 (rl_f4_retrained) | personal | 0.4-0.6 | 1782 | 1445 | 0.0 | 1.07 | 549.14 |
| E9_power_44 (rl_f4_retrained) | personal | 0.6-0.8 | 2597 | 2251 | 0.0 | 0.61 | 538.95 |
| E9_power_44 (rl_f4_retrained) | personal | 0.8-1.0 | 3321 | 2984 | 0.0 | 0.11 | 532.39 |
| E9_power_44 (rl_f4_retrained) | commercial | 0.0-0.2 | 3238 | 2941 | 0.0 | 5.64 | 555.45 |
| E9_power_44 (rl_f4_retrained) | commercial | 0.2-0.4 | 8892 | 7480 | 0.0 | 3.38 | 523.83 |
| E9_power_44 (rl_f4_retrained) | commercial | 0.4-0.6 | 6245 | 5104 | 0.0 | 2.22 | 483.57 |
| E9_power_44 (rl_f4_retrained) | commercial | 0.6-0.8 | 1489 | 1235 | 0.0 | 1.55 | 473.32 |
| E9_power_44 (rl_f4_retrained) | commercial | 0.8-1.0 | 81 | 81 | 0.0 | 0.63 | 329.29 |

Full columns in comparison.csv and comparison_by_income.csv.
