# Formulation 4 -- rubric results (Solomon + Gehring-Homberger)

Every number is computed from the run tables (`runs.csv`), split by dataset. CSV files are in this folder.

## Experimental Evaluation -- solution quality (pooled over n)

| dataset | method | n | instances | valid | optimal | mean_Z | mean_gap_% | median_gap_% |
|---|---|---|---|---|---|---|---|---|
| Solomon | MILP | all | 560 | 474 | 277 | 43998.865 | 2.364 | 0.0 |
| Solomon | RL | all | 560 | 500 | 0 | 46507.879 |  |  |
| Solomon | greedy | all | 560 | 461 | 0 | 46057.454 |  |  |

## Experimental Evaluation -- computational performance / speed

`performance.csv` (per n) and the head-to-head below. RL speed-up = mean MILP solve time / mean RL solve time.

| dataset | milp_valid | rl_valid | paired | rl_only | milp_only | mean_RL-MILP_Z_% | median_% | mean_milp_s | mean_rl_s | speedup_x |
|---|---|---|---|---|---|---|---|---|---|---|
| Solomon | 474 | 500 | 474 | 26 | 0 | 0.852 | 0.761 | 139.693 | 0.119 | 1173.9 |

## Analysis -- how each factor changes the optimal solution

Full table: `sensitivity_summary.csv`. Rubric factors broken out in `sensitivity_mobile_chargers.csv` (E2), `sensitivity_station_capacity.csv` (E9), `sensitivity_charging_demand.csv` (E10). `dZ_%` is the mean change of Z on instances valid in both the variant and the baseline of the same dataset.

| experiment | factor | variant | label | dataset | run | paired | dZ_% | d_oper_% | d_cust_% | van_base_% | van_var_% | q_base | q_var | new | lost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E5 | personal c^{C,1} | E5_cC1_personal_7.5 | personal c^{C,1} = 7.5 (-25 %) | Solomon | RL retrained | 499 | -2.459 | -0.587 | -3.238 | 2.859 | 1.565 | 1.861 | 1.939 | 1 | 1 |
| E5 | personal c^{C,1} | E5_cC1_personal_7.5 | personal c^{C,1} = 7.5 (-25 %) | Solomon | MILP | 472 | -2.212 | -1.147 | -2.593 | 6.696 | 4.5 | 0.933 | 0.948 | 1 | 2 |
| E5 | personal c^{C,1} | E5_cC1_personal_12.5 | personal c^{C,1} = 12.5 (+25 %) | Solomon | RL retrained | 500 | 2.38 | 0.928 | 3.022 | 2.864 | 4.088 | 1.88 | 1.883 | 0 | 0 |
| E5 | personal c^{C,1} | E5_cC1_personal_12.5 | personal c^{C,1} = 12.5 (+25 %) | Solomon | MILP | 472 | 2.078 | 2.746 | 1.837 | 6.696 | 9.064 | 0.936 | 0.876 | 1 | 2 |
| E5 | commercial omega^1 | E5_omega1_commercial_4.5 | commercial downtime rate at theta = 1: 4.5 (-25 %) | Solomon | RL retrained | 499 | -2.521 | 0.17 | -3.695 | 2.854 | 2.801 | 1.859 | 1.896 | 1 | 1 |
| E5 | commercial omega^1 | E5_omega1_commercial_4.5 | commercial downtime rate at theta = 1: 4.5 (-25 %) | Solomon | MILP | 471 | -2.666 | -0.17 | -3.799 | 6.682 | 6.596 | 0.921 | 0.944 | 1 | 3 |
| E5 | commercial omega^1 | E5_omega1_commercial_7.5 | commercial downtime rate at theta = 1: 7.5 (+25 %) | Solomon | RL retrained | 499 | 2.752 | 0.14 | 3.917 | 2.854 | 2.901 | 1.859 | 1.855 | 0 | 1 |
| E5 | commercial omega^1 | E5_omega1_commercial_7.5 | commercial downtime rate at theta = 1: 7.5 (+25 %) | Solomon | MILP | 472 | 2.662 | 0.26 | 3.815 | 6.696 | 6.643 | 0.936 | 0.876 | 3 | 2 |
| E6 | personal mu^0 | E6_mu0_personal_1.125 | personal mu^0 = 1.125 (-25 %) | Solomon | RL retrained | 500 | -1.224 | 0.066 | -1.789 | 2.864 | 3.098 | 1.88 | 1.952 | 1 | 0 |
| E6 | personal mu^0 | E6_mu0_personal_1.125 | personal mu^0 = 1.125 (-25 %) | Solomon | MILP | 472 | -1.353 | -0.014 | -1.957 | 6.696 | 6.693 | 0.933 | 0.925 | 1 | 2 |
| E6 | personal mu^0 | E6_mu0_personal_1.875 | personal mu^0 = 1.875 (+25 %) | Solomon | RL retrained | 499 | 1.36 | 0.095 | 1.938 | 2.854 | 2.886 | 1.859 | 1.879 | 0 | 1 |
| E6 | personal mu^0 | E6_mu0_personal_1.875 | personal mu^0 = 1.875 (+25 %) | Solomon | MILP | 473 | 1.326 | -0.154 | 2.042 | 6.705 | 6.338 | 0.944 | 0.923 | 2 | 1 |
| E6 | commercial mu^0 | E6_mu0_commercial_1.125 | commercial mu^0 = 1.125 (-25 %) | Solomon | RL retrained | 500 | -6.246 | -0.121 | -9.008 | 2.864 | 2.86 | 1.88 | 1.909 | 0 | 0 |
| E6 | commercial mu^0 | E6_mu0_commercial_1.125 | commercial mu^0 = 1.125 (-25 %) | Solomon | MILP | 474 | -6.383 | 0.047 | -9.396 | 6.707 | 6.634 | 0.955 | 0.891 | 1 | 0 |
| E6 | commercial mu^0 | E6_mu0_commercial_1.875 | commercial mu^0 = 1.875 (+25 %) | Solomon | RL retrained | 499 | 6.334 | 0.121 | 9.154 | 2.854 | 2.945 | 1.859 | 1.869 | 1 | 1 |
| E6 | commercial mu^0 | E6_mu0_commercial_1.875 | commercial mu^0 = 1.875 (+25 %) | Solomon | MILP | 473 | 6.429 | 0.057 | 9.466 | 6.705 | 6.668 | 0.944 | 0.97 | 2 | 1 |
| E1 | p^M | E1_pM_15 | p^M = 15 Rs/kWh (= p^F) | Solomon | RL retrained | 499 | -3.777 | 1.038 | -5.973 | 2.854 | 4.378 | 1.859 | 1.918 | 0 | 1 |
| E1 | p^M | E1_pM_15 | p^M = 15 Rs/kWh (= p^F) | Solomon | MILP | 470 | -4.189 | 3.686 | -7.8 | 6.657 | 10.666 | 0.918 | 0.84 | 1 | 4 |
| E1 | p^M | E1_pM_35 | p^M = 35 Rs/kWh | Solomon | RL retrained | 499 | 3.727 | -0.763 | 5.8 | 2.854 | 0.769 | 1.859 | 1.94 | 1 | 1 |
| E1 | p^M | E1_pM_35 | p^M = 35 Rs/kWh | Solomon | MILP | 474 | 3.914 | -1.386 | 6.532 | 6.707 | 4.247 | 0.955 | 0.966 | 3 | 0 |
| E2 | vans | E2_vans_3 | 3 vans | Solomon | RL retrained | 466 | 0.086 | -0.12 | 0.178 | 2.602 | 2.414 | 1.36 | 1.397 | 0 | 34 |
| E2 | vans | E2_vans_3 | 3 vans | Solomon | MILP | 441 | -0.042 | -0.527 | 0.195 | 6.345 | 5.923 | 0.648 | 0.623 | 0 | 33 |
| E2 | vans | E2_vans_7 | 7 vans | Solomon | RL retrained | 500 | 0.004 | 0.057 | -0.002 | 2.864 | 2.779 | 1.88 | 1.938 | 6 | 0 |
| E2 | vans | E2_vans_7 | 7 vans | Solomon | MILP | 474 | 0.019 | 0.22 | -0.051 | 6.707 | 6.781 | 0.955 | 0.931 | 13 | 0 |
| E2 | seats | E2_seats_4 | 4 EVs per van | Solomon | RL retrained | 494 | 0.878 | 4.043 | -0.46 | 2.717 | 2.868 | 1.781 | 1.786 | 0 | 6 |
| E2 | seats | E2_seats_4 | 4 EVs per van | Solomon | MILP | 467 | 1.08 | 3.03 | 0.264 | 6.548 | 4.687 | 0.868 | 0.894 | 4 | 7 |
| E2 | seats | E2_seats_8 | 8 EVs per van | Solomon | RL retrained | 499 | 0.066 | -0.308 | 0.247 | 2.854 | 2.912 | 1.859 | 1.89 | 0 | 1 |
| E2 | seats | E2_seats_8 | 8 EVs per van | Solomon | MILP | 471 | -0.082 | -0.283 | 0.062 | 6.681 | 7.056 | 0.923 | 0.886 | 0 | 3 |
| E8 | battery | E8_battery_200 | B = 200 kWh | Solomon | RL retrained | 499 | -0.075 | -0.059 | -0.07 | 2.854 | 3.109 | 1.859 | 1.917 | 1 | 1 |
| E8 | battery | E8_battery_200 | B = 200 kWh | Solomon | MILP | 473 | -0.005 | -0.4 | 0.191 | 6.705 | 6.418 | 0.944 | 0.946 | 3 | 1 |
| E9 | guns_per_station | E9_guns_1 | 1 gun per station (half) | Solomon | RL retrained | 432 | 4.762 | 5.397 | 4.555 | 2.434 | 5.799 | 0.858 | 2.828 | 0 | 68 |
| E9 | guns_per_station | E9_guns_4 | 4 guns per station (double) | Solomon | RL retrained | 500 | -2.215 | -0.903 | -2.723 | 2.864 | 2.144 | 1.88 | 0.371 | 9 | 0 |
| E9 | station_power_kW | E9_power_11 | 11 kW per gun (half) | Solomon | RL retrained | 398 | 20.126 | 25.613 | 18.425 | 2.524 | 18.73 | 0.898 | 0.673 | 0 | 102 |
| E9 | station_power_kW | E9_power_44 | 44 kW per gun (double) | Solomon | RL retrained | 500 | -12.296 | -1.543 | -16.973 | 2.864 | 0.002 | 1.88 | 1.67 | 32 | 0 |
| E10 | energy_scale | E10_demand_0.75 | energy per EV x0.75 (-25 %) | Solomon | RL retrained | 500 | -21.819 | -18.586 | -23.351 | 2.864 | 2.446 | 1.88 | 1.843 | 39 | 0 |
| E10 | energy_scale | E10_demand_1.25 | energy per EV x1.25 (+25 %) | Solomon | RL retrained | 421 | 22.188 | 19.326 | 23.575 | 2.577 | 2.274 | 1.171 | 1.256 | 0 | 79 |

## Analysis -- charging demand LEVEL (number of EVs)

`charging_demand_by_n.csv`: how the baseline solution moves as the number of EVs grows.

| dataset | method | n | valid | mean_Z | van_share_% | queue_min | time_min | vans_used |
|---|---|---|---|---|---|---|---|---|
| Solomon | MILP | 10 | 55 | 9291.601 | 10.78 | 0.057 | 58.507 | 0.909 |
| Solomon | MILP | 20 | 55 | 18142.993 | 7.66 | 0.104 | 60.895 | 1.218 |
| Solomon | MILP | 30 | 55 | 26373.858 | 4.124 | 0.186 | 61.713 | 1.4 |
| Solomon | MILP | 40 | 55 | 35499.304 | 4.851 | 0.499 | 63.585 | 1.782 |
| Solomon | MILP | 50 | 53 | 44276.67 | 4.542 | 1.158 | 63.955 | 2.302 |
| Solomon | MILP | 60 | 49 | 53041.204 | 7.504 | 1.713 | 63.808 | 2.796 |
| Solomon | MILP | 70 | 44 | 60523.693 | 6.777 | 2.035 | 63.223 | 2.886 |
| Solomon | MILP | 80 | 39 | 68958.802 | 7.436 | 1.809 | 63.195 | 3.179 |
| Solomon | MILP | 90 | 36 | 78370.535 | 8.025 | 1.529 | 64.567 | 3.444 |
| Solomon | MILP | 100 | 33 | 85578.121 | 5.624 | 1.382 | 62.676 | 3.455 |
| Solomon | RL | 10 | 55 | 9335.545 | 6.776 | 0.057 | 59.315 | 0.909 |
| Solomon | RL | 20 | 55 | 18249.097 | 3.811 | 0.14 | 61.355 | 1.218 |
| Solomon | RL | 30 | 55 | 26529.833 | 1.067 | 0.348 | 62.304 | 1.4 |
| Solomon | RL | 40 | 55 | 35768.021 | 0.78 | 0.882 | 64.735 | 1.709 |
| Solomon | RL | 50 | 53 | 44579.381 | 1.251 | 1.87 | 65.1 | 2.17 |
| Solomon | RL | 60 | 50 | 53763.766 | 2.658 | 2.647 | 65.872 | 2.64 |
| Solomon | RL | 70 | 47 | 62017.712 | 2.689 | 3.192 | 66.74 | 2.787 |
| Solomon | RL | 80 | 47 | 71237.511 | 2.7 | 3.696 | 67.813 | 3.17 |
| Solomon | RL | 90 | 45 | 80886.077 | 3.773 | 4.042 | 69.216 | 3.578 |
| Solomon | RL | 100 | 38 | 88332.571 | 3.305 | 3.279 | 68.168 | 3.395 |

## Analysis -- where mobile chargers significantly improve the network

`mobile_charger_impact.csv`: stranded EVs (only a van can serve them), van share, and the fleet-size feasibility gain (how many more instances get a plan with 7 vans than with 3), by dataset x class x n. The gain is largest in the short-period classes (R1/RC1) and at large n.

| dataset | class | n | instances | stranded | van_share_% | valid_3v | valid_5v | valid_7v | gain_7-3 |
|---|---|---|---|---|---|---|---|---|---|
| Solomon | C1 | 90 | 9 | 14.0 | 7.489 | 8 | 9 | 9 | 1 |
| Solomon | C1 | 100 | 9 | 15.0 | 4.237 | 8 | 8 | 9 | 1 |
| Solomon | C2 | 90 | 8 | 13.0 | 8.45 | 7 | 8 | 8 | 1 |
| Solomon | C2 | 100 | 8 | 15.0 | 3.886 | 7 | 7 | 8 | 1 |
| Solomon | R1 | 60 | 12 | 8.0 | 3.814 | 6 | 7 | 7 | 1 |
| Solomon | R1 | 70 | 12 | 10.0 | 6.914 | 4 | 7 | 7 | 3 |
| Solomon | R1 | 80 | 12 | 11.0 | 9.167 | 1 | 3 | 7 | 6 |
| Solomon | R1 | 90 | 12 |  |  | 0 | 0 | 7 | 7 |
| Solomon | R1 | 100 | 12 |  |  | 0 | 0 | 5 | 5 |
| Solomon | R2 | 100 | 11 | 15.0 | 7.245 | 9 | 11 | 11 | 2 |
| Solomon | RC1 | 50 | 8 | 6.0 | 12.014 | 6 | 7 | 7 | 1 |
| Solomon | RC1 | 70 | 8 | 8.0 | 11.3 | 3 | 1 | 4 | 1 |
| Solomon | RC1 | 80 | 8 |  |  | 0 | 0 | 4 | 4 |
| Solomon | RC1 | 90 | 8 |  |  | 0 | 0 | 4 | 4 |
| Solomon | RC1 | 100 | 8 |  |  | 0 | 0 | 1 | 1 |
| Solomon | RC2 | 100 | 8 | 15.0 | 6.4 | 7 | 7 | 8 | 1 |

