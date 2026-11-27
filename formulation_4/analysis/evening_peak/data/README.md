# Formulation-4 data with an evening demand peak

Same as `formulation_4/data/solomon_*.csv` except the column `request_min` of EV rows (Solomon 100-EV files only; the
Gehring-Homberger files are not included). Built by `../code/make_evening_peak_dataset.py`.

- Profile: (1 - w) uniform + w truncated normal, w = 0.4, peak at 0.75 of each EV's feasible window,
  sigma = 0.08.
- Order of the EVs over the day kept from Solomon (rank of the original request time; ties at random).
- Every request leaves time to finish charging by T: r_i <= L_i = T - 1.25 x (fastest charging path).
  EVs with L_i = 0: 0.
- Original data for comparison: 76 of 5600 EVs had a request
  after their L_i.

Per class (n = 100):

| class   |    T |   mean_need |   mean_L_over_T |   old_after_L |
|:--------|-----:|------------:|----------------:|--------------:|
| C1      | 1236 |      60.257 |           0.939 |             0 |
| C2      | 3390 |      62.888 |           0.977 |             0 |
| R1      |  230 |      55.327 |           0.699 |            57 |
| R2      | 1000 |      56.086 |           0.93  |             0 |
| RC1     |  240 |      61.548 |           0.679 |            19 |
| RC2     |  960 |      62.365 |           0.919 |             0 |

Run any formulation-4 solver on it by pointing it at this folder, e.g. from `formulation_4`:
`python methods/milp/code/solve_milp_f4.py ../formulation_4/analysis/evening_peak/data --out <new results folder>`
(use a new results folder: the default one holds the baseline results on the original data).
