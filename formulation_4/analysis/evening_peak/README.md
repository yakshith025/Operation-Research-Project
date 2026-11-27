# Evening-peak experiments (27 Nov 2026)

Question: do the 560 EV-Solomon instances (56 Solomon files × first n EVs, n = 10, 20, …, 100) show an evening
demand peak, and if not, what do they look like with one?

1. **Original data: no evening peak.** Requests are front-loaded (many at t = 0) and fade towards T. On average
   7.3 % of requests fall in the last third of T (33 % if uniform); the request rate there is about 5× lower than
   earlier (median ratio 0.18). The 12 instances flagged (C202–C204, mostly small n) rest on 2–8 requests. Cause:
   the request time is Solomon's ready time, which is always early enough for the customer to be served before
   the depot closes.
2. **Reshaped data (`data/`): evening peak added.** Only `request_min` of EV rows changes; everything else is the
   formulation-4 dataset (`../formulation_4/data`). 284 / 560 instances are flagged (median late/earlier
   ratio 1.33); the peak sits at about 0.7 T in C1, C2, R2, RC2. In R1 / RC1 (T = 230–240 min) a full charge
   takes up to about 110 min, so to stay finishable by T the peak lands at about 0.5 T.
   Feasibility: the greedy heuristic of the formulation-4 MILP finds a plan for 471 / 560 instances (original
   data: 461 / 560); the 89 failures are R1 / RC1, as before. MILP and RL have **not** been run on it yet.

## Folder

```
formulation_4/analysis/evening_peak/
├── README.md
├── data/                          the reshaped instances (formulation-4 format, Solomon 100-EV files only)
│   ├── solomon_config.csv         copied unchanged
│   ├── solomon_data.csv           request_min of EV rows reshaped; segment, theta kept
│   ├── request_mapping.csv        per EV: old and new request time, rank, latest feasible time L_i
│   ├── settings.json, README.md   profile parameters, seed, per-class summary
├── code/
│   ├── plot_temporal_demand.py        profiles of the original data       -> results/original_profiles/
│   ├── evening_peak_check.py          evening-peak metrics of 560 instances -> results/evening_peak_check_*/
│   ├── make_evening_peak_dataset.py   builds data/
│   └── compare_profiles.py            original vs reshaped figures         -> results/evening_peak_check_reshaped/
└── results/
    ├── original_profiles/                       per-class demand profiles, overview heatmap / cumulative curves
    ├── evening_peak_check_original/             evening = last third of T
    ├── evening_peak_check_original_late0.75/    evening = last quarter of T
    └── evening_peak_check_reshaped/             same check on data/, plus before_after_*.png
```

## How the request times are reshaped (`code/make_evening_peak_dataset.py`)

- Target profile on u = t / L_i: (1 − w) Uniform(0, 1) + w truncated Normal(μ, σ), default w = 0.4, μ = 0.75,
  σ = 0.08.
- EVs keep Solomon's order (rank of the original request time, ties such as t = 0 broken at random with a seed
  per file); the EV of rank k gets the target quantile (k − 0.5) / 100.
- r_i = ⌊u_k · L_i⌋ with L_i = T − 1.25 × (fastest way to finish charging: drive to a reachable station + station
  charging, or van charging + van drive back to the depot), so every EV can still finish by T (constraint 21).

## Commands (from `code/`)

```
python make_evening_peak_dataset.py --force                      # rebuild data/ (deterministic)
python make_evening_peak_dataset.py --w 0.5 --mu 0.8 --force     # another peak
python plot_temporal_demand.py [--n 30]                          # original profiles
python evening_peak_check.py [--late 0.75]                       # check the original data
python evening_peak_check.py --data ../data --out ../results/evening_peak_check_reshaped
python compare_profiles.py
```

Solve the reshaped instances with the formulation-4 code (from `../formulation_4`; use a new results
folder so the baseline results on the original data stay as they are):

```
python methods/milp/code/solve_milp_f4.py ../formulation_4/analysis/evening_peak/data --out ../formulation_4/analysis/evening_peak/results/milp_baseline
python methods/rl/code/solve_rl_f4.py --data ../formulation_4/analysis/evening_peak/data ...
```
