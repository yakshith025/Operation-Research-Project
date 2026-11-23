# RL2: reinforcement learning method 2

RL2 is RL1's method with two improvements aimed at where RL1 lost most against the MILP (n = 60–90, class C2):

1. **Active search on every test instance** (Efficient Active Search, EAS-Emb variant; Hottung, Kwon, Tierney, ICLR 2022). After the usual sampling, the trained network is frozen. Only this instance's embeddings are tuned on this instance, for a few seconds, with the same REINFORCE update plus an imitation term that pulls towards the best plan found so far.
2. **Changed training:**
   - 64 instances per batch instead of 32, for a less noisy policy gradient;
   - more batches with large n (n drawn with probability ∝ n instead of uniformly);
   - a larger validation set (4 sizes × 50 instances instead of 3 × 24), so the best epoch is chosen less by chance.

Everything else is RL1 unchanged: the simulator, network, warm-up, REINFORCE with a shared baseline, stop rules (at least 200 epochs, early stop after 10 without improvement, at most 500) and a fresh start at epoch 1 on every run.

**RL1 is not touched.** RL2 reads the **same data** as RL1 (`../dataset_ev_solomon/csv` and `income/`, nothing copied) and the same project code (`../code`). It writes only into this folder.

## Folder

```
RL2_reinforcement_learning_method_2/
├── README.md            this file
├── code/
│   ├── rl2_data.py      CSV dataset -> instance dict (copy of RL1's)
│   ├── rl2_generator.py random training instances (copy of RL1's)
│   ├── rl2_env.py       batched simulator of formulation 3 (copy of RL1's)
│   ├── rl2_model.py     policy network (copy of RL1's, same architecture)
│   ├── train_rl2.py     training: RL1's procedure + batch 64, size weighting, larger validation   [changed]
│   ├── rl2_eas.py       active search for one instance (EAS-Emb)                                 [new]
│   ├── solve_rl2.py     two runs per instance: sampling (as RL1) and active search                [changed]
│   ├── compare_rl2.py   RL2-EAS vs RL2 vs RL1 vs greedy, MILP as reference                        [new]
│   ├── test_rl2.py      RL1's self-checks + check 6 for active search                            [changed]
│   ├── rl2_params.json  all RL2 settings (new: n_weight_power, active_search section)
│   ├── run_rl2.sh       everything in one command on the server
│   └── requirements.txt numpy, torch
├── papers/README.md     the paper behind the new part (the RL1 papers are in ../RL1_.../papers)
├── model/               written by train_rl2.py
└── results/             written by solve_rl2.py / compare_rl2.py (format of code/results_io.py)
```

The code is a copy of RL1's, so this experiment keeps the exact code that produced its results even if RL1 changes. The three files marked [changed] and the [new] ones hold the differences.

## What a solve produces

For each instance (file + first *n* EVs), `solve_rl2.py` writes **two runs** to `results/runs.csv` and `plan.csv`:

| run id | method | what |
|---|---|---|
| `rl2_sampling` | `RL2` | As RL1: 8 map symmetries × (1 greedy + 64 sampled) plans; the best is kept. |
| `rl2_eas` | `RL2-EAS` | Active search starting from that plan, for up to 10 s / 200 iterations. Each iteration: 8 symmetries × 32 sampled plans, loss = REINFORCE (shared baseline) + 0.01 × imitation of the incumbent, then one Adam step on the instance embeddings. The best plan seen is kept, so it is never worse than `rl2_sampling`. |

Both plans are scored by `check_plan_f3` through `results_io.evaluate`, like every other method. `solve_s` of `rl2_eas` includes the sampling step. Comparing the runs separates the two changes:
- `rl2_sampling` vs RL1's `rl1` shows the effect of the new training;
- `rl2_eas` vs `rl2_sampling` shows the effect of active search.

Ablation (optional): active search on RL1's trained model, without retraining. This reads RL1's model and does not change it:
```
python solve_rl2.py --model ../../RL1_reinforcement_learning_method_1/model/model_best.pt --run-id-prefix rl1model
python compare_rl2.py --prefix rl1model
```

`compare_rl2.py` joins RL2's runs with RL1's `results/runs.csv` and the MILP's `new_results/runs.csv` (MILP and greedy). It writes `results/comparison.csv` and `results/comparison_summary.md`:
- per method: valid plans, plans where the MILP has none, gap to the MILP and to proven optima, how often it beats greedy, seconds per instance;
- the mean gap by class and by n;
- what active search changed.

## Running on the server

RL2 goes next to RL1 inside the server's `rl1/` folder, where the data, `code/` and `new_results/` already are:

```
or_project/project/rl1/
├── code/  dataset_ev_solomon/  new_results/      (already there, shared)
├── RL1_reinforcement_learning_method_1/          (already there, not changed)
└── RL2_reinforcement_learning_method_2/          (new)
```

1. Upload (laptop, PowerShell):
```
scp -r "E:\AUG-2026\02_CP320_OR_for_Mobility\project\RL2_reinforcement_learning_method_2" romit@10.72.19.181:/mnt/DataDrive_2/Yakshith/or_project/project/rl1/
```
2. On the server:
```
conda activate yakshith
cd /mnt/DataDrive_2/Yakshith/or_project/project/rl1/RL2_reinforcement_learning_method_2/code
python test_rl2.py --quick
nohup bash run_rl2.sh 8 32 > run_rl2.log 2>&1 &
```
`run_rl2.sh` runs the self-checks, the training (new model from epoch 1), the 560 Solomon instances with both runs, and the comparison.

3. Bring the results back (laptop, PowerShell):
```
scp -r romit@10.72.19.181:/mnt/DataDrive_2/Yakshith/or_project/project/rl1/RL2_reinforcement_learning_method_2/results romit@10.72.19.181:/mnt/DataDrive_2/Yakshith/or_project/project/rl1/RL2_reinforcement_learning_method_2/model "E:\AUG-2026\02_CP320_OR_for_Mobility\project\RL2_reinforcement_learning_method_2\"
```

**Time on the RTX 4090** (estimates from the RL1 run):
- Training: an epoch now has twice as many rollouts as RL1's (about 10 s there), so expect roughly 15–20 s. 200–500 epochs take about 1–2.5 h.
- Solving: the active search is capped at 10 s per instance, so 560 instances take at most about 1.6 h.

## Checks made when the code was written (22 Nov 2026, laptop CPU)

- `test_rl2.py --quick` passed all six checks. That includes the new check 6: the replayed incumbent rebuilds the same plan, active search never returns a worse plan, its plan passes the checker, and the starting embeddings are left untouched.
- A 2-epoch training run and `compare_rl2.py` ran end to end (validation with 200 instances at n = 20/50/80/100).
- **Active search on RL1's trained model** (the ablation, laptop CPU, 30 iterations each; RL1 itself was only read). The three instances had the largest gaps in an earlier sample:

  | instance | sampling, gap to MILP | after active search | change in Z |
  |---|---|---|---|
  | EV-R208 n=60 | 3.92 % | 1.25 % | −2.57 % |
  | EV-C104 n=90 | 2.59 % | 0.81 % | −1.74 % |
  | EV-R112 n=80 | 1.76 % | −0.23 % (cheaper than the MILP's 300 s plan) | −1.96 % |

  The mean gap went from 2.76 % to 0.61 %. For comparison, 8× more sampling on the first two instances only reached 3.25 % and 0.97 %. The best plans were found late (iterations 23–28 of 30), so more iterations may help further. On the CPU an iteration took 6–14 s. On the 4090 it should take a fraction of a second, so the 10 s limit should give a similar or larger number of iterations. Nothing from these tests is in `model/` or `results/`.
