# RL1: reinforcement learning method 1

**Attention policy trained with REINFORCE and a shared baseline**, for `main_formulation_3` (EV assignment to FCS guns or mobile vans, van routing, income index θ).

It is a learning-to-optimise method. A neural policy builds the whole plan one decision at a time: "EV *i* goes to resource *k*", where *k* is one of the guns or one of the vans. The network is trained only on random instances made with the dataset's rules and then tested on the EV-Solomon CSV files. Every plan is scored by the same checker as the MILP (`code/check_plan_f3.py` through `code/results_io.py`), so its Z means exactly what the MILP's Z means.

## Folder

```
RL1_reinforcement_learning_method_1/
├── README.md            this file
├── code/
│   ├── rl1_data.py      CSV dataset (dataset_ev_solomon/csv + income) -> instance dict (same layout as the MILP's)
│   ├── rl1_generator.py random training instances built with the EV-Solomon rules (templates read from the CSVs)
│   ├── rl1_env.py       batched simulator of formulation 3: masks, start times, increase of Z per decision
│   ├── rl1_model.py     policy network: Transformer encoder + two-step pointer decoder (EV, then resource)
│   ├── train_rl1.py     training (REINFORCE, shared baseline, validation against the greedy rule)
│   ├── solve_rl1.py     solves the instances, writes runs.csv / plan.csv / meta.json / logs.zip
│   ├── compare_rl1.py   RL1 vs MILP vs greedy (comparison.csv, comparison_summary.md)
│   ├── test_rl1.py      self-checks (CSV = text loader, simulator = greedy_f3 + checker, ...)
│   ├── rl1_params.json  all RL1 settings (generator, network, training, inference)
│   ├── run_rl1.sh       everything in one command on the server
│   └── requirements.txt numpy, torch
├── papers/              the four papers the method is built from, plus README.md (what comes from which paper)
├── model/               written by train_rl1.py: model_best.pt, model_last.pt, train_log.csv, train_meta.json
└── results/             written by solve_rl1.py / compare_rl1.py (format of code/results_io.py)
```

The code imports the project's own modules from `../../code` (`results_io`, `check_plan_f3`, `make_income`, `solve_milp_f3`, `make_ev_solomon`), so the RL1 folder must sit inside `project/`, next to `code/` and `dataset_ev_solomon/`. Gurobi is not needed.

## Method

### The decision process (`rl1_env.py`)

| | |
|---|---|
| **State** | Which EVs are placed. Per gun: the minute it becomes free. Per van: where it is (depot or its last EV), when it is free, battery left, EVs served so far. |
| **Action** | A pair (EV *i*, resource *k*). *k* is one of the G guns (2 per station) or one of the M vans. EV *i* goes to the end of *k*'s queue or route. |
| **Timing** | Not learned. Every session starts as early as possible: gun $S_i=\max(r_i+t_{if},\ \text{free}_g)$, van $S_i=\max(r_i,\ \text{free}_m+t_{pos,i})$. Once assignment and order are fixed this is optimal, because every time cost of formulation 3 grows with a later start. (The fixed-plan LP of `compute_duals.py` agrees: LP value = plan cost.) |
| **Mask** | Only feasible pairs can be chosen. Gun: station within range $R_i$, queue $\le W_i$, done by $T$. Van: fewer than $Q$ EVs so far, battery $\ge E_i/\eta+\beta(d_{pos,i}+d_{i,0})$, back at the depot by $T$. Symmetry (lossless): of two idle guns of one station only the first is offered; of the unused, identical vans only the first. |
| **Reward** | Minus the increase of $Z=Z_1+\dots+Z_6$ caused by the action, using the same terms as `check_plan_f3`. The van term $c_M(d_{pos,i}+d_{i,0}-d_{pos,0})$ already contains the way back, so the increases add up to exactly Z. |
| **End** | Every EV is placed, or can no longer be placed. An EV with no feasible pair never gets one later (queues only grow, batteries only shrink, triangle inequality), so it is dropped at once. Training adds a penalty per dropped EV. In the results the plan is marked invalid (see Outputs). |

### The policy (`rl1_model.py`)

- **Encoder** (Kool et al. 2019): each node gets a linear embedding of its static features, then 3 Transformer layers run over all nodes. The nodes are the EVs (position, request, charging times, *W*, θ, energy, range, reachable stations, ...), the stations and the depot. There is no positional encoding, so one network handles any *n*, any number of stations and any number of vans.
- **Decoder, step 1, choose an EV:**
  - The context is the mean embedding of the open EVs, the depot and four global numbers. A multi-head glimpse over the open EVs turns it into a query.
  - The score of EV *i* is $C\tanh(q\cdot W h_i/\sqrt d + \text{MLP}(\phi_i))$.
  - $\phi_i$ holds the options left, the cheapest option, the regret (second cheapest minus cheapest), the earliest finish, cheapest gun vs cheapest van, the request relative to "now", and the EV's place in the queue of open requests (earliest first, ties: higher income first).
- **Decoder, step 2, choose its resource** (the vehicle selection of Li et al. 2021, with guns that stay put and vans that move):
  - The key of a gun is its station's key; the key of a van is the key of its current node. Each gets a linear term in the resource's state.
  - The score is $C\tanh(q_i\cdot \text{key}_k/\sqrt d + \text{MLP}(\text{pair features}))$. The pair features are the increase of Z, the extra over the cheapest option, the wait, the finish and the battery after.
- $\log\pi(\text{action})=\log\pi(i)+\log\pi(k\mid i)$.

### Training (`train_rl1.py`)

- **Warm-up (imitation):** the first `imitation_batches` batches (default 500) teach the policy the actions of the greedy cost rule of `solve_milp_f3`. This is teacher forcing: the loss is the negative log-likelihood of the rule's choice at each step. Such a batch is about 8× cheaper than a REINFORCE batch. REINFORCE then starts from a sensible policy instead of random weights and has to beat the rule's choices to lower the cost. With `--imitation-batches 0` it is pure RL from random weights.
- **Batches:** each batch has `batch_instances` random instances with the same *n*, with *n* drawn from [`n_min`, `n_max`]. Each instance gets a random map symmetry and is rolled out `samples` times.
- **Loss:**
  - Cost of a rollout: $\text{cost}=Z+\text{drop\_penalty}\cdot\text{scale}\cdot(\text{dropped EVs})$.
  - Shared baseline (POMO): $\text{adv}=(\text{cost}-\overline{\text{cost}})/\overline{\text{cost}}$, where the mean is over the instance's own rollouts.
  - REINFORCE loss: $\text{mean}(\text{adv}\cdot\log\pi)$.
  - Optimiser: Adam, gradient clipping, slow learning-rate decay. After the warm-up REINFORCE uses a 5× smaller learning rate (`lr_reinforce`). With the warm-up's rate, the noisy policy gradient first undid part of the warm-up.
- **Validation:** after every epoch, the policy's greedy rollout is compared with the greedy cost rule of `solve_milp_f3` on fixed random instances (n = 20, 50, 100). The rule runs inside the same simulator, and `test_rl1.py` checks that it gives the identical plan. `model_best.pt` keeps the best epoch.
- **Training data** (`rl1_generator.py`): for each Solomon class, the CSVs give T, depot, city area, fleet, stations and costs. The customers are new:
  - positions: uniform (R), clustered (C) or mixed (RC);
  - stations: k-means of the customers;
  - energy, range and stranded share: as `make_ev_solomon`;
  - request times: a share at 0, the rest Beta-shaped, fitted to the dataset;
  - θ: the Beta of `f3_params.json`.

  The 116 EV-Solomon files are never seen in training.

### Solving (`solve_rl1.py`)

- **Rollouts:** for each instance (file + first *n* EVs, read from `dataset_ev_solomon/csv`, θ from `income/`), the policy makes `augment` × (1 greedy + `samples` sampled) plans. The default is 8 × 65 = 520. `augment` counts the 8 rotations / reflections of the map: the instance is the same, but the network input differs (POMO).
- **Choice:** the plan with the fewest unplaced EVs, then the lowest Z, is kept.
- **Scoring:** that plan is scored by `results_io.evaluate`. Nothing else changes it: no repair, no heuristic replacement.

## Running on the server

1. **Copy** (from Windows cmd on the laptop). The project folder on the server was emptied on 22 Nov, so create it first, then upload three folders:
   - `code/`, because RL1 uses `results_io.py`, `check_plan_f3.py` and `make_income.py`;
   - `dataset_ev_solomon/` (4 MB; RL1 reads `csv/` and `income/`);
   - this folder.

```
ssh romit@10.72.19.181 "mkdir -p /mnt/DataDrive_2/Yakshith/or_project/project/new_results"
scp -r "E:\AUG-2026\02_CP320_OR_for_Mobility\project\code" "E:\AUG-2026\02_CP320_OR_for_Mobility\project\dataset_ev_solomon" "E:\AUG-2026\02_CP320_OR_for_Mobility\project\RL1_reinforcement_learning_method_1" romit@10.72.19.181:/mnt/DataDrive_2/Yakshith/or_project/project/
scp "E:\AUG-2026\02_CP320_OR_for_Mobility\project\new_results\runs.csv" romit@10.72.19.181:/mnt/DataDrive_2/Yakshith/or_project/project/new_results/
```

The last line is only for the comparison with the MILP. You can skip it and run `compare_rl1.py` on the laptop after copying `results/` back.

2. **Install numpy and torch** once in the conda env (it has neither). Use the CPU build of torch if the server has no GPU:

```
conda activate yakshith
pip install numpy
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

3. **Run.** Use `nohup` so it survives logging out:

```
cd /mnt/DataDrive_2/Yakshith/or_project/project/RL1_reinforcement_learning_method_1/code
nohup bash run_rl1.sh 8 32 > run_rl1.log 2>&1 &
tail -f run_rl1.log
```

`run_rl1.sh HOURS THREADS` runs these steps in order:
- the self-checks (`test_rl1.py --quick`, about a minute);
- training for HOURS of compute (`train_rl1.py --resume`). If it is stopped, the same command continues from `model_last.pt`;
- solving the 56 Solomon files × n = 10..100 (`solve_rl1.py --resume`; with an unchanged model, instances already in `runs.csv` are skipped);
- the comparison with `../../new_results`.

Set `FILE_SIZE="100 200"` to also solve the Homberger files.

4. **Bring the results back** to the laptop (Windows cmd):

```
scp -r romit@10.72.19.181:/mnt/DataDrive_2/Yakshith/or_project/project/RL1_reinforcement_learning_method_1/results romit@10.72.19.181:/mnt/DataDrive_2/Yakshith/or_project/project/RL1_reinforcement_learning_method_1/model "E:\AUG-2026\02_CP320_OR_for_Mobility\project\RL1_reinforcement_learning_method_1\"
```

Single steps:

```
python test_rl1.py                                   # full self-checks (all files, a few minutes)
python train_rl1.py --hours 8 --threads 32           # training only
python solve_rl1.py --only EV-C101 --sizes 10 --details
python solve_rl1.py --file-size 100 200 --resume
python compare_rl1.py --milp ../../new_results
```

## Outputs

`model/` (training):
- `model_best.pt` is used by `solve_rl1.py`.
- `model_last.pt` is for resuming.
- `train_log.csv` has one row per epoch: the validation ratio RL1 / greedy rule (overall and per *n*), the unplaced-EV share of both, the learning rate and the minutes so far.
- `train_meta.json` holds the settings and the total training seconds.

`results/` uses the schema of `code/results_io.py`, the same as `new_results/`:
- `runs.csv` has one row per instance with run_id `rl1` and method `RL1`. It holds:
  - status: `plan`, `invalid_plan` or `error`;
  - `valid`, `violations`, Z1–Z6 and the service numbers;
  - `solve_s` (inference seconds) and `train_s` (training seconds of the model);
  - `message`: rollouts used, and any unplaced EVs.
- `plan.csv` has a row per EV (resource, position, start/end, queue, van wait, money) and per van (stops, km, energy, return time).
- `meta.json` records the model, the training state and the inference settings.
- `logs.zip` holds the console output and any error tracebacks.
- `comparison.csv` and `comparison_summary.md` give RL1 vs MILP vs greedy: gap to the MILP plan and to proven optima, plans where only one method has one, and seconds per instance.

**Invalid plans.** If the policy cannot place some EVs within the limits, they are appended to the resource that finishes them earliest. The plan is then complete, and the checker lists the broken limits (`valid = no`, status `invalid_plan`). An instance with no feasible plan at all (the 51 data-infeasible ones in the MILP run) can only end this way.

## Settings

All settings are in `code/rl1_params.json`, each with an `_about` line, and can be overridden on the command line. The formulation's own data (costs, fleet, T, tariffs, income sensitivities) is never set here; it comes from the CSVs and `code/f3_params.json`.

## Checks made when the code was written (22 Nov 2026, laptop)

- `test_rl1.py` (full) passed:
  - The CSV loader gives exactly the instance of `solve_milp_f3.load_instance` on all 228 tested instances (56 files × n = 10/50/100, plus 60 Homberger files at n = 200).
  - The greedy cost rule run inside the simulator gives the same plan and Z as `greedy_f3` on all 190 instances where greedy succeeds, and also fails on the other 38.
  - The simulator's Z equals `check_plan_f3`'s Z.
  - Plans of the untrained network with no unplaced EV pass the checker.
- Laptop training tests (4 CPU threads, about 30 min in all; nothing from them is in `model/` or `results/`):
  - From random weights, the untrained policy costs about 1.4–1.8× the greedy rule on the validation instances.
  - Pure REINFORCE reached 1.18 after 77 batches.
  - The imitation warm-up reached about 1.002 after 400 batches.
  - REINFORCE at `lr_reinforce` first wobbled up to 1.01, then came back to 1.0006, and below 1 at n = 100 (0.994–0.996) after about 110 batches.
- That 30-minute model solved 8 EV-Solomon instances (C101, C205, R201, RC105 at n = 50 and 100) at about 4–19 s each with 520 rollouts:
  - valid plans wherever the MILP has one;
  - on average 2.9 % above the MILP plan and 0.4 % above greedy, cheaper than greedy on R201 n = 100;
  - invalid plans on the two RC105 instances, which the MILP proves infeasible.

  The server run trains far longer; `train_log.csv` shows whether REINFORCE keeps improving.
- Time on the server: an RL batch at n = 100 took about 13 s on the laptop's 4 threads (about 6 s averaged over n = 10..100). With 32 threads, 8 h gives a few thousand batches. Solving 560 instances at 520 rollouts each takes roughly 1 h on the laptop, less on the server.
