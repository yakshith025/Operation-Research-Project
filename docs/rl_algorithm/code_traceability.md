# Code traceability: Algorithm 1, the system model and the repository

Paths are relative to `formulation_4/methods/rl/code/` unless written in full; `file:line` points to the
line where the statement is implemented. Line numbers are those of the repository as of 27 Nov 2026 (not modified).
Labels: **[code]** what the code does; **[note]** where it differs from usual RL wording; **[inferred]** a reading not
stated literally in the code.

## 1. Which RL implementation this is

| Component | Implementation file | Function/Class | Actual implementation |
|---|---|---|---|
| RL algorithm | `train_rl_f4.py:129-166` | `imitation_step`, `train_step` | Two phases on one network: (1) behaviour cloning of the greedy cost rule (teacher forcing, NLL) for the first 500 batches; (2) REINFORCE (Williams 1992) with a **shared baseline** = mean penalised cost of the 8 rollouts of the same instance (POMO-style). No critic, no value function, no replay, no PPO clipping, no entropy term. |
| Scope | `train_rl_f4.py:1-16`, `rl_f4_params.json` "_about" | module docstrings | "RL1's method" applied to formulation 4. `RL1_reinforcement_learning_method_1` (formulation 3) and `RL2_reinforcement_learning_method_2` (RL1 + active search) are **other** implementations and are not covered. |

## 2. Algorithm 1 (`algorithm_1.tex`), line by line

| Alg. line | Component | Implementation file | Function/Class | Actual implementation |
|---|---|---|---|---|
| Input | hyper-parameters | `rl_f4_params.json` "training", "model"; `train_rl_f4.py:246-260` | `load_rl_params`, `main` | imitation_batches 500, iters_per_epoch 50, batch_instances 32, samples 8, n 10..100, lr 1e-4, lr_reinforce 2e-5, decay 0.9998, max_grad_norm 1, drop_penalty 10, max_epochs 500, min_epochs 200, patience 10, hours 8; CLI can override. |
| 1 | policy and optimiser | `train_rl_f4.py:285-286` | `Policy(**RP["model"])`, `torch.optim.Adam` | d 128, heads 8, layers 3, ff 512, clip 10, dyn_hidden 32; Adam with default betas. |
| 2 | instance generator | `train_rl_f4.py:284`; `rl_f4_generator.py:51-145` | `Generator` | templates per (file size, class) from `data/*.csv`; new random positions, k-means stations, energies, ranges (15 % stranded), requests, segments $\sim$Bernoulli($\pi_C$), $\theta\sim$Beta per segment, $T$ jitter ±15 %. |
| 3-6 | validation set + reference | `train_rl_f4.py:92-106` | `make_validation`, `heuristic_rollout` (`rl_f4_env.py:368`) | 24 instances per n ∈ {20, 50, 100} for file size 100 (seed 777); reference = penalised cost of the greedy rule in the same simulator. |
| 7 | counters | `train_rl_f4.py:288` | `state` dict | epoch, iters, train_s, best_val, best_epoch, stale. |
| 8 | outer loop / stop | `train_rl_f4.py:321-327` | `early_stop`, `while` | stops at hours budget, max_epochs, or `epoch ≥ min_epochs+patience and stale ≥ patience`. |
| 9 | learning rate | `train_rl_f4.py:328-332` | `main` | base = lr_reinforce if a warm-up was configured and iters ≥ imitation_batches, else lr; × 0.9998^epoch; set once at the start of each epoch. |
| 10 | batches per epoch, failure handling | `train_rl_f4.py:335-352` | `main` | 50 batches; an exception skips the batch (logged), 20 in a row abort; inner loop also breaks on the time budget. |
| 11 | sample batch | `train_rl_f4.py:132-136`, `148-152` | `imitation_step`, `train_step` | one n ~ U{10..100} per batch, 32 instances, one random symmetry per instance. |
| 12 | encode once per instance | `train_rl_f4.py:137`, `153-154`; `rl_f4_model.py:66-75` | `Policy.encode`, `Policy.select` | in REINFORCE the embeddings are computed once per instance and repeated for its 8 rollouts (gradients flow through all of them). |
| 13-15 | imitation loss | `train_rl_f4.py:138-139`; `rl_f4_model.py:141,144,151` | `rollout(..., teacher=GreedyRule(env))` | actions forced to the rule's; loss = −mean_b (log π(τ_b)/n). |
| 16-19 | REINFORCE loss | `train_rl_f4.py:155-160` | `train_step` | env = batch repeated 8×; cost = `penalised_cost`; base = mean over the 8; adv = (cost−base)/max(|base|,1e-9) (float64 → float32, no grad); loss = mean(adv·logp). |
| 21-22 | update | `train_rl_f4.py:140-143`, `161-164` | `clip_grad_norm_`, `optim.step` | global L2 norm clipping to 1.0, one Adam step per batch. |
| 24 | validation | `train_rl_f4.py:110-126`, `353` | `validate` | greedy rollout (`greedy=ones`), **no** augmentation; ratio = mean over instances of penalised cost RL / rule. |
| 25-32 | best model, early-stop counter | `train_rl_f4.py:355-369` | `main` | better if ratio < best − min_improvement (0); stale += 1 only for epochs containing REINFORCE batches and epoch > min_epochs; otherwise reset to 0. |
| 33 | checkpoints, log | `train_rl_f4.py:367-381` | `save`, `append_log`, `write_meta` | `model_last.pt` every epoch, `model_best.pt` when better, one CSV row per epoch. |
| 35-52 | `Rollout` | `rl_f4_model.py:123-156` | `rollout` | loop: `options` → `drop_dead` → stop if all done → EV logits → choose i → pair features → resource logits → choose k → add log-probs → `Env.step`. |
| 36 | initial state | `rl_f4_env.py:127-146` | `Env.__init__` | done/dropped false, gun_free 0, van_pos 0 (depot), van_free 0, van_batt B, van_cnt 0, cost 0. |
| 37 | cost scale σ | `rl_f4_env.py:150-157` | `Env.__init__` | mean over EVs of the cheapest feasible first ΔZ on the empty state, fallback mean(van_cost0 + c_F/Q), clamp ≥ 1. |
| 39 | timing, mask, ΔZ | `rl_f4_env.py:160-195` | `Env.options` | see §4 below. |
| 40 | dropping | `rl_f4_env.py:201-207` | `Env.drop_dead` | EVs with no feasible pair → done and dropped. |
| 41 | termination | `rl_f4_env.py:237-238`; `rl_f4_model.py:136-138` | `Env.finished` | all EVs done in every row of the batch. |
| 43 | EV choice | `rl_f4_model.py:89-99`, `139-145` | `Policy.ev_logits`, `_choose` | mask = EV has ≥1 feasible resource. |
| 44 | resource choice | `rl_f4_model.py:101-113`, `146-152` | `Policy.res_logits`, `pair_features` | mask = feasible resources of the chosen EV. |
| 45 | log-probability | `rl_f4_model.py:153-154` | `rollout` | log softmax(u)_i + log softmax(v)_k, summed over active steps. |
| 46-49 | transition | `rl_f4_env.py:209-235` | `Env.step` | cost += ΔZ; record res, S, C, order; gun: gun_free = finish; van: battery −= E/η + β·d(pos,i), pos = i, free = finish, count += 1. |
| 51 | episode cost | `rl_f4_env.py:240-242` | `Env.penalised_cost` | Z_acc + drop_penalty·cost_scale·(#dropped). |
| Alg. 2 | evaluation | `solve_rl_f4.py:80-103`, `111-151`; `rl_f4_env.py:384-439` | `solve_instance`, `run_one`, `plan_of` | 8 symmetries × (1 greedy + 64 sampled), chunks of 1040; key = dropped·1e12 + cost; dropped EVs appended to the earliest-finishing resource; `results_io_f4.evaluate` → `check_plan_f4`. |

## 3. System model (`system_model.tex`), block by block

| Component | Implementation file | Function/Class | Actual implementation |
|---|---|---|---|
| Instance generator | `rl_f4_generator.py:51-145` | `Generator.batch`, `Generator.sample` | random training instances (§2 line 2). |
| Greedy cost rule | `rl_f4_env.py:348-380` | `GreedyRule`, `heuristic_rollout` | order: stranded first, then r_i, then larger c^Q_i (stable sorts); each EV takes argmin ΔZ over feasible resources. Used as teacher (`teacher=`) and as validation reference. |
| Test instances | `formulation_4/common/f4_data.py` | `Dataset.instance` | 56 Solomon files × first n = 10..100 EVs (`solve_rl_f4.collect_tasks`). |
| State $s_t$ | `rl_f4_env.py:126-146` | `Env` attributes | done, dropped, gun_free, van_pos, van_free, van_batt, van_cnt, cost, res, S, C, order. |
| Feasibility & timing | `rl_f4_env.py:160-195` | `Env.options` | earliest-start timing and mask `feas` (§4). |
| Cost engine | `rl_f4_env.py:94-98`, `172`, `188-191` | `Batch.__init__` (`gun_cost0`, `van_cost0`), `Env.options` (`dz`) | ΔZ per pair (§4). Called "reward = minus the increase of Z" in the docstring (`rl_f4_env.py:23`). |
| Drop & termination | `rl_f4_env.py:201-207`, `237-238` | `drop_dead`, `finished` | as Alg. lines 40-41. |
| State transition | `rl_f4_env.py:209-235` | `Env.step` | as Alg. lines 46-49. |
| Static features | `rl_f4_env.py:245-274` | `Env.static_features` | EV 19, station 4, depot 4; `dihedral` symmetry on coordinates (`rl_f4_env.py:114-123`). |
| Dynamic features | `rl_f4_env.py:276-344` | `global_features`, `ev_features`, `resource_features`, `pair_features` | 4 / 7 / 6 / 6 features, recomputed every step from `options`. |
| Encoder | `rl_f4_model.py:46-51`, `66-70` | `Policy.encode` | per-type linear embedding → `nn.TransformerEncoder` (pre-LN, ReLU, dropout 0) → LayerNorm. |
| EV pointer | `rl_f4_model.py:89-99` | `Policy.ev_logits` | context Linear([mean open h; h_dep; global]) → multi-head glimpse over feasible EVs → u = C·tanh(W_evk h·g/√d + MLP(φ)). |
| Resource pointer | `rl_f4_model.py:101-113` | `Policy.res_logits` | q = Linear([h_i; g]); key = res_key(station emb) + gun_dyn(ρ) or res_key(node emb at van_pos) + van_dyn(ρ); v = C·tanh(key·q/√d + MLP(ψ)). |
| Action selection | `rl_f4_model.py:116-120`, `141-152` | `_choose`, `rollout` | multinomial sample / argmax / teacher action. |
| Episode totals | `rl_f4_model.py:131,153-154`; `rl_f4_env.py:240-242` | `rollout` (logp), `penalised_cost` | summed log-probability and penalised cost. |
| Loss | `train_rl_f4.py:139`, `157-160` | `imitation_step`, `train_step` | imitation NLL or REINFORCE with shared baseline. |
| Parameter update | `train_rl_f4.py:140-143`, `161-164`, `328-332` | Adam, `clip_grad_norm_` | every batch; lr schedule per epoch. |
| Validation | `train_rl_f4.py:110-126`, `353-369` | `validate`, `main` | every epoch (= 50 batches); best → `model_best.pt`; early stop. |
| Inference | `solve_rl_f4.py:80-103` | `solve_instance` | best of 520 rollouts; `torch.no_grad`. |
| Output | `rl_f4_env.py:384-439`; `solve_rl_f4.py:125-134`; `common/results_io_f4.py`, `common/check_plan_f4.py` | `plan_of`, `rio.evaluate`, `validate` | plan dict → independent checker → `runs.csv`, `plan.csv`; status `invalid_plan` if violations. |

## 4. Feasibility, timing and cost (the equations of `mathematical_formulation.md` §4-5)

| Equation | Implementation file | Function/Class | Actual implementation |
|---|---|---|---|
| gun start, wait, finish | `rl_f4_env.py:93,166-168` | `Batch`, `options` | `arrive_g = r + tEF[:, :, gun_station]`; `start_g = max(arrive_g, gun_free)`; `wait_g = start_g − arrive_g`; `fin_g = start_g + tauF`. |
| gun mask | `rl_f4_env.py:92,169-171` | `options` | `reach_g` (d ≤ R) & wait ≤ W + 1e-9 & fin ≤ T + 1e-9 & not (same station, earlier gun with equal free time, within 1e-12). |
| gun ΔZ | `rl_f4_env.py:94-96,172` | `Batch`, `options` | μ(c_EV d + p_F E) + π_F E + c^D t + c^C τ^F + c^Q wait. |
| van start, wait, finish | `rl_f4_env.py:174-181` | `options` | `start_v = max(r, van_free + t(pos, i))`, `wait_v = start_v − r`, `fin_v = start_v + tauM`. |
| van mask | `rl_f4_env.py:182-187` | `options` | count < Q & (used or first unused) & E/η + β(d(pos,i)+d(i,0)) ≤ battery + 1e-9 & fin_v + t(i,0) ≤ T + 1e-9. |
| van ΔZ | `rl_f4_env.py:97-98,188-191` | `Batch`, `options` | μ p_M E + π_M E/η + c^V τ^M + c_M(d(pos,i)+d(i,0)−d(pos,0)) + c_F·[unused] + c^V wait. |
| per-EV coefficients | `formulation_4/common/f4_params.py:144-172` | `line`, `sensitivities` | μ, W, c^D, c^Q, c^C, c^V straight lines in θ per segment; c^a ≥ 0, W ≥ 0. |
| sum of ΔZ = Z | `rl_f4_env.py:23-27`; `test_rl_f4.py` check 2-3 | docstring, self-test | stated in code and tested against `check_plan_f4` (the test was not re-run here: torch is not installed on this machine). |

## 5. Consistency between the pseudocode and the diagram (Part 4)

| Aspect | Algorithm 1 | System model | Same implementation? |
|---|---|---|---|
| State representation | line 36 (initial), 46-49 (update) | "State $s_t$", "State transition" | yes: `Env` tensors, `Env.step`. |
| Network inputs | lines 12, 43-44 (`StaticFeatures`, φ, ρ, ψ, global) | "Static features", "Dynamic features" | yes: `Env.*_features`. |
| Action selection | lines 43-45; mode note under Alg. 1 | "EV pointer", "Resource pointer", "Action selection" | yes: two-step (EV, then resource), masked softmax, sample / argmax / teacher. |
| Reward / cost | lines 39, 46, 51 | "Cost engine" (ΔZ), "Episode totals" (L) | yes: ΔZ per step, summed, + drop penalty at the end; used only as the episode total. |
| Update mechanism | lines 13-22 | "Loss", "Parameter update" | yes: imitation NLL then REINFORCE with shared baseline; clip 1.0; Adam every batch. |
| Termination of an episode | lines 40-41 | "Drop & termination" | yes: all EVs placed or dropped. |
| Termination of training | lines 8, 25-32 | "Validation" (early stop) | yes: max epochs, time budget, early stop after epoch 200 + 10. |
| Training vs evaluation | Alg. 1 vs Alg. 2 | "Between episodes" frame vs "Evaluation" frame | yes: evaluation uses `model_best.pt`, no updates, best of 520 rollouts. |

## 6. What could not be verified, ambiguities and assumptions

1. **The two reference images were not available.** The request mentions an attached "Algorithm 1 – RL Training" image and
   a "System Model" image; neither reached this session. Layout and level of detail follow the request text instead.
2. **Which RL code.** The repository holds three RL implementations. These documents cover the formulation-4 code (the
   one behind every formulation-4 result). RL1 is the same algorithm for formulation 3 (16 instead of 19 static EV
   features); RL2 adds per-instance active search at inference, which is **not** in this algorithm.
3. **"Reward".** The code minimises a cost; the per-step increment ΔZ is called the reward (negated) only in the
   docstring. It is never used per step in learning: no discounting, no reward-to-go, no bootstrapping.
4. **Validation ratio uses penalised costs** of both RL and the rule, averaged per instance (not a ratio of sums).
5. **Batching details not drawn.** Rows of a batch that have finished take a dummy action (EV 0, resource 0) that is
   neither counted in log π nor applied (`rl_f4_model.py:140,145,148,152-154`; `Env.step` uses `active`). Omitted from
   the pseudocode as an implementation detail of batching.
6. **Symmetry pruning** (only the first of equally free guns of a station; only the first unused van) changes the
   action set but, per the docstring, not the set of reachable plans; this claim was not independently proven here.
7. **No runtime check.** torch is not installed on this machine, so nothing was executed; all statements come from
   reading the code and the recorded outputs (`model/train_meta.json`, `model/train_log.csv`): stop reason "early
   stop … best at epoch 180", 210 epochs, 10 500 batches, 2317 s, 765 858 parameters, lr 1.918e-05 at epoch 209
   (= 2e-5 · 0.9998^209, consistent with line 9).
8. **Imitation phase lr.** With `imitation_batches = 500` and 50 batches per epoch, the switch to `lr_reinforce` falls
   exactly on an epoch boundary (epoch 11). If those values changed so that the switch fell mid-epoch, the code would
   keep the warm-up lr until the epoch ends (the lr is set once per epoch).
