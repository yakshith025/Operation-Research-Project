# Baseline RL of formulation 4

The method of RL1 (`../../RL1_reinforcement_learning_method_1`, the tested one: 502 valid plans vs the MILP's 479 in
formulation 3, +1.07 % Z, 0.12 s per instance), rebuilt for formulation 4. RL1 itself is not changed.

A neural policy builds the plan one decision at a time, "EV *i* goes to resource *k*" (a gun or a van); start times
are the earliest possible (optimal once assignment and order are fixed, because every time cost grows with a later
start). It is trained only on random instances made with the dataset's rules and the segment / income
distributions of formulation 4, then tested on the 56 Solomon files. Every plan is scored by the same checker as
the MILP (`../common/check_plan_f4.py`), so Z means the same for both.

## What changed from RL1

| part | change |
|---|---|
| simulator `rl_f4_env.py` | reads the per-EV coefficients of formulation 4 (segment and income), so masks and the reward (= −increase of Z) are exact for formulation 4. Greedy teacher ordered by c^Q (as the MILP's greedy). |
| network inputs | 3 new static EV features: commercial (0/1), money sensitivity μ, cost of queueing one charging duration c^Q τ^F (relative to the instance's cost scale); 19 instead of 16. The pair features already contain each option's exact increase of Z. |
| generator `rl_f4_generator.py` | draws segment ~ Bernoulli(π_C) and θ ~ Beta of the segment (as `../data/make_f4_dataset.py`); with `--set` it uses the changed parameters and fleet (experiments). |
| data / output | `../data`, `../common/results_io_f4.py` (per-segment columns), `../common/analyze_f4.py` at the end. |
| `--set`, `--variant` | in `train_rl_f4.py` and `solve_rl_f4.py`, for the experiments. |
| settings `rl_f4_params.json` | RL1's values (network, REINFORCE + POMO, imitation warm-up 500 batches, 200–500 epochs, early stop after 10, inference 8 × (1 + 64) rollouts). |

## Run (server, GPU; from `code/`)

```
python test_rl_f4.py --quick      # self-checks (seconds)
python train_rl_f4.py             # new model from epoch 1 -> ../model/ (previous run moved to model/previous_runs/)
python solve_rl_f4.py --resume    # 560 Solomon instances with ../model/model_best.pt -> ../results/ (run id rl_f4)
```
Training stops at 500 epochs, after 10 epochs without a better validation (counted after epoch 200), or at the
hours cap (8 h); with RL1 on the RTX 4090 this took about 35–40 min. The experiments use `../model/model_best.pt`
for their zero-shot runs, so train this model before running the RL experiment folders.

## Self-checks (22 Nov, laptop CPU, all 228 test instances: PASS)

1. every EV's μ, c^a, W equals its segment's line at its θ (20,960 EVs); commercial c^D = c^C = c^V;
2. the greedy rule inside the simulator gives the MILP's greedy plan (185 identical plans; 43 instances where the
   rule fails in both) and the simulator's Z equals the checker's Z;
3. random-policy plans pass the checker; plans with unplaced EVs are flagged; teacher forcing reproduces the rule;
4. generated instances run through simulator and checker; commercial share 0.703 for π_C = 0.7;
5. `--set` overrides reach the instance, the simulator and the generator;
6. one imitation and one REINFORCE step give finite losses and gradients.

A 3-epoch training and solving of 4 instances also ran end to end (outside this folder). `model/` and `results/` are
empty until the server run.

## Code report

`report/RL_code_explained.pdf` (30 pages, 26 Nov 2026): every file and function of this folder explained with the
mathematics (state, action, masks, transition, reward = -increase of Z and why it sums to Z, network, REINFORCE,
solving), a constraint-by-constraint comparison with the MILP, and one real episode traced step by step
(EV-C101, 8 cars: sum of step costs = checker Z = 7251.93; best of 520 tries 7208.84 vs MILP optimum 7207.21).
Code excerpts are read from the source files. Rebuild: `python trace_episode.py` then `latexmk -pdf RL_code_explained.tex`.
