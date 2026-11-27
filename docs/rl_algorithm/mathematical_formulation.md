# Mathematical formulation of the implemented RL method (formulation 4)

**What this documents.** The RL method of formulation 4 in `formulation_4/methods/rl/code/`:
- `rl_f4_env.py`
- `rl_f4_model.py`
- `rl_f4_generator.py`
- `train_rl_f4.py`
- `solve_rl_f4.py`
- `rl_f4_params.json`

This code produced every formulation-4 RL result.

**Other RL code in the repository** (not documented here):
- `RL1_reinforcement_learning_method_1/`: the same method for formulation 3, with 16 static EV features instead of 19.
- `RL2_reinforcement_learning_method_2/`: RL1 plus per-instance active search at inference time.

The settings quoted below are those of `rl_f4_params.json`. `model/train_meta.json` records the same values for the run that produced `model_best.pt`.

**Labels used throughout:**
- **[code]**: exactly what the code does.
- **[term]**: standard RL vocabulary, applied only where it fits.
- **[note]**: where the code differs from what the vocabulary would suggest.

---

## 1. Problem instance (input data)

An instance $J$ (built by `f4_data.make_instance`) has the following data.
- **EVs:** $n$ EVs $i = 0,\dots,n-1$. Each has:
  - energy $E_i$, request time $r_i$ and range $R_i$;
  - charging times $\tau^F_i = 60E_i/P^F$ and $\tau^M_i = 60E_i/P^M$;
  - income $\theta_i$ and segment $s_i \in\{0=\text{personal},1=\text{commercial}\}$;
  - per-EV coefficients
    $\mu_i, c^D_i, c^Q_i, c^C_i, c^V_i, W_i$ = straight lines in $\theta_i$ with end points per segment
    (`f4_params.sensitivities`; commercial: $c^D=c^C=c^V=\omega(\theta)$).
- **Stations:** $F$ stations; station $f$ has $g_f$ guns. In total $G=\sum_f g_f$ guns, and gun $k$ belongs to station $f(k)$ (`Batch.gun_station`).
- **Vans:** $M$ identical vans with:
  - battery $B$, seats $Q$, power $P^M$;
  - use $\beta$ kWh/km and transfer efficiency $\eta$.
- **Distances and times:** $d_{if}, t_{if}$ (EV→station) and $d_{ab}, t_{ab}$ between nodes (node 0 = depot, node $i+1$ = EV $i$).
- **Prices:** $c^{EV}, c^M, c^F, \pi^F, \pi^M$ (operator / customer costs) and tariffs $p^F, p^M$.
- **Horizon:** $T$.

**Resources.** There are $K = G + M$ resources, indexed $k=0..G-1$ for the guns and $k=G..G+M-1$ for the vans.

**Training instances** (`rl_f4_generator.Generator.sample`) are random instances of the same form. For each (file size, Solomon class), a template copied from the dataset supplies $T$, the fleet, the costs and the city box. On top of it the generator draws new values:
- **Positions:**
  - uniform (R);
  - Gaussian clusters (C);
  - half and half (RC).
- **Stations:** k-means of the positions.
- **Energy:** $E \in\{10..40\}$, from Solomon demand 1–50 mapped linearly.
- **Range:** $R\sim U[5,40]$ km, with 15 % stranded.
- **Requests:** a share at 0, the rest $T\,u_{\max}\,\mathrm{Beta}(a,b)$.
- **Segment:** $s_i\sim\mathrm{Bernoulli}(\pi_C)$.
- **Income:** $\theta_i\sim\mathrm{Beta}(a_s,b_s)$.
- **Horizon:** $T$ jittered by ±15 %.

The 560 test instances (Solomon files) are never used in training.

## 2. Objective that the method minimises

The cost of a complete plan is formulation 4's $Z = Z_1+\dots+Z_6$, the same terms as `check_plan_f4.validate`:
- $Z_1$: perceived EV driving cost;
- $Z_2$: van km;
- $Z_3$: van dispatch;
- $Z_4$: operator energy;
- $Z_5$: perceived tariff;
- $Z_6$: perceived time.

**Training cost** [code] (`Env.penalised_cost`):
$$
L(\text{plan}) = Z(\text{plan}) + \lambda_{\text{drop}}\;\sigma\;N_{\text{drop}},\qquad \lambda_{\text{drop}}=10 .
$$
Here:
- $N_{\text{drop}}$ is the number of EVs the episode could not place.
- $\sigma$ is the instance's `cost_scale`, the mean over EVs of the cheapest feasible first option, evaluated on the empty state when the `Env` is created and clamped to $\ge 1$. If no EV has a feasible option at the start, the fallback is the mean of $\text{van\_cost0}_i + c^F/Q$.

The method **minimises** $L$. [note] The code is written in terms of a cost, not a reward; "reward" in the docstring means $-\Delta Z$ (Section 5).

## 3. State

The state is not a single vector. [code] It is the `Env` object's mutable tensors together with the instance data.

| Symbol | Code | Meaning |
|---|---|---|
| $\mathcal D \subseteq\{0..n-1\}$ | `done` | EVs already placed or dropped |
| $\mathcal X\subseteq\mathcal D$ | `dropped` | EVs dropped (no feasible resource) |
| $a_k$, $k<G$ | `gun_free[k]` | minute gun $k$ becomes free (0 at start) |
| $p_m$ | `van_pos[m]` | node of van $m$ (0 = depot, $i+1$ = EV $i$) |
| $a_m$ | `van_free[m]` | minute van $m$ becomes free |
| $b_m$ | `van_batt[m]` | battery left (starts at $B$) |
| $\nu_m$ | `van_cnt[m]` | EVs served by van $m$ so far |
| $Z_{\text{acc}}$ | `cost` | accumulated $\sum\Delta Z$ |
| $\mathrm{res}_i, S_i, C_i, o_i$ | `res, S, C, order` | resource, start, completion and step index of each placed EV |

**Initial state** (`Env.__init__`):
- $\mathcal D=\mathcal X=\emptyset$;
- $a_k=0$;
- $p_m=0$, $a_m=0$, $b_m=B$, $\nu_m=0$;
- $Z_{\text{acc}}=0$.

### 3.1 State generator: network inputs computed from the state

- **Static node features** are computed once per episode (`Env.static_features(aug)`):
  - EV $i$, 19 features (`N_EV_FEAT`):
    - depot-relative coordinates scaled to $[-1,1]$, then transformed by one of the 8 dihedral symmetries $\text{aug}$;
    - $r_i/T$, $\tau^F_i/T$, $\tau^M_i/T$, $W_i/T$;
    - $\theta_i$, $E_i/B$;
    - stranded flag, fraction of stations in range;
    - time to the nearest reachable station $/T$, time back to the depot $/T$, $R_i/$city size;
    - cheapest gun cost $/\sigma$, van cost $/\sigma$, van slack $(T-r_i-\tau^M_i-t_{i0})/T$;
    - $s_i$, $\mu_i/2$, $c^Q_i\tau^F_i/\sigma$.
  - Station $f$, 4 features (`N_ST_FEAT`): coordinates (same symmetry), $g_f/2$, share of EVs in range.
  - Depot, 4 features (`N_DEP_FEAT`): $QM/n$, $G/n$, $\ln T/8$, $n/100$.
- **Dynamic features** are recomputed at every decision step:
  - global, 4 features (`global_features`): share of EVs done, mean $a_k/T$, share of vans used, mean battery left $/B$.
  - per EV $\phi_i$, 7 features (`ev_features`):
    - feasible resources $/K$;
    - cheapest $\Delta Z/\sigma$;
    - regret (second cheapest − cheapest)$/\sigma$;
    - earliest finish $/T$;
    - cheapest gun − cheapest van;
    - $(r_i - \min_{j\notin\mathcal D} r_j)/T$;
    - rank among open requests.
  - per resource $\rho_k$, 6 features (`resource_features`):
    - gun: $a_k/T$ and used flag; the van-only slots are 0;
    - van: $a_m/T$, a van flag of 1, $b_m/B$, $\nu_m/Q$, used flag, $t_{p_m,0}/T$.
  - per (chosen EV $i$, resource $k$) $\psi_{ik}$, 6 features (`pair_features`):
    - $\Delta Z_{ik}/\sigma$;
    - $(\Delta Z_{ik}-\min_k\Delta Z_{ik})/\sigma$;
    - wait $/T$;
    - finish $/T$;
    - battery after $/B$;
    - is-van flag.

    Each is zeroed where $(i,k)$ is infeasible.

## 4. Actions, feasibility and timing

**Action** [code]: a pair $(i,k)$, an open EV $i\notin\mathcal D$ and a resource $k$. EV $i$ is **appended to the end** of $k$'s sequence (gun queue or van route).

**Timing is not learned.** The start is the earliest possible (`Env.options`):
$$
\text{gun }k:\; S_{ik} = \max(r_i + t_{i f(k)},\; a_k),\quad w_{ik} = S_{ik} - r_i - t_{if(k)};\qquad
\text{van }m:\; S_{im} = \max(r_i,\; a_m + t_{p_m,i+1}),\quad w_{im} = S_{im}-r_i .
$$
Finish: $C_{ik} = S_{ik}+\tau^F_i$ (gun) or $S_{im}+\tau^M_i$ (van).

**Feasibility mask** $\mathcal F_{ik}$ (`options()["feas"]`):
- **Gun $k$:**
  - $d_{if(k)}\le R_i$;
  - $w_{ik}\le W_i$;
  - $C_{ik}\le T$;
  - not a duplicate gun: if two guns of one station are free at the same minute, only the first is offered (symmetry rule).
- **Van $m$:**
  - $\nu_m<Q$;
  - the van is used, or it is the first unused van (symmetry rule);
  - $E_i/\eta + \beta\,(d_{p_m,i+1}+d_{i+1,0}) \le b_m$;
  - $C_{im} + t_{i+1,0}\le T$.
- **Both:** $i\notin\mathcal D$.

All comparisons use tolerance $10^{-9}$ in float64.

## 5. Per-step cost increment and the termination rule

**Increment of $Z$** (`options()["dz"]`), the [term] negative reward:
$$
\Delta Z_{ik}^{\text{gun}} = \mu_i\big(c^{EV} d_{if} + p^F E_i\big) + \pi^F E_i + c^D_i t_{if} + c^C_i\tau^F_i + c^Q_i w_{ik},
$$
$$
\Delta Z_{im}^{\text{van}} = \mu_i p^M E_i + \pi^M E_i/\eta + c^V_i\tau^M_i + c^V_i w_{im}
 + c^M\big(d_{p_m,i+1} + d_{i+1,0} - d_{p_m,0}\big) + c^F\,\mathbf 1[\nu_m=0].
$$
The van-km term charges the return leg each time and removes the previous return leg, so the increments sum to $Z$. **Episode cost** $Z_{\text{acc}}=\sum_t \Delta Z_{i_tk_t}$.

[note] The increments are summed **without discount** and are used **only as the episode total** in training. There is no per-step return, no reward-to-go and no value function.

**Transition** (`Env.step`). For the chosen $(i,k)$:
- $Z_{\text{acc}}$ += $\Delta Z_{ik}$;
- $\text{res}_i=k$, $S_i$, $C_i$, $o_i=$ step;
- $\mathcal D\leftarrow\mathcal D\cup\{i\}$;
- **gun:** $a_k\leftarrow C_{ik}$;
- **van:** $b_m \leftarrow b_m - E_i/\eta - \beta d_{p_m,i+1}$, $p_m\leftarrow i+1$, $a_m\leftarrow C_{im}$, $\nu_m\leftarrow\nu_m+1$.

**Dropping** (`Env.drop_dead`, before every decision): every open $i$ with $\sum_k\mathcal F_{ik}=0$ is added to $\mathcal D$ and $\mathcal X$. The docstring justifies this: gun queues only grow and batteries only shrink, so such an EV stays infeasible.

**Termination** (`Env.finished`): $\mathcal D=\{0..n-1\}$, i.e. every EV is placed or dropped. The episode therefore has at most $n$ decisions.

## 6. Policy network $\pi_\theta$ (`rl_f4_model.Policy`)

**Settings:** $d=128$, 8 heads, 3 layers, feed-forward width 512, clip $C=10$, MLP hidden width 32. This gives 765 858 parameters (`train_meta.json`).

**Encoder** (once per instance, `encode`):
- per node type a linear embedding: $x_i = W^{ev} f_i + b$, $x_f = W^{st} f_f + b$, $x_0 = W^{dep} f_0 + b$;
- then a 3-layer Transformer encoder (pre-norm, ReLU, no dropout, no positional encoding) and a final LayerNorm;
- outputs: $h_i$ (EVs), $h_f$ (stations), $h_0$ (depot).

**Step 1, choose the EV** (`ev_logits`):
$$
c = W^{ctx}\big[\bar h_{\text{open}};\,h_0;\,\text{global}\big],\quad \bar h_{\text{open}}=\tfrac{1}{|\{i\notin\mathcal D\}|}\sum_{i\notin\mathcal D}h_i,
$$
$$
g = W^{o}\,\mathrm{MHA}\big(Q=W^{q}c,\;K=W^{k}h,\;V=W^{v}h;\ \text{mask } \mathcal E\big),\qquad
u_i = C\tanh\!\Big(\tfrac{(W^{evk}h_i)^\top g}{\sqrt d} + \mathrm{MLP}_{ev}(\phi_i)\Big),
$$
- Masking: $u_i=-\infty$ for $i\notin\mathcal E$, where $\mathcal E=\{i:\exists k\ \mathcal F_{ik}\}$.
- Policy: $\pi_\theta(i\mid s)=\mathrm{softmax}(u)_i$.

**Step 2, choose the resource for that EV** (`res_logits`):
$$
q_i = W^{rq}[h_i;\,g] + b,\qquad
\kappa_k = \begin{cases} W^{rk}h_{f(k)} + W^{gun}\rho_k + b & k<G\\ W^{rk}h_{p_m} + W^{van}\rho_k + b & k=G+m\end{cases},
$$
$$
v_k = C\tanh\!\Big(\tfrac{\kappa_k^\top q_i}{\sqrt d} + \mathrm{MLP}_{pair}(\psi_{ik})\Big),\qquad \pi_\theta(k\mid s,i)=\mathrm{softmax}(v)_k .
$$
- Masking: $v_k=-\infty$ where $\neg\mathcal F_{ik}$.
- The node embedding $h_{p_m}$ is the depot ($p_m=0$) or the EV the van last served.

**Action log-probability:** $\log\pi_\theta(i,k\mid s)=\log\pi_\theta(i\mid s)+\log\pi_\theta(k\mid s,i)$.

**Trajectory log-probability:** $\log\pi_\theta(\tau)=\sum_t\log\pi_\theta(i_t,k_t\mid s_t)$ (`rollout` return value).

**Action selection** (`_choose`):
- sampled: $\sim\mathrm{Categorical}(\mathrm{softmax})$;
- greedy: $\arg\max$;
- teacher-forced: the greedy rule's action, whose log-probability is still computed (imitation).

## 7. Training (`train_rl_f4.py`)

**Two phases.** The phase is chosen by the running batch counter `iters`:

**(a) Imitation warm-up**, for `iters` < 500 (`imitation_step`), with learning rate $10^{-4}$:
- **Batch:** $B=32$ random instances with one $n\sim U\{10..100\}$ and one random symmetry each.
- **Teacher:** `GreedyRule`. It processes EVs in a fixed order: stranded EVs first, then by request time, then by higher $c^Q$. Each EV gets its cheapest feasible resource.
- **Loss** (one rollout per instance):
$$
\mathcal L_{\text{imit}} = -\frac{1}{B}\sum_{b}\frac{1}{n}\log\pi_\theta(\tau^{\text{teacher}}_b).
$$

**(b) REINFORCE with a shared (POMO-style) baseline**, afterwards (`train_step`):
- **Batch:** $B=32$ instances, one $n$, one random symmetry per instance. Each instance is encoded once and rolled out $S=8$ times by **sampling**.
- **Cost:** $L_{bj}$ of rollout $j$ of instance $b$ (Section 2).
- **Baseline and advantage** (normalised by the baseline):
$$
\bar L_b=\frac1S\sum_j L_{bj},\qquad A_{bj}=\frac{L_{bj}-\bar L_b}{\max(|\bar L_b|,10^{-9})},
$$
- **Loss:**
$$
\mathcal L_{\text{RL}}=\frac{1}{BS}\sum_{b,j}A_{bj}\,\log\pi_\theta(\tau_{bj}).
$$
  $A$ is a constant here (no gradient through the cost). Minimising $\mathcal L_{\text{RL}}$ lowers the probability of rollouts costlier than their instance's mean.

**Parameter update** after every batch, in both phases:
- gradient-norm clipping to 1.0;
- an Adam step:
$$
\theta \leftarrow \mathrm{Adam}\big(\theta,\ \mathrm{clip}_{\|\cdot\|\le 1}\nabla_\theta\mathcal L,\ \alpha_e\big),\qquad
\alpha_e = \alpha_{\text{base}}\cdot 0.9998^{\,e},
$$
- $\alpha_{\text{base}}$: $10^{-4}$ while the epoch starts in the warm-up, $2\cdot10^{-5}$ once `iters` ≥ 500.
- $e$ = number of completed epochs. The rate is set at the start of each epoch.

**Epochs** = 50 batches each, followed by validation:
- **Instances:** 72 fixed generated instances, $n\in\{20,50,100\}$ × 24, seed 777.
- **Rollout:** one **greedy** rollout of the policy per instance, with no symmetry augmentation.
- **Measure:** $q = \text{mean}_b\, L^{\pi}_b / L^{\text{rule}}_b$ against the greedy rule on the same instances.
- **Saving:** the epoch with the lowest $q$ is saved as `model_best.pt`.

**Stopping**, whichever comes first:
- 500 epochs;
- early stop: 10 consecutive REINFORCE epochs without a lower $q$, counted only after epoch 200;
- 8 h of training time.

The recorded run stopped early at epoch 210, best at epoch 180 ($q=0.9694$), after 38.6 min (`train_meta.json`).

**Exploration** [code]: only the stochasticity of sampling from $\pi_\theta$ during REINFORCE. There is no $\epsilon$-greedy, no entropy bonus and no temperature.

**Experience storage** [code]: none. Each batch is generated, used for one gradient step and discarded (on-policy). Only the summed $\log\pi_\theta(\tau)$ per rollout and the final cost are kept, inside the autograd graph.

## 8. Evaluation / inference (`solve_rl_f4.py`), separate from training

For each test instance (Solomon file, first $n$ EVs) the policy loaded from `model_best.pt` is run without gradients:
- **Rollouts:** for each of $A=8$ map symmetries, 1 greedy rollout + 64 sampled rollouts, so 520 rollouts in total.
- **Selection:** the rollout with the lowest $N_{\text{drop}}\cdot10^{12} + Z$ is kept, i.e. fewest dropped EVs first, then lowest $Z$.
- **Dropped EVs:** if any, `plan_of` appends each to the resource that finishes it earliest, **ignoring** the violated limits. `check_plan_f4` then flags the plan as `invalid_plan`.
- **Scoring:** the plan is scored by the independent checker, `results_io_f4.evaluate`.

[note] No parameter is updated at inference. RL2's per-instance active search is not part of this method.
