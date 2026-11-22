# Papers behind RL1

Downloaded from arXiv on 22 Nov 2026.

| File | Paper | What RL1 takes from it |
|---|---|---|
| `Kool2019_Attention_learn_to_solve_routing_problems.pdf` | W. Kool, H. van Hoof, M. Welling, *Attention, Learn to Solve Routing Problems!*, ICLR 2019 (arXiv:1803.08475) | **Main paper.** Transformer encoder over the nodes (no positional encoding, so one network handles any number of EVs); decoder that builds the solution one step at a time with a context, a multi-head glimpse and a pointer whose logits are clipped by C·tanh; infeasible choices masked; trained with REINFORCE. |
| `Kwon2020_POMO_policy_optimization_with_multiple_optima.pdf` | Y.-D. Kwon et al., *POMO: Policy Optimization with Multiple Optima for Reinforcement Learning*, NeurIPS 2020 (arXiv:2010.16011) | **Shared baseline**: each instance is rolled out several times and the mean cost of those rollouts is the REINFORCE baseline, so no critic network and no separate greedy-rollout baseline are needed. **Instance augmentation** at test time: the 8 rotations / reflections of the map give different network inputs for the same instance; the best plan is kept. |
| `Li2021_DRL_heterogeneous_capacitated_VRP.pdf` | J. Li et al., *Deep Reinforcement Learning for Solving the Heterogeneous Capacitated Vehicle Routing Problem*, IEEE Trans. Cybernetics 2021 (arXiv:2110.02629) | **Two decisions per step**: a vehicle-selection decoder and a node-selection decoder. RL1 uses the same idea with the order swapped: first the EV, then its resource. The "fleet" is mixed: G guns that stay at their station and M vans that move. |
| `Lin2021_DRL_electric_VRP_with_time_windows.pdf` | B. Lin, B. Ghaddar, J. Nathwani, *Deep Reinforcement Learning for the Electric Vehicle Routing Problem with Time Windows*, IEEE T-ITS (arXiv:2010.02068) | **Closest application.** An attention policy with masks for battery, time windows and charging stations, which shows that the method copes with the hard time and energy limits that formulation 3 also has (queue tolerance W_i, horizon T, van battery B). |

REINFORCE itself is R. J. Williams, *Simple statistical gradient-following algorithms for connectionist reinforcement learning*, Machine Learning 8, 229–256 (1992). It is not on arXiv, so it is cited only.
