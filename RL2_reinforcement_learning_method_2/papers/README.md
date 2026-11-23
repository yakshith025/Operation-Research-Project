# Papers behind RL2

RL2 = RL1's method + active search. The papers of the base method (attention model, POMO, heterogeneous-fleet decoder, EV routing with DRL) are in `../../RL1_reinforcement_learning_method_1/papers/`.

| Paper | What RL2 takes from it |
|---|---|
| `Hottung2022_Efficient_active_search_for_combinatorial_optimization.pdf`: A. Hottung, Y.-D. Kwon, K. Tierney, *Efficient Active Search for Combinatorial Optimization Problems*, ICLR 2022. arXiv:2106.05126v3, https://arxiv.org/abs/2106.05126 | **Efficient Active Search, EAS-Emb variant** (`code/rl2_eas.py`). At test time the trained network stays frozen and only the instance's embeddings are tuned on that instance. The loss is REINFORCE with the shared baseline plus λ × the negative log-likelihood of the best solution found so far. The paper shows this closes much of the remaining gap of POMO-type models on routing problems, for a fraction of the cost of retraining the whole network. |
| W. Kool, H. van Hoof, M. Welling, *Attention, Learn to Solve Routing Problems!*, ICLR 2019 | Base model (see RL1). |
| Y.-D. Kwon et al., *POMO*, NeurIPS 2020 | Shared baseline and map-symmetry augmentation (see RL1). |

The EAS PDF was downloaded from arXiv on 23 Nov 2026 (version 3, 15 Mar 2022, the ICLR camera-ready). Section 3.1 (EAS-Emb: embedding updates with the REINFORCE loss of eq. (1) plus the imitation loss) is the part implemented in `code/rl2_eas.py`; section 4.5 is the paper's ablation of the imitation term.
