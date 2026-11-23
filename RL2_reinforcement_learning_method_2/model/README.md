# RL2 trained models

This folder is empty until `code/train_rl2.py` runs (on the server: `bash run_rl2.sh`).

| File | Content |
|---|---|
| `model_best.pt` | Network of the epoch with the best validation (lowest mean cost ratio RL2 / greedy rule on 200 fixed random instances). `solve_rl2.py` uses this one. |
| `model_last.pt` | Network and optimiser after the last epoch (only used with an explicit `train_rl2.py --resume`) |
| `train_log.csv` | One row per epoch: phase (imitation / reinforce), batches, imitation loss, training cost spread, unplaced EVs, validation ratio overall and per n, epochs without improvement, learning rate, minutes |
| `train_meta.json` | Settings (rl2_params, f3_params), data templates, device, threads, torch version, parameter count, total training seconds, stop reason |
| `previous_runs/<date-time>/` | Files of earlier runs. Every new training starts at epoch 1 and first moves the previous run's files here. |
