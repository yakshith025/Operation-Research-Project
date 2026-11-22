# RL1 trained models

This folder is empty until `code/train_rl1.py` runs (on the server: `bash run_rl1.sh`).

| File | Content |
|---|---|
| `model_best.pt` | Network of the epoch with the best validation (lowest mean cost ratio RL1 / greedy rule on fixed random instances). `solve_rl1.py` uses this one. |
| `model_last.pt` | Network and optimiser after the last epoch, for `train_rl1.py --resume` |
| `train_log.csv` | One row per epoch: phase (imitation / reinforce), batches, imitation loss, training cost spread, unplaced EVs, validation ratio overall and per n, learning rate, minutes |
| `train_meta.json` | Settings (rl1_params, f3_params), data templates, device, threads, torch version, parameter count, total training seconds |
