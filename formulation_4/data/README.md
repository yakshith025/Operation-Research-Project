# Data of formulation 4

The complete dataset used by every formulation-4 method and experiment. All code reads only this folder.

| file | content |
|---|---|
| `solomon_config.csv` | one row per Solomon file (56): T, speed, km per grid unit, detour, costs, vans, battery, seats, stations, guns |
| `solomon_data.csv` | depot, 12 stations and 100 EVs per file: `instance, type, id, x, y, energy_kWh, request_min, range_km, guns, power_kW, segment, theta` |
| `gehring_homberger_config.csv`, `gehring_homberger_data.csv` | the same for the 60 Gehring–Homberger files (200 EVs, 24 stations, 12 vans) |
| `dataset_settings.json` | the distributions and seed the attributes were drawn with, fingerprint of the source |
| `summary.csv` | per file: commercial share (all EVs, first 10, first 50), mean theta per segment |
| `calibration.py` → `calibration.md`, `calibration.json`, `calibration_densities.png` | where π_C and the two Beta distributions come from (inputs, sources, formulas) |
| `make_f4_dataset.py` | builds the four CSV files |

Every column except `segment` and `theta` is the EV-Solomon dataset of formulation 3
(`../../dataset/csv`), unchanged: same positions, energies, request times, ranges, fleet and costs, so
results are comparable with formulation 3. `segment` (personal / commercial) and `theta` (income index in [0, 1])
are on EV rows only; depot and station rows leave them blank.

## How the attributes are drawn

- `s_i ~ Bernoulli(π_C)`, commercial with probability π_C = 0.70 (`../common/f4_params.json`, `segment_distribution`)
- `θ_i | personal ~ Beta(2.5, 1)`, `θ_i | commercial ~ Beta(3, 5)`
- one seed per Solomon class and file size, three independent streams (segment, θ if personal, θ if commercial):
  files of one class share positions, energies, ranges **and** attributes (they differ only in request times, as
  in formulation 3), and changing π_C only re-labels EVs while the θ draws stay the same.

Over all 116 files 69 % of the EVs are commercial (Solomon 70.6 %, Gehring–Homberger 68.8 %); among the first 10
EVs of a Solomon file the share ranges from 30 % to 90 %.

## Commands (from this folder)

```
python calibration.py                # recompute the recommendation (calibration.md)
python make_f4_dataset.py --check    # verify: network columns = source, attributes = draws of f4_params.json
python make_f4_dataset.py --force    # rebuild after changing pi_commercial or a Beta in f4_params.json
```
`--force` changes the data of every later run; results made with the old data then refer to another dataset.
