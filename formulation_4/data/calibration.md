# Calibration of the use segment and the income index (formulation 4)

Written by `calibration.py` (inputs and formulas are at the top of that script). θ is read as the
household's percentile in the **urban** distribution of monthly per-capita consumption (HCES), because an
EV-Solomon instance is a 20 × 20 km city.

## 1. Segment: share of commercial vehicles among public charging requests, π_C

The model's customers are **charging requests at public chargers / vans**, not registered vehicles. A
commercial car (taxi, ride-hailing, fleet) drives several times as far per day and depends much more on
public charging than a private car, so its share among requests is far above its share among registrations:

π_C = s_C k / (s_C k + 1 − s_C),  k = (daily km ratio) × (public share commercial / public share personal)

| input | value | source |
|---|---|---|
| s_C, cabs among e-4W registrations (CY 2024) | 0.10 | Vahan via EVreporter: nearly 90 % personal, rest cabs |
| daily km, commercial / personal | 5.0 | assumption (about 180 vs 36 km/day) |
| share of energy charged in public, personal | 0.10 | TATA.ev: 93 % of its EV owners charge privately |
| share of energy charged in public, commercial | 0.50 | assumption (fleets also use depots) |

k = 25.0, so **π_C = 0.735** (range 0.45–0.91 over the input ranges in the script).
**Recommended: s_i ~ Bernoulli(π_C = 0.70)**, rounded down from the central value. The survey of CSTEP
(878 personal and 855 commercial EV users) is close to 50/50, but that is a sampling design, not a
population share.

## 2. Income index of personal EV owners, θ | P

Data for India (HCES): about 14 % of urban households own a car, close to 30 % in the top urban quintile
(fewer than 5 % in the bottom four rural quintiles). If the ownership rate rises with the percentile as
o(θ) = o_max θ^γ, then (top-quintile rate) / (overall rate) = (1 − 0.8^(γ+1)) / 0.2, which gives
γ + 1 = 2.508. The car **owners'** percentiles then follow **Beta(γ + 1, 1) = Beta(2.51, 1)**, mean
0.71. Implied urban ownership by quintile: Q1 1.2 %, Q2 5.8 %, Q3 12.4 %, Q4 20.6 %, Q5 30.0 % (only Q5 and the overall rate are published as
text; the fit reproduces both exactly).

**Recommended: θ | P ~ Beta(2.5, 1).** EVs cost more than the average car, so EV owners are probably richer
still: this is the least skewed choice the data allow. Formulation 3 used Beta(2, 2) (mean 0.5) for
everybody, which the data do not support for car owners.

## 3. Income index of commercial drivers, θ | C

App-based cab drivers report low net earnings: 43 % below Rs 15,000 a month (PAIGAM / IFAT 2024, 5,300
drivers in 8 cities); the median for Mumbai taxi workers, adjusted to a 48-hour week, is Rs 14,846 (CIS).
With a household of about four this puts a driver household around the 25th–50th urban percentile
(assumption). Mean 0.375 and standard deviation 0.16 give
Beta(3.06, 5.10). **Recommended: θ | C ~ Beta(3, 5)** (mean 0.375, sd 0.16). This is the least
supported of the three numbers; it should be the first to replace when a driver income survey is available.

## 4. The resulting population

With these choices the mean income index of all requests is 0.477, but the distribution is
bimodal: rich personal owners (around 0.7–1) and lower-middle-income commercial drivers (around 0.2–0.5).
A commercial driver's value of time is the vehicle's lost earnings (downtime), not household income;
f4_params.json prices driving, charging and van time of commercial EVs all at that downtime rate.

## Sources

- **DFI**: Data for India, 'Vehicle ownership in India' (HCES, MPCE quintiles), https://www.dataforindia.com/vehicle-ownership/
- **EVR**: EVreporter, 'Analysis of India's CY 2024 EV sales' (Vahan registrations), https://evreporter.com/analysis-of-indias-cy-2024-ev-sales/
- **TATA**: Tata Motors / TATA.ev statement reported in the press: 93 % of its EV owners charge privately (home / work) and use public chargers mainly for emergencies and long trips
- **PAIGAM**: PAIGAM, IFAT and University of Pennsylvania (2024), 'Prisoners on Wheels': survey of about 5,300 app-based cab drivers in 8 cities; 43 % report net monthly earnings below Rs 15,000
- **CIS**: Centre for Internet and Society, 'Platforming precarity: Mumbai': median monthly income of taxi workers, adjusted to a 48-hour week, Rs 14,846, https://cis-india.org/raw/mumbai-platforming-precarity
- **CSTEP**: CSTEP (2024), 'EV perceptions and usage patterns: lessons from four Indian cities' (878 personal and 855 commercial EV users); its per-segment charging figures could not be read (PDF link returned 404 on 22 Nov 2026), so they are not used
