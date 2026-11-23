"""Calibration of the two customer attributes of formulation 4 from published Indian data.

Derives, with every number and assumption written out:
  1. pi_C       share of commercial vehicles among PUBLIC charging requests  (segment distribution)
  2. theta | P  income index of personal EV owners  ~ Beta(a_P, b_P)
  3. theta | C  income index of commercial drivers  ~ Beta(a_C, b_C)
and writes calibration.json, calibration.md and (if matplotlib is installed) calibration_densities.png
next to this script. The recommended values are what common/f4_params.json uses; change an input
below and re-run to see how the recommendation moves.

theta is read as the household's position (percentile) in the URBAN distribution of monthly per-capita
consumption expenditure (MPCE, HCES), because the EV-Solomon instances are a 20 x 20 km city.

Usage (from data/):  python calibration.py
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------------------------- inputs
SOURCES = {
    "DFI": ("Data for India, 'Vehicle ownership in India' (HCES, MPCE quintiles), "
            "https://www.dataforindia.com/vehicle-ownership/"),
    "EVR": ("EVreporter, 'Analysis of India's CY 2024 EV sales' (Vahan registrations), "
            "https://evreporter.com/analysis-of-indias-cy-2024-ev-sales/"),
    "TATA": ("Tata Motors / TATA.ev statement reported in the press: 93 % of its EV owners charge privately "
             "(home / work) and use public chargers mainly for emergencies and long trips"),
    "PAIGAM": ("PAIGAM, IFAT and University of Pennsylvania (2024), 'Prisoners on Wheels': survey of about "
               "5,300 app-based cab drivers in 8 cities; 43 % report net monthly earnings below Rs 15,000"),
    "CIS": ("Centre for Internet and Society, 'Platforming precarity: Mumbai': median monthly income of taxi "
            "workers, adjusted to a 48-hour week, Rs 14,846, https://cis-india.org/raw/mumbai-platforming-precarity"),
    "CSTEP": ("CSTEP (2024), 'EV perceptions and usage patterns: lessons from four Indian cities' "
              "(878 personal and 855 commercial EV users); its per-segment charging figures could not be read "
              "(PDF link returned 404 on 22 Nov 2026), so they are not used"),
}

INPUTS = {
    # Data for India (HCES): share of URBAN households owning a car
    "urban_car_ownership_all": 0.14,          # "around 14 %"            [DFI]
    "urban_car_ownership_top_quintile": 0.30,  # "close to 30 %"          [DFI]
    # Vahan registrations, CY 2024: e-4W registered as cabs (motor cab + luxury cab)
    "commercial_share_of_registrations_sC": 0.10,   # "nearly 90 % personal, the rest cabs"  [EVR]
    # public charging per vehicle-day, commercial relative to personal = km ratio x public-share ratio
    "daily_km_ratio_commercial_over_personal": 5.0,  # ASSUMPTION: ~180 km/day taxi vs ~36 km/day private car
    "public_share_personal": 0.10,                   # [TATA] 93 % charge privately; 0.10 allows office/public use
    "public_share_commercial": 0.50,                 # ASSUMPTION: half of commercial energy at public chargers
    #                                                  (fleets also charge at their own depots)
    # commercial drivers' household position in the urban MPCE distribution
    "commercial_percentile_low": 0.25,     # ASSUMPTION from [PAIGAM], [CIS]: net earnings ~ Rs 15,000/month,
    "commercial_percentile_high": 0.50,    # household of about four -> roughly the 25th-50th urban percentile
    "commercial_percentile_sd": 0.16,      # so that mean +- 1 sd covers about 0.21-0.54
}

RANGES = {   # low / high values of the uncertain inputs, for the pi_C range
    "commercial_share_of_registrations_sC": (0.08, 0.15),
    "daily_km_ratio_commercial_over_personal": (4.0, 6.0),
    "public_share_personal": (0.07, 0.15),
    "public_share_commercial": (0.35, 0.65),
}


# ---------------------------------------------------------------------------------------------- formulas
def pi_commercial(sC, km_ratio, pub_P, pub_C):
    """Share of commercial vehicles among public charging requests. A commercial vehicle makes
    k = km_ratio x pub_C / pub_P times as many public charging requests per day as a personal one, so
    pi_C = sC k / (sC k + (1 - sC))."""
    k = km_ratio * pub_C / pub_P
    return sC * k / (sC * k + 1.0 - sC), k


def personal_beta(own_all, own_top):
    """Car ownership rate rising with the income percentile as o(theta) = o_max theta^gamma. Then
        mean over all households        = o_max / (gamma + 1)
        mean over the top quintile      = o_max (1 - 0.8^(gamma + 1)) / (0.2 (gamma + 1))
    so  top / all = (1 - 0.8^(gamma + 1)) / 0.2  fixes gamma, and the density of car OWNERS over the
    percentile is proportional to theta^gamma, i.e. Beta(gamma + 1, 1)."""
    ratio = own_top / own_all
    a = math.log(1.0 - 0.2 * ratio) / math.log(0.8)          # a = gamma + 1
    o_max = own_all * a
    quint = [o_max * ((q / 5) ** a - ((q - 1) / 5) ** a) / (0.2 * a) for q in range(1, 6)]
    return a, quint


def beta_from_moments(mean, sd):
    s = mean * (1 - mean) / sd ** 2 - 1.0
    return mean * s, (1 - mean) * s


def beta_pdf(x, a, b):
    if x <= 0 or x >= 1:
        return 0.0
    return math.exp((a - 1) * math.log(x) + (b - 1) * math.log(1 - x)
                    + math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b))


def main():
    I = INPUTS
    pi_C, k = pi_commercial(I["commercial_share_of_registrations_sC"], I["daily_km_ratio_commercial_over_personal"],
                            I["public_share_personal"], I["public_share_commercial"])
    lo = pi_commercial(RANGES["commercial_share_of_registrations_sC"][0],
                       RANGES["daily_km_ratio_commercial_over_personal"][0],
                       RANGES["public_share_personal"][1], RANGES["public_share_commercial"][0])[0]
    hi = pi_commercial(RANGES["commercial_share_of_registrations_sC"][1],
                       RANGES["daily_km_ratio_commercial_over_personal"][1],
                       RANGES["public_share_personal"][0], RANGES["public_share_commercial"][1])[0]
    aP, quint = personal_beta(I["urban_car_ownership_all"], I["urban_car_ownership_top_quintile"])
    mC = 0.5 * (I["commercial_percentile_low"] + I["commercial_percentile_high"])
    aC, bC = beta_from_moments(mC, I["commercial_percentile_sd"])
    rec = {"pi_commercial": 0.70, "personal_beta": [2.5, 1.0], "commercial_beta": [3.0, 5.0]}
    mix_mean = (1 - rec["pi_commercial"]) * rec["personal_beta"][0] / sum(rec["personal_beta"]) \
        + rec["pi_commercial"] * rec["commercial_beta"][0] / sum(rec["commercial_beta"])
    out = {"inputs": I, "ranges": RANGES, "sources": SOURCES,
           "derived": {"k_public_requests_ratio": round(k, 2), "pi_commercial_central": round(pi_C, 3),
                       "pi_commercial_low": round(lo, 3), "pi_commercial_high": round(hi, 3),
                       "personal_beta_a": round(aP, 3), "personal_beta_b": 1.0,
                       "implied_urban_car_ownership_by_quintile": [round(q, 4) for q in quint],
                       "personal_theta_mean": round(aP / (aP + 1), 3),
                       "commercial_beta_a": round(aC, 2), "commercial_beta_b": round(bC, 2),
                       "commercial_theta_mean": round(mC, 3)},
           "recommended": {**rec, "mixture_theta_mean": round(mix_mean, 3),
                           "note": "used in common/f4_params.json; pi_C rounded down from the central value, "
                                   "Beta parameters rounded to half units"}}
    with open(os.path.join(HERE, "calibration.json"), "w") as fh:
        json.dump(out, fh, indent=1)

    q = ", ".join(f"Q{j + 1} {100 * v:.1f} %" for j, v in enumerate(quint))
    md = f"""# Calibration of the use segment and the income index (formulation 4)

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
| s_C, cabs among e-4W registrations (CY 2024) | {I['commercial_share_of_registrations_sC']:.2f} | Vahan via EVreporter: nearly 90 % personal, rest cabs |
| daily km, commercial / personal | {I['daily_km_ratio_commercial_over_personal']:.1f} | assumption (about 180 vs 36 km/day) |
| share of energy charged in public, personal | {I['public_share_personal']:.2f} | TATA.ev: 93 % of its EV owners charge privately |
| share of energy charged in public, commercial | {I['public_share_commercial']:.2f} | assumption (fleets also use depots) |

k = {k:.1f}, so **π_C = {pi_C:.3f}** (range {lo:.2f}–{hi:.2f} over the input ranges in the script).
**Recommended: s_i ~ Bernoulli(π_C = 0.70)**, rounded down from the central value. The survey of CSTEP
(878 personal and 855 commercial EV users) is close to 50/50, but that is a sampling design, not a
population share.

## 2. Income index of personal EV owners, θ | P

Data for India (HCES): about 14 % of urban households own a car, close to 30 % in the top urban quintile
(fewer than 5 % in the bottom four rural quintiles). If the ownership rate rises with the percentile as
o(θ) = o_max θ^γ, then (top-quintile rate) / (overall rate) = (1 − 0.8^(γ+1)) / 0.2, which gives
γ + 1 = {aP:.3f}. The car **owners'** percentiles then follow **Beta(γ + 1, 1) = Beta({aP:.2f}, 1)**, mean
{aP / (aP + 1):.2f}. Implied urban ownership by quintile: {q} (only Q5 and the overall rate are published as
text; the fit reproduces both exactly).

**Recommended: θ | P ~ Beta(2.5, 1).** EVs cost more than the average car, so EV owners are probably richer
still: this is the least skewed choice the data allow. Formulation 3 used Beta(2, 2) (mean 0.5) for
everybody, which the data do not support for car owners.

## 3. Income index of commercial drivers, θ | C

App-based cab drivers report low net earnings: 43 % below Rs 15,000 a month (PAIGAM / IFAT 2024, 5,300
drivers in 8 cities); the median for Mumbai taxi workers, adjusted to a 48-hour week, is Rs 14,846 (CIS).
With a household of about four this puts a driver household around the 25th–50th urban percentile
(assumption). Mean {mC:.3f} and standard deviation {I['commercial_percentile_sd']:.2f} give
Beta({aC:.2f}, {bC:.2f}). **Recommended: θ | C ~ Beta(3, 5)** (mean 0.375, sd 0.16). This is the least
supported of the three numbers; it should be the first to replace when a driver income survey is available.

## 4. The resulting population

With these choices the mean income index of all requests is {mix_mean:.3f}, but the distribution is
bimodal: rich personal owners (around 0.7–1) and lower-middle-income commercial drivers (around 0.2–0.5).
A commercial driver's value of time is the vehicle's lost earnings (downtime), not household income;
f4_params.json prices driving, charging and van time of commercial EVs all at that downtime rate.

## Sources

""" + "\n".join(f"- **{k_}**: {v}" for k_, v in SOURCES.items()) + "\n"
    with open(os.path.join(HERE, "calibration.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [i / 400 for i in range(1, 400)]
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        aP_, bP_ = rec["personal_beta"]
        aC_, bC_ = rec["commercial_beta"]
        p = rec["pi_commercial"]
        ax.plot(xs, [beta_pdf(x, aP_, bP_) for x in xs], label=f"personal  Beta({aP_:g},{bP_:g})", color="#2a6fb0")
        ax.plot(xs, [beta_pdf(x, aC_, bC_) for x in xs], label=f"commercial  Beta({aC_:g},{bC_:g})", color="#c4612b")
        ax.plot(xs, [(1 - p) * beta_pdf(x, aP_, bP_) + p * beta_pdf(x, aC_, bC_) for x in xs],
                label=f"all requests (pi_C = {p:g})", color="#333333", lw=2)
        ax.plot(xs, [beta_pdf(x, 2, 2) for x in xs], label="formulation 3: Beta(2,2)", color="#999999", ls="--")
        ax.set_xlabel("income index theta (urban MPCE percentile)")
        ax.set_ylabel("density")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "calibration_densities.png"), dpi=150)
    except Exception as e:                       # the figure is optional
        print(f"figure not written ({type(e).__name__}: {e})")
    print(json.dumps(out["derived"], indent=1))
    print(f"written: calibration.json, calibration.md in {HERE}")


if __name__ == "__main__":
    main()
