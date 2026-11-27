"""Income-index densities assumed in formulation 4 (data/calibration.md): personal Beta(2.5, 1), commercial
Beta(3, 5), mixed with pi_C = 0.70. Writes figures/beta_income.png (300 dpi, transparent, deck accents).
Usage: python make_beta_figure.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ACCP, ACCA, DARK = "#5F4B6B", "#C58A63", "#333333"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": DARK, "axes.labelcolor": DARK, "xtick.color": DARK, "ytick.color": DARK,
                     "legend.frameon": False, "savefig.transparent": True, "savefig.dpi": 300})


def beta_pdf(x, a, b):
    return x ** (a - 1) * (1 - x) ** (b - 1) * math.gamma(a + b) / (math.gamma(a) * math.gamma(b))


x = np.linspace(0.001, 0.999, 400)
pP, pC = beta_pdf(x, 2.5, 1.0), beta_pdf(x, 3.0, 5.0)
mix = 0.30 * pP + 0.70 * pC
fig, ax = plt.subplots(figsize=(3.6, 2.6))
ax.fill_between(x, pP, color=ACCP, alpha=0.25)
ax.plot(x, pP, color=ACCP, lw=1.6, label="personal: Beta(2.5, 1), mean 0.71")
ax.fill_between(x, pC, color=ACCA, alpha=0.25)
ax.plot(x, pC, color=ACCA, lw=1.6, label="commercial: Beta(3, 5), mean 0.375")
ax.plot(x, mix, color=DARK, lw=1.0, ls="--", label=r"all requests ($\pi_C=0.70$), mean 0.48")
ax.set_xlim(0, 1)
ax.set_ylim(0, 3.0)
ax.set_xlabel(r"income index $\theta$ (urban percentile)")
ax.set_ylabel("density")
ax.legend(fontsize=6.5, loc="upper left")
fig.tight_layout()
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
fig.savefig(os.path.join(HERE, "figures", "beta_income.png"))
print("mean of mixture", round(float(np.trapezoid(x * mix, x)), 3))
