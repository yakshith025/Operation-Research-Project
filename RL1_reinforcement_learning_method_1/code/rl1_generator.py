"""Random training instances for RL1, made with the rules of the EV-Solomon dataset.

The policy is trained on new random instances only; the 116 EV-Solomon files are never seen in
training and are used for testing (solve_rl1.py). Per (file size, Solomon class) a template is
read from the CSV dataset: planning period T, depot, city area, fleet and cost settings. For each
training instance the customers are new:

  positions   class R: uniform in the class's city area; class C: Gaussian clusters, customers listed
              cluster by cluster (as in Solomon, so the first n EVs are spatially concentrated);
              class RC: half clustered, half uniform
  stations    k-means of all customer positions (make_ev_solomon.build rule), guns and power as in
              the template
  energy      Solomon DEMAND 1..50 mapped linearly onto [E_MIN, E_MAX] (make_ev_solomon)
  range       uniform [RANGE_MIN, RANGE_MAX]; a share STRANDED_SHARE gets 20-80 % of the distance to
              its nearest station (make_ev_solomon)
  request     a share of requests at 0, the rest T * u_max * Beta(a, b), whole minutes
              (shape fitted to the request times of the dataset; settings in rl1_params.json)
  theta       Beta(a, b) of f3_params.json (as make_income)
  T           the class's T times a random factor in [1 - T_jitter, 1 + T_jitter]

Like in the dataset, an instance is the first n EVs of a full file of 100 (or 200) customers.
"""
import numpy as np

from rl1_data import config_scalars, make_instance
import make_ev_solomon as mes


class Generator:
    def __init__(self, dataset, gen_params, P3, seed=0, file_sizes=(100,)):
        self.P3 = P3
        self.g = gen_params
        self.rng = np.random.default_rng(seed)
        self.templates = {}
        for name in dataset.files():
            size = dataset.n_evs(name)
            if size not in file_sizes:
                continue
            key = (size, dataset.family(name))
            raw = dataset.raw(name)
            box_lo = np.minimum(raw["ev_xy"].min(0), raw["fcs"].min(0))
            box_hi = np.maximum(raw["ev_xy"].max(0), raw["fcs"].max(0))
            if key not in self.templates:
                self.templates[key] = {"cfg": config_scalars(dataset.config[name]), "depot": raw["depot"],
                                       "n_fcs": len(raw["fcs"]), "guns": raw["guns"],
                                       "station_power": raw["station_power"], "N": size,
                                       "lo": box_lo, "hi": box_hi}
            else:
                t = self.templates[key]
                t["lo"], t["hi"] = np.minimum(t["lo"], box_lo), np.maximum(t["hi"], box_hi)
        if not self.templates:
            raise ValueError(f"no CSV files with {file_sizes} EVs found for the training templates")
        self.keys = sorted(self.templates)
        self.count = 0

    # ------------------------------------------------------------------ pieces
    def _uniform(self, k, lo, hi):
        return self.rng.uniform(lo, hi, size=(k, 2))

    def _clustered(self, k, lo, hi):
        g, rng = self.g, self.rng
        n_cl = int(rng.integers(g["clusters"][0], g["clusters"][1] + 1))
        width = float(np.max(hi - lo))
        margin = 0.05 * width
        centers = rng.uniform(lo + margin, hi - margin, size=(n_cl, 2))
        sigma = rng.uniform(*g["cluster_sigma_share"], size=n_cl) * width
        sizes = rng.multinomial(k, np.ones(n_cl) / n_cl)
        pts = [centers[c] + rng.normal(0.0, sigma[c], size=(sizes[c], 2)) for c in range(n_cl)]
        return np.clip(np.vstack(pts), lo, hi)                # cluster by cluster, as in Solomon C

    def _requests(self, N, T):
        g, rng = self.g, self.rng
        zero = rng.uniform(*g["request_zero_share"])
        u_max = rng.uniform(*g["request_max_share"])
        r = T * u_max * rng.beta(g["request_beta"][0], g["request_beta"][1], size=N)
        r[rng.random(N) < zero] = 0.0
        return np.round(r)

    # ------------------------------------------------------------------ one instance
    def sample(self, n, key=None):
        """One random instance with n EVs (the first n of a full random file)."""
        rng = self.rng
        key = key if key is not None else self.keys[rng.integers(len(self.keys))]
        tpl = self.templates[key]
        N, family = tpl["N"], key[1]
        n = min(int(n), N)
        lo, hi = tpl["lo"], tpl["hi"]
        if family.startswith("RC"):
            half = N // 2
            xy = np.vstack([self._clustered(half, lo, hi), self._uniform(N - half, lo, hi)])
            xy = xy[rng.permutation(N)]
        elif family.startswith("C"):
            xy = self._clustered(N, lo, hi)
        else:
            xy = self._uniform(N, lo, hi)
        xy = np.round(xy, 2)
        fcs = np.round(mes.kmeans(xy, tpl["n_fcs"], rng, iters=self.g.get("kmeans_iters", 100)), 2)
        cfg = dict(tpl["cfg"])
        cfg["T"] = float(round(tpl["cfg"]["T"] * rng.uniform(1 - self.g["T_jitter"], 1 + self.g["T_jitter"])))
        demand = rng.integers(1, 51, size=N)
        E = np.round(mes.E_MIN + (mes.E_MAX - mes.E_MIN) * (demand - 1) / 49.0, 2)
        k = cfg["detour"] * cfg["km_per_unit"]
        d_near = k * np.linalg.norm(xy[:, None] - fcs[None], axis=2).min(1)
        R = rng.uniform(mes.RANGE_MIN, mes.RANGE_MAX, N)
        stranded = rng.choice(N, size=int(round(mes.STRANDED_SHARE * N)), replace=False)
        R[stranded] = rng.uniform(0.2, 0.8, len(stranded)) * d_near[stranded]
        R = np.round(R, 2)
        r = self._requests(N, cfg["T"])
        d = self.P3["theta_distribution"]
        theta = np.round(rng.beta(d["a"], d["b"], size=N), 4)
        self.count += 1
        return make_instance(f"GEN-{family}-{self.count}", n, "generated", family, cfg, tpl["depot"], fcs,
                             tpl["guns"], tpl["station_power"], xy[:n], E[:n], r[:n], R[:n], theta[:n],
                             self.P3, analysis=False, as_lists=False)

    def batch(self, n, size, file_size=None):
        """size instances with n EVs, all of one file size (same number of stations, guns and vans,
        so they stack into one tensor batch); each instance gets a random class."""
        sizes = sorted({k[0] for k in self.keys})
        file_size = file_size if file_size is not None else sizes[self.rng.integers(len(sizes))]
        keys = [k for k in self.keys if k[0] == file_size]
        return [self.sample(n, keys[self.rng.integers(len(keys))]) for _ in range(size)]
