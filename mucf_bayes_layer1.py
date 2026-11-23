"""
mucf_bayes_layer1.py — Bayesian inference with the layer-1 physics model embedded, and prior sensitivity tables

Structure:
  1. Embed the layer-1 closed-form solution (mucf_layer1_closed.layer1_forward) in a PyMC likelihood
     and infer the rate constants jointly from experimental data (mucf_observations.csv)
       inferred: λ_dt, κ (q1s), coefficients b0..b3 of λ_dtμ(T, φ), ω_s0, coefficients r0, r1 of the
                 reactivation R(φ), model error σ_model, and T / φ / c_t where experimental conditions are unknown
  2. Posterior prediction of Xc at design conditions (T, φ, c_t, c_He) → layer-3 Q_eng → P(Q_eng > 1)
  3. Automatic prior sensitivity table (prior_sensitivity_table)
       Scenarios perturbing each prior to "mean ±1σ", "width ×2", "width ×0.5" are generated automatically,
       and the posterior is reweighted with Pareto-smoothed importance sampling (PSIS).
       If k̂ > 0.7 (unstable weights), the scenario is refit by MCMC instead
  4. --synthetic-check: fit synthetic data generated from known true values and check that they are recovered

Usage:
  python mucf_bayes_layer1.py                         # real data only
  python mucf_bayes_layer1.py --synthetic-check       # pipeline validation
  python mucf_bayes_layer1.py --sensitivity           # prior sensitivity table (CSV output)
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy import stats

from mucf_layer1_closed import NP, layer1_forward, lam_dtmu_model, reactivation_model
from mucf_bayes import q_eng, breakeven_xc, sample_plant_priors

PT = SimpleNamespace(exp=pt.exp, log=pt.log, sigmoid=pt.sigmoid)

# ---------------------------------------------------------------------------
# Priors (expressed as (kind, params...) tuples, used by both PyMC and scipy)
# ---------------------------------------------------------------------------

BASE_PRIORS: dict[str, tuple] = {
    # transfer rate λ_dt [1/s]: centered on the layer-1 guess 2.8e8, width of a factor e^0.5 ≈ 1.6
    "log_lam_dt": ("normal", math.log(2.8e8), 0.5),
    # κ in q1s = 1/(1+κ c_t)
    "log_kappa": ("normal", math.log(3.5), 0.5),
    # log λ_dtμ = b0 + b1 x + b2 x² + b3 (φ-1),  x = log(T/100K)
    "b0": ("normal", math.log(6.0e8), 1.0),
    "b1": ("normal", 0.0, 1.0),
    "b2": ("normal", 0.0, 0.5),
    "b3": ("normal", 0.0, 0.5),
    # initial sticking [%]: covers theory 0.857 (Kamimura+ 2023) to 0.93 (earlier literature values)
    "omega_s0": ("normal", 0.89, 0.04),
    # reactivation R(φ) = sigmoid(r0 + r1 (φ-1)): R(1) ≈ 0.35 (value adopted by Kamimura+ 2023)
    "r0": ("normal", math.log(0.35 / 0.65), 0.5),
    "r1": ("normal", 0.0, 1.0),
    # relative model error of Xc and λc (imperfection of the phenomenological parameterization)
    "sigma_model": ("halfnormal", 0.05),
}
GLOBAL_PARAMS = list(BASE_PRIORS)


def make_rv(name: str, spec: tuple):
    kind = spec[0]
    if kind == "normal":
        return pm.Normal(name, mu=spec[1], sigma=spec[2])
    if kind == "halfnormal":
        return pm.HalfNormal(name, sigma=spec[1])
    if kind == "lognormal":
        return pm.LogNormal(name, mu=spec[1], sigma=spec[2])
    if kind == "uniform":
        return pm.Uniform(name, lower=spec[1], upper=spec[2])
    raise ValueError(kind)


def log_prior(spec: tuple, x: np.ndarray) -> np.ndarray:
    kind = spec[0]
    if kind == "normal":
        return stats.norm.logpdf(x, spec[1], spec[2])
    if kind == "halfnormal":
        return stats.halfnorm.logpdf(x, scale=spec[1])
    if kind == "lognormal":
        return stats.lognorm.logpdf(x, s=spec[2], scale=np.exp(spec[1]))
    if kind == "uniform":
        return stats.uniform.logpdf(x, spec[1], spec[2] - spec[1])
    raise ValueError(kind)


def sample_prior(spec: tuple, n: int, rng) -> np.ndarray:
    kind = spec[0]
    if kind == "normal":
        return rng.normal(spec[1], spec[2], n)
    if kind == "halfnormal":
        return np.abs(rng.normal(0.0, spec[1], n))
    if kind == "lognormal":
        return rng.lognormal(spec[1], spec[2], n)
    if kind == "uniform":
        return rng.uniform(spec[1], spec[2], n)
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

OBSERVABLES = {"Xc", "lam_c_over_phi", "omega_s_pct"}
NEEDS = {"Xc": ("T", "phi", "c_t"), "lam_c_over_phi": ("T", "phi", "c_t"), "omega_s_pct": ("phi",)}
COND_DEFAULT = {"T": 100.0, "phi": 1.0, "c_t": 0.4}


def load_observations(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    df = df[df["use"] == 1].reset_index(drop=True)
    bad = set(df["observable"]) - OBSERVABLES
    if bad:
        raise ValueError(f"unknown observable: {bad}")
    if "c_he" not in df:
        df["c_he"] = 0.0
    df["c_he"] = df["c_he"].fillna(0.0)
    for c in ["T", "phi", "c_t"]:
        col = "T_K" if c == "T" else c
        for suffix in ["_lo", "_hi"]:
            if f"{c}{suffix}" not in df:
                df[f"{c}{suffix}"] = np.nan
        if col not in df:
            df[col] = np.nan
    # error if a required condition has neither a value nor a range
    for i, row in df.iterrows():
        for c in NEEDS[row["observable"]]:
            col = "T_K" if c == "T" else c
            if np.isnan(row[col]) and (np.isnan(row[f"{c}_lo"]) or np.isnan(row[f"{c}_hi"])):
                raise ValueError(f"{row['source']}: a value or range for {c} is required")
    return df


def condition_tensor(name: str, df: pd.DataFrame):
    """Turn a condition column into a pytensor tensor; missing entries with a range become uniform latent variables."""
    col = "T_K" if name == "T" else name
    vals = df[col].to_numpy(float)
    lo, hi = df[f"{name}_lo"].to_numpy(float), df[f"{name}_hi"].to_numpy(float)
    miss = np.isnan(vals)
    latent = miss & ~np.isnan(lo) & ~np.isnan(hi)
    filled = np.where(miss, COND_DEFAULT[name], vals)
    filled = np.where(latent, 0.5 * (lo + hi), filled)
    t = pt.as_tensor_variable(filled)
    if latent.any():
        idx = np.flatnonzero(latent)
        z = pm.Uniform(f"{name}_latent", lower=lo[idx], upper=hi[idx], shape=len(idx))
        t = pt.set_subtensor(t[idx], z)
    return t


# ---------------------------------------------------------------------------
# PyMC model
# ---------------------------------------------------------------------------

def build_model(df: pd.DataFrame, priors: dict[str, tuple] = BASE_PRIORS) -> pm.Model:
    obs = df["observable"].to_numpy()
    val = df["value"].to_numpy(float)
    sig = df["sigma"].to_numpy(float)

    with pm.Model() as m:
        theta = {k: make_rv(k, priors[k]) for k in GLOBAL_PARAMS}
        cond = {
            "T": condition_tensor("T", df),
            "phi": condition_tensor("phi", df),
            "c_t": condition_tensor("c_t", df),
            "c_he": pt.as_tensor_variable(df["c_he"].to_numpy(float)),
        }
        out = layer1_forward(theta, cond, PT)
        sm = theta["sigma_model"]

        i = np.flatnonzero(obs == "Xc")
        if i.size:
            mu = out["Xc"][i]
            pm.Normal("Xc_obs", mu=mu, sigma=pt.sqrt(sig[i] ** 2 + (sm * mu) ** 2), observed=val[i])

        i = np.flatnonzero(obs == "lam_c_over_phi")
        if i.size:  # scale by 1e8 for numerical convenience
            mu = out["lam_c"][i] / cond["phi"][i] / 1e8
            pm.Normal("lamc_obs", mu=mu,
                      sigma=pt.sqrt((sig[i] / 1e8) ** 2 + (sm * mu) ** 2), observed=val[i] / 1e8)

        i = np.flatnonzero(obs == "omega_s_pct")
        if i.size:
            pm.Normal("omega_obs", mu=out["omega_eff_pct"][i], sigma=sig[i], observed=val[i])
    return m


def fit(df, priors=BASE_PRIORS, draws=1000, tune=1500, chains=4, seed=1):
    with build_model(df, priors):
        return pm.sample(draws, tune=tune, chains=chains, target_accept=0.95,
                         random_seed=seed, progressbar=False)


def posterior_draws(idata) -> dict[str, np.ndarray]:
    return {k: idata.posterior[k].values.reshape(-1) for k in GLOBAL_PARAMS}


# ---------------------------------------------------------------------------
# Prediction at design conditions and Q_eng
# ---------------------------------------------------------------------------

@dataclass
class Design:
    T: float = 25.0
    phi: float = 1.2
    c_t: float = 0.4
    c_he: float = 0.0

    def cond(self):
        return {"T": self.T, "phi": self.phi, "c_t": self.c_t, "c_he": self.c_he}


def predict_xc(draws: dict, design: Design, rng, include_model_error=True) -> np.ndarray:
    xc = layer1_forward(draws, design.cond(), NP)["Xc"]
    if include_model_error:
        xc = xc * (1.0 + draws["sigma_model"] * rng.standard_normal(xc.shape))
    return xc


@dataclass
class PlantPrior:
    name: str = "base (E_μ 5 GeV, η_acc 0.18)"
    emu_median: float = 5.0
    emu_sigma: float = 0.15
    eta_acc_mean: float = 0.18
    eta_acc_sd: float = 0.04

    def sample(self, n, seed=7):
        return sample_plant_priors(n, np.random.default_rng(seed), self.emu_median,
                                   self.emu_sigma, self.eta_acc_mean, self.eta_acc_sd)


# ---------------------------------------------------------------------------
# PSIS (Vehtari, Simpson, Gelman, Yao, Gabry 2024, JMLR 25(72))
# ---------------------------------------------------------------------------

def psis(log_w: np.ndarray) -> tuple[np.ndarray, float]:
    """Return Pareto-smoothed normalized weights and the shape parameter k̂."""
    lw = log_w - np.max(log_w)
    S = lw.size
    M = int(np.ceil(min(0.2 * S, 3.0 * np.sqrt(S))))
    order = np.argsort(lw)
    tail_idx = order[-M:]
    cutoff = lw[order[-M - 1]]
    w_tail = np.exp(lw[tail_idx])
    u = np.exp(cutoff)
    exceed = w_tail - u
    k = 0.0
    if np.all(exceed > 0) and np.ptp(exceed) > 0:
        k, _, scale = stats.genpareto.fit(exceed, floc=0.0)
        q = (np.arange(1, M + 1) - 0.5) / M
        smoothed = u + stats.genpareto.ppf(q, k, loc=0.0, scale=scale)
        smoothed = np.minimum(smoothed, 1.0)  # truncate at the original maximum weight (= 1)
        lw = lw.copy()
        lw[tail_idx] = np.log(smoothed)  # tail_idx is sorted in ascending order
    w = np.exp(lw - np.max(lw))
    return w / w.sum(), float(k)


def weighted_quantile(x, w, qs):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    cw /= cw[-1]
    return np.interp(qs, cw, x[o])


# ---------------------------------------------------------------------------
# Prior sensitivity table
# ---------------------------------------------------------------------------

def auto_prior_scenarios(priors: dict[str, tuple] = BASE_PRIORS,
                         params: list[str] | None = None) -> list[dict]:
    """Automatically generate scenarios that perturb one prior at a time."""
    out = []
    for name in params or list(priors):
        spec = priors[name]
        kind = spec[0]
        if kind in ("normal", "lognormal"):
            _, mu, sd = spec
            variants = [("mean+1sd", (kind, mu + sd, sd)), ("mean-1sd", (kind, mu - sd, sd)),
                        ("sd x2", (kind, mu, 2 * sd)), ("sd x0.5", (kind, mu, 0.5 * sd))]
        elif kind == "halfnormal":
            variants = [("scale x2", (kind, 2 * spec[1])), ("scale x0.5", (kind, 0.5 * spec[1]))]
        elif kind == "uniform":
            lo, hi = spec[1], spec[2]
            w = hi - lo
            variants = [("widen 50%", (kind, lo - 0.25 * w, hi + 0.25 * w))]
        else:
            continue
        for label, new in variants:
            out.append({"param": name, "variant": label, "overrides": {name: new}})
    return out


def scenario_metrics(draws, weights, design, plants, ks_list, rng_seed=11) -> dict:
    S = weights.size
    xc = predict_xc(draws, design, np.random.default_rng(rng_seed))
    res = {
        "ess": float(1.0 / np.sum(weights**2)),
        "Xc_p50": float(weighted_quantile(xc, weights, 0.5)),
        "Xc_p5": float(weighted_quantile(xc, weights, 0.05)),
        "Xc_p95": float(weighted_quantile(xc, weights, 0.95)),
        "omega_eff_pct": float(np.sum(weights * (draws["omega_s0"]
                                                  * (1 - reactivation_model(draws, design.phi))))),
        "lam_dtmu_p50": float(weighted_quantile(
            lam_dtmu_model(draws, design.T, design.phi), weights, 0.5)),
    }
    for plant in plants:
        P = plant.sample(S)
        for ks in ks_list:
            res[f"P(Q>1)|k_s={ks}|{plant.name}"] = float(np.sum(weights * (q_eng(xc, ks, P) > 1)))
    return res


def prior_sensitivity_table(df, idata_base, design: Design,
                            scenarios: list[dict] | None = None,
                            plants: list[PlantPrior] | None = None,
                            ks_list=(0.90, 0.95),
                            base_priors=BASE_PRIORS,
                            k_threshold=0.7, max_refits=5,
                            refit_kwargs: dict | None = None) -> pd.DataFrame:
    """Tabulate how posterior quantities change when the priors are perturbed.

    For each scenario, log w = log p_new(θ) - log p_base(θ) is evaluated on the baseline posterior draws
    and the draws are reweighted with PSIS.
    If k̂ > k_threshold the scenario is refit by MCMC (up to max_refits times); beyond that it is marked 'unreliable'.
    """
    scenarios = scenarios if scenarios is not None else auto_prior_scenarios(base_priors)
    plants = plants or [PlantPrior()]
    refit_kwargs = refit_kwargs or {"draws": 800, "tune": 1000}
    draws = posterior_draws(idata_base)
    S = next(iter(draws.values())).size

    base = scenario_metrics(draws, np.full(S, 1.0 / S), design, plants, ks_list)
    rows = [{"param": "(base)", "variant": "", "method": "posterior", "k_hat": 0.0, **base}]
    n_refit = 0
    for sc in scenarios:
        log_w = np.zeros(S)
        for name, new in sc["overrides"].items():
            log_w += log_prior(new, draws[name]) - log_prior(base_priors[name], draws[name])
        w, k = psis(log_w)
        method, d = "psis", draws
        if k > k_threshold:
            if n_refit < max_refits:
                pri = {**base_priors, **sc["overrides"]}
                d = posterior_draws(fit(df, pri, **refit_kwargs))
                w = np.full(next(iter(d.values())).size, 1.0 / next(iter(d.values())).size)
                method = "refit"
                n_refit += 1
            else:
                method = "unreliable"
        rows.append({"param": sc["param"], "variant": sc["variant"], "method": method,
                     "k_hat": k, **scenario_metrics(d, w, design, plants, ks_list)})

    table = pd.DataFrame(rows)
    pcols = [c for c in table.columns if c.startswith("P(Q>1)")]
    for c in pcols:
        table[f"Δ{c}"] = table[c] - table.loc[0, c]
    return table


def influence_ranking(table: pd.DataFrame) -> pd.DataFrame:
    """Rank parameters by the largest change in P(Q>1) (unreliable rows excluded)."""
    dcols = [c for c in table.columns if c.startswith("ΔP(Q>1)")]
    t = table[(table["param"] != "(base)") & (table["method"] != "unreliable")]
    r = t.groupby("param")[dcols].agg(lambda s: s.abs().max())
    r["max_abs_ΔP"] = r.max(axis=1)
    return r.sort_values("max_abs_ΔP", ascending=False)


# ---------------------------------------------------------------------------
# Pipeline validation with synthetic data
# ---------------------------------------------------------------------------

SYNTH_TRUTH = {
    "log_lam_dt": math.log(3.0e8), "log_kappa": math.log(3.0),
    "b0": math.log(5.0e8), "b1": 0.35, "b2": -0.15, "b3": 0.40,
    "omega_s0": 0.88, "r0": -0.50, "r1": 0.60, "sigma_model": 0.03,
}


def make_synthetic(truth=SYNTH_TRUTH, rel_noise=0.05, seed=5) -> pd.DataFrame:
    """Generate synthetic data on a grid resembling the Jones et al. measurements from the true values (not real data)."""
    rng = np.random.default_rng(seed)
    rows = []

    def add(obs, T, phi, ct, value, sigma):
        rows.append({"source": "SYNTHETIC (not real data)", "observable": obs,
                     "value": value, "sigma": sigma, "T_K": T, "phi": phi, "c_t": ct,
                     "c_he": 0.0, "use": 1, "verified": 0})

    for T in [15, 25, 100, 300, 550]:
        for phi in [0.1, 0.4, 0.8, 1.2]:
            for ct in [0.1, 0.3, 0.5, 0.7]:
                o = layer1_forward(truth, {"T": T, "phi": phi, "c_t": ct, "c_he": 0.0})
                true = o["lam_c"] / phi * (1 + truth["sigma_model"] * rng.standard_normal())
                s = rel_noise * true
                add("lam_c_over_phi", T, phi, ct, true + s * rng.standard_normal(), s)
    for T, phi, ct in [(25, 1.2, 0.4), (12, 1.45, 0.35), (300, 0.8, 0.5)]:
        o = layer1_forward(truth, {"T": T, "phi": phi, "c_t": ct, "c_he": 0.0})
        true = o["Xc"] * (1 + truth["sigma_model"] * rng.standard_normal())
        add("Xc", T, phi, ct, true + 8.0 * rng.standard_normal(), 8.0)
    for phi in [0.17, 0.5, 1.3]:
        w = layer1_forward(truth, {"T": 25, "phi": phi, "c_t": 0.4, "c_he": 0.0})["omega_eff_pct"]
        add("omega_s_pct", np.nan, phi, np.nan, w + 0.03 * rng.standard_normal(), 0.03)

    df = pd.DataFrame(rows)
    for c in ["T", "phi", "c_t"]:
        df[f"{c}_lo"] = np.nan
        df[f"{c}_hi"] = np.nan
    return df


def recovery_table(idata, truth=SYNTH_TRUTH) -> pd.DataFrame:
    d = posterior_draws(idata)
    rows = []
    for k in GLOBAL_PARAMS:
        lo, med, hi = np.percentile(d[k], [5, 50, 95])
        rows.append({"param": k, "truth": truth[k], "p5": lo, "p50": med, "p95": hi,
                     "covered_90%": bool(lo <= truth[k] <= hi),
                     "post_sd/prior_sd": np.std(d[k]) / prior_sd(BASE_PRIORS[k])})
    return pd.DataFrame(rows)


def prior_sd(spec):
    if spec[0] in ("normal", "lognormal"):
        return spec[2]
    if spec[0] == "halfnormal":
        return spec[1] * math.sqrt(1 - 2 / math.pi)
    return (spec[2] - spec[1]) / math.sqrt(12)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_posterior(idata, df, design: Design, path="mucf_layer1_posterior.png", truth=None):
    import matplotlib.pyplot as plt

    d = posterior_draws(idata)
    rng = np.random.default_rng(0)
    pr = {k: sample_prior(BASE_PRIORS[k], 4000, rng) for k in GLOBAL_PARAMS}

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    Ts = np.geomspace(10, 800, 80)
    for src, lab, c in [(pr, "prior", "0.6"), (d, "posterior", "C0")]:
        L = lam_dtmu_model({k: v[:, None] for k, v in src.items()}, Ts[None, :], design.phi)
        lo, med, hi = np.percentile(L, [5, 50, 95], axis=0)
        ax[0].fill_between(Ts, lo, hi, color=c, alpha=0.3, label=f"{lab} 90%")
        ax[0].plot(Ts, med, color=c)
    if truth:
        ax[0].plot(Ts, lam_dtmu_model(truth, Ts, design.phi), "k--", label="truth (synthetic)")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("T [K]"); ax[0].set_ylabel(f"λ_dtμ [1/s]  (φ={design.phi})")
    ax[0].legend(fontsize=8)

    phis = np.linspace(0.05, 1.5, 60)
    for src, lab, c in [(pr, "prior", "0.6"), (d, "posterior", "C0")]:
        W = src["omega_s0"][:, None] * (1 - reactivation_model({k: v[:, None] for k, v in src.items()},
                                                               phis[None, :]))
        lo, med, hi = np.percentile(W, [5, 50, 95], axis=0)
        ax[1].fill_between(phis, lo, hi, color=c, alpha=0.3, label=f"{lab} 90%")
        ax[1].plot(phis, med, color=c)
    om = df[df["observable"] == "omega_s_pct"]
    phi_pts = om["phi"].fillna(0.5 * (om["phi_lo"] + om["phi_hi"]))
    xerr = (0.5 * (om["phi_hi"] - om["phi_lo"])).fillna(0)
    ax[1].errorbar(phi_pts, om["value"], yerr=om["sigma"], xerr=xerr, fmt="o", color="C3",
                   ms=4, label="data")
    ax[1].set_xlabel("φ (LHD units)"); ax[1].set_ylabel("effective sticking ω_s [%]")
    ax[1].legend(fontsize=8)

    cts = np.linspace(0.05, 0.9, 60)
    X = layer1_forward({k: v[:, None] for k, v in d.items()},
                       {"T": design.T, "phi": design.phi, "c_t": cts[None, :], "c_he": design.c_he})["Xc"]
    lo, med, hi = np.percentile(X, [5, 50, 95], axis=0)
    ax[2].fill_between(cts, lo, hi, alpha=0.3, color="C0", label="posterior 90%")
    ax[2].plot(cts, med, color="C0")
    xd = df[df["observable"] == "Xc"]
    ct_pts = xd["c_t"].fillna(0.5 * (xd["c_t_lo"] + xd["c_t_hi"]))
    ax[2].errorbar(ct_pts, xd["value"], yerr=xd["sigma"],
                   xerr=(0.5 * (xd["c_t_hi"] - xd["c_t_lo"])).fillna(0),
                   fmt="o", color="C3", ms=4, label="Xc data (all T, φ)")
    ax[2].set_xlabel("tritium fraction c_t"); ax[2].set_ylabel("Xc")
    ax[2].set_title(f"T={design.T} K, φ={design.phi}", fontsize=9)
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="mucf_observations.csv")
    ap.add_argument("--synthetic-check", action="store_true",
                    help="fit synthetic instead of real data and check recovery of the true values")
    ap.add_argument("--sensitivity", action="store_true", help="build the prior sensitivity table")
    ap.add_argument("--max-refits", type=int, default=5)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1500)
    ap.add_argument("--design-T", type=float, default=25.0)
    ap.add_argument("--design-phi", type=float, default=1.2)
    ap.add_argument("--design-ct", type=float, default=0.4)
    ap.add_argument("--design-che", type=float, default=0.0)
    args = ap.parse_args()

    design = Design(args.design_T, args.design_phi, args.design_ct, args.design_che)
    df = make_synthetic() if args.synthetic_check else load_observations(args.data)
    tag = "synthetic" if args.synthetic_check else "real"
    print(f"data: {len(df)} rows  " + str(df["observable"].value_counts().to_dict()))

    idata = fit(df, draws=args.draws, tune=args.tune)
    import arviz as az
    rhat = float(pd.to_numeric(az.summary(idata, var_names=GLOBAL_PARAMS)["r_hat"]).max())
    div = int(idata.sample_stats["diverging"].sum())
    print(f"convergence: max r_hat = {rhat:.3f},  divergences = {div}")

    if args.synthetic_check:
        print("\n=== Recovery of true values ===")
        print(recovery_table(idata).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    else:
        d = posterior_draws(idata)
        print("\n=== Posterior (5 / 50 / 95%) and width relative to the prior ===")
        for k in GLOBAL_PARAMS:
            p = np.percentile(d[k], [5, 50, 95])
            print(f" {k:12s} {p[0]:9.3f} {p[1]:9.3f} {p[2]:9.3f}   "
                  f"post_sd/prior_sd = {np.std(d[k]) / prior_sd(BASE_PRIORS[k]):.2f}")

    d = posterior_draws(idata)
    xc = predict_xc(d, design, np.random.default_rng(3))
    print(f"\n=== Posterior prediction of Xc at {design} ===")
    print(" 5/50/95% =", np.round(np.percentile(xc, [5, 50, 95]), 1))
    P = PlantPrior().sample(xc.size)
    print(" k_s   P(Q_eng>1)   Xc* median")
    for ks in [0.85, 0.90, 0.95]:
        print(f" {ks:.2f}  {np.mean(q_eng(xc, ks, P) > 1):8.3f}   {np.median(breakeven_xc(ks, P)):8.1f}")

    plot_posterior(idata, df, design, path=f"mucf_layer1_posterior_{tag}.png",
                   truth=SYNTH_TRUTH if args.synthetic_check else None)
    print(f"\nSaved figure to mucf_layer1_posterior_{tag}.png")

    if args.sensitivity:
        plants = [PlantPrior(),
                  PlantPrior("Kelly 2021 (4.7 GeV, 0.18)", 4.7, 0.15, 0.18, 0.04),
                  PlantPrior("pessimistic (6 GeV, 0.12)", 6.0, 0.15, 0.12, 0.03)]
        table = prior_sensitivity_table(df, idata, design, plants=plants,
                                        max_refits=args.max_refits)
        path = f"mucf_prior_sensitivity_{tag}.csv"
        table.to_csv(path, index=False)
        show = ["param", "variant", "method", "k_hat", "Xc_p50",
                f"P(Q>1)|k_s=0.9|{plants[0].name}", f"ΔP(Q>1)|k_s=0.9|{plants[0].name}"]
        print("\n=== Prior sensitivity table (excerpt, k_s = 0.90, baseline plant) ===")
        print(table[show].rename(columns={show[5]: "P(Q>1)", show[6]: "ΔP"})
              .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print("\n=== Parameter influence (max |ΔP(Q>1)|) ===")
        print(influence_ranking(table)[["max_abs_ΔP"]].to_string(float_format=lambda x: f"{x:.3f}"))
        print(f"\nSaved all columns to {path}")


if __name__ == "__main__":
    main()
