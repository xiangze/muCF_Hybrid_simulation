"""
mucf_bayes.py — Bayesian inference from published experimental data and the posterior probability
of breakeven (layers 1 + 3)

Structure:
  (A) Parts updated by data (PyMC)
      - effective α sticking ω_s : hierarchical (random-effects) model of three high-density D-T measurements
      - cycling rate λc          : inferred from three Xc measurements through Xc = 1/(λ0/λc + ω_s);
                                   differences in experimental conditions are treated as the
                                   between-experiment scatter τ of log λc
      → posterior predictive distribution of Xc under "conditions already reached in experiments"
  (B) Parts not updated by data (literature-based priors only)
      - muon production cost E_μ, accelerator efficiency η_acc, thermal efficiency η_th, auxiliary fraction f_aux
  (C) Evaluate the layer-3 Q_eng for each posterior draw and report, for each k_s,
      P(Q_eng > 1), the distribution of the breakeven Xc*, and the distribution of Q_eng

Important: (B) is not updated by observations, so that part of the result is merely prior propagation.
           Always report the sensitivity to the priors (--emu-sigma, --sweep, etc.).

Data sources:
  ω_s (high density, φ≈1.2–1.4):
    RIKEN-RAL liquid 0.532±0.030%, solid 0.515±0.030%, PSI liquid 0.505±0.029%
    (values quoted in eq. (5.23) of Kamimura, Kino, Yamashita, Phys. Rev. C 107, 034607 (2023))
  Xc:
    Jones+ PRL 56, 588 (1986)       150 ± 4(stat) ± 20(syst)   liquid, φ=1.2
    Breunlich+ (1987)               113 ± 10                   liquid, 24 K, φ=1.20
    Petitjean+ (1988)               124 ± 10                   solid, 12 K, φ=1.45
    (from the table in Petitjean, OECD/NEA Villigen report vil-3-04; check each original source)
  Basis for the priors:
    E_μ  : 4.70 GeV/μ in the design of Kelly+ J. Phys. Energy 3, 035003 (2021); literature range 5–6 GeV
    η_acc: 18% quoted in the same paper as the current accelerator efficiency
"""

from __future__ import annotations

import argparse

import numpy as np
import pymc as pm
import arviz as az

LAMBDA0 = 1.0 / 2.197e-6       # s^-1
E_ALPHA, E_N_DT, Q_LI6 = 3.52, 14.06, 4.78  # MeV

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
OMEGA_OBS = np.array([0.532, 0.515, 0.505])   # %
OMEGA_SD = np.array([0.030, 0.030, 0.029])
OMEGA_LABEL = ["RIKEN-RAL liquid", "RIKEN-RAL solid", "PSI liquid"]

XC_OBS = np.array([150.0, 113.0, 124.0])
XC_SD = np.array([np.hypot(4.0, 20.0), 10.0, 10.0])  # statistical and systematic added in quadrature
XC_LABEL = ["Jones 1986", "Breunlich 1987", "Petitjean 1988"]


# ---------------------------------------------------------------------------
# (A) Inference model
# ---------------------------------------------------------------------------

def build_model() -> pm.Model:
    with pm.Model() as m:
        # --- hierarchical model of the sticking probability [%] ---
        mu_w = pm.Normal("mu_omega", mu=0.55, sigma=0.10)       # weakly informative, near the theory value 0.557% (R=0.35)
        tau_w = pm.HalfNormal("tau_omega", sigma=0.05)
        z_w = pm.Normal("z_omega", 0.0, 1.0, shape=len(OMEGA_OBS))
        w_i = pm.Deterministic("omega_i", mu_w + tau_w * z_w)
        pm.Normal("omega_like", mu=w_i, sigma=OMEGA_SD, observed=OMEGA_OBS)

        # --- cycling rate (log λc is hierarchical because conditions differ between experiments) ---
        mu_l = pm.Normal("mu_log_lamc", mu=np.log(1.5e8), sigma=0.7)
        tau_l = pm.HalfNormal("tau_log_lamc", sigma=0.5)
        z_l = pm.Normal("z_lamc", 0.0, 1.0, shape=len(XC_OBS))
        lam_c = pm.Deterministic("lam_c_j", pm.math.exp(mu_l + tau_l * z_l))

        W = mu_w / 100.0  # effective loss (sticking); dd/tt-branch sticking neglected (a few % in layer 1)
        xc_j = pm.Deterministic("Xc_j", 1.0 / (LAMBDA0 / lam_c + W))
        pm.Normal("Xc_like", mu=xc_j, sigma=XC_SD, observed=XC_OBS)

        # --- prediction: λc and Xc for a "new experiment" under already-reached conditions ---
        z_new = pm.Normal("z_new", 0.0, 1.0)
        lam_new = pm.Deterministic("lam_c_new", pm.math.exp(mu_l + tau_l * z_new))
        pm.Deterministic("Xc_new", 1.0 / (LAMBDA0 / lam_new + W))
        pm.Deterministic("Xc_ceiling", 1.0 / W)   # upper limit 1/ω_s as λc → ∞
    return m


# ---------------------------------------------------------------------------
# (B) Priors on plant parameters (not updated by data)
# ---------------------------------------------------------------------------

def beta_ab(mean: float, sd: float) -> tuple[float, float]:
    k = mean * (1 - mean) / sd**2 - 1
    return mean * k, (1 - mean) * k


def sample_plant_priors(n: int, rng, emu_median=5.0, emu_sigma=0.15,
                        eta_acc_mean=0.18, eta_acc_sd=0.04) -> dict:
    return {
        "E_mu": 1000.0 * rng.lognormal(np.log(emu_median), emu_sigma, n),  # MeV
        "eta_acc": rng.beta(*beta_ab(eta_acc_mean, eta_acc_sd), n),
        "eta_th": rng.beta(*beta_ab(0.42, 0.02), n),
        "f_aux": rng.uniform(0.03, 0.08, n),
    }


# ---------------------------------------------------------------------------
# (C) Layer 3 (vectorized)
# ---------------------------------------------------------------------------

def E_per_fusion(k_s: float, nu=2.49, E_f=198.0, TBR=1.05) -> float:
    F = k_s / (nu * (1 - k_s)) if k_s > 0 else 0.0
    return E_ALPHA + E_N_DT + TBR * Q_LI6 + E_f * F


def q_eng(Xc, k_s, P):
    E_gross = P["eta_th"] * Xc * E_per_fusion(k_s)
    return E_gross / (P["E_mu"] / P["eta_acc"] + P["f_aux"] * E_gross)


def breakeven_xc(k_s, P):
    return P["E_mu"] / (P["eta_acc"] * P["eta_th"] * (1 - P["f_aux"]) * E_per_fusion(k_s))


# ---------------------------------------------------------------------------
# Prior sensitivity sweep
# ---------------------------------------------------------------------------

def sweep_plant_priors(xc_samples: np.ndarray,
                       emu_medians=(3.0, 4.0, 4.7, 5.0, 6.0, 8.0),
                       eta_acc_means=(0.10, 0.14, 0.18, 0.25, 0.35),
                       k_s_values=(0.85, 0.90, 0.95),
                       emu_sigma: float = 0.15,
                       eta_acc_sd: float | None = None,
                       seed: int = 3) -> list[dict]:
    """Compute P(Q_eng>1) and quantiles of Q_eng on a grid of E_μ median × η_acc mean.

    xc_samples are the MCMC posterior predictive draws of Xc (they do not depend on the plant priors,
    so they can be reused). The same random stream (common random numbers) is used at every grid point,
    so differences between grid points are not blurred by Monte Carlo noise.
    If eta_acc_sd is omitted, it is set to 22% of the mean (the same relative width as the baseline 0.18±0.04).
    """
    n = xc_samples.size
    rows = []
    for emu in emu_medians:
        for eta in eta_acc_means:
            sd = eta_acc_sd if eta_acc_sd is not None else 0.22 * eta
            P = sample_plant_priors(n, np.random.default_rng(seed),
                                    emu_median=emu, emu_sigma=emu_sigma,
                                    eta_acc_mean=eta, eta_acc_sd=sd)
            for ks in k_s_values:
                q = q_eng(xc_samples, ks, P)
                p5, p50, p95 = np.percentile(q, [5, 50, 95])
                rows.append({"emu_median_GeV": emu, "eta_acc_mean": eta, "k_s": ks,
                             "P_Q_gt_1": float(np.mean(q > 1)),
                             "Q_p5": p5, "Q_p50": p50, "Q_p95": p95,
                             "Xc_star_p50": float(np.median(breakeven_xc(ks, P)))})
    return rows


def print_sweep(rows: list[dict]) -> None:
    emus = sorted({r["emu_median_GeV"] for r in rows})
    etas = sorted({r["eta_acc_mean"] for r in rows})
    for ks in sorted({r["k_s"] for r in rows}):
        print(f"\n P(Q_eng > 1)   k_s = {ks}")
        print("  E_μ[GeV] \\ η_acc " + "".join(f"{e:>8.2f}" for e in etas))
        for emu in emus:
            vals = [next(r["P_Q_gt_1"] for r in rows
                         if r["k_s"] == ks and r["emu_median_GeV"] == emu and r["eta_acc_mean"] == e)
                    for e in etas]
            print(f"  {emu:>8.1f}         " + "".join(f"{v:8.3f}" for v in vals))


def save_sweep(rows: list[dict], csv_path="mucf_sweep.csv", png_path="mucf_sweep.png") -> None:
    import csv
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    emus = sorted({r["emu_median_GeV"] for r in rows})
    etas = sorted({r["eta_acc_mean"] for r in rows})
    kss = sorted({r["k_s"] for r in rows})
    fig, axes = plt.subplots(1, len(kss), figsize=(4.2 * len(kss), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ks in zip(axes, kss):
        Z = np.array([[next(r["P_Q_gt_1"] for r in rows if r["k_s"] == ks
                            and r["emu_median_GeV"] == emu and r["eta_acc_mean"] == e)
                       for e in etas] for emu in emus])
        im = ax.imshow(Z, origin="lower", cmap="viridis", vmin=0, vmax=1, aspect="auto")
        for i in range(len(emus)):
            for j in range(len(etas)):
                ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if Z[i, j] < 0.6 else "black")
        ax.set_xticks(range(len(etas)), [f"{e:.2f}" for e in etas])
        ax.set_yticks(range(len(emus)), [f"{e:g}" for e in emus])
        ax.set_xlabel("η_acc (prior mean)")
        ax.set_title(f"P(Q_eng > 1),  k_s = {ks}")
    axes[0].set_ylabel("E_μ prior median [GeV]")
    fig.colorbar(im, ax=axes.tolist(), shrink=0.85)
    fig.savefig(png_path, dpi=120, bbox_inches="tight")


def parse_floats(s: str) -> tuple[float, ...]:
    return tuple(float(x) for x in s.split(","))


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--emu-median", type=float, default=5.0, help="median of the E_μ prior [GeV]")
    ap.add_argument("--emu-sigma", type=float, default=0.15, help="log standard deviation of E_μ")
    ap.add_argument("--eta-acc", type=float, default=0.18, help="mean of the η_acc prior")
    ap.add_argument("--sweep", action="store_true", help="run the E_μ × η_acc sensitivity sweep")
    ap.add_argument("--emu-grid", type=parse_floats, default=(3.0, 4.0, 4.7, 5.0, 6.0, 8.0),
                    help="E_μ medians to sweep [GeV] (comma separated)")
    ap.add_argument("--eta-grid", type=parse_floats, default=(0.10, 0.14, 0.18, 0.25, 0.35),
                    help="η_acc means to sweep (comma separated)")
    ap.add_argument("--ks-grid", type=parse_floats, default=(0.85, 0.90, 0.95),
                    help="k_s values to sweep (comma separated)")
    args = ap.parse_args()

    with build_model():
        idata = pm.sample(args.draws, tune=1500, chains=4, target_accept=0.95,
                          random_seed=1, progressbar=False)

    print("=== (A) Parameters updated by data ===")
    names = ["mu_omega", "tau_omega", "lam_c_j", "Xc_j", "Xc_new", "Xc_ceiling"]
    try:  # ArviZ >= 1.0
        summ = az.summary(idata, var_names=names, ci_prob=0.9)
    except TypeError:  # ArviZ 0.x
        summ = az.summary(idata, var_names=names, hdi_prob=0.9)
    print(summ.to_string())

    post = idata.posterior
    xc_new = post["Xc_new"].values.ravel()
    n = xc_new.size
    rng = np.random.default_rng(2)
    P = sample_plant_priors(n, rng, args.emu_median, args.emu_sigma, args.eta_acc)

    print("\n=== (C) Posterior probability of breakeven (Xc = posterior predictive at reached conditions) ===")
    print(f" prior: E_μ median {args.emu_median} GeV (σ_log={args.emu_sigma}), η_acc mean {args.eta_acc}")
    print(" k_s    P(Q>1)   Q_eng 5/50/95%            Xc* 5/50/95%")
    results = {}
    for ks in [0.0, 0.85, 0.90, 0.95]:
        q = q_eng(xc_new, ks, P)
        xs = breakeven_xc(ks, P)
        results[ks] = (q, xs)
        qp, xp = np.percentile(q, [5, 50, 95]), np.percentile(xs, [5, 50, 95])
        print(f" {ks:.2f}  {np.mean(q > 1):6.3f}   {qp[0]:6.2f} {qp[1]:6.2f} {qp[2]:6.2f}"
              f"      {xp[0]:8.1f} {xp[1]:8.1f} {xp[2]:8.1f}")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        bins = np.linspace(0, 300, 120)
        ax[0].hist(xc_new, bins=bins, density=True, alpha=0.6, label="Xc (posterior predictive)")
        for ks, c in [(0.90, "C1"), (0.95, "C2")]:
            ax[0].hist(results[ks][1], bins=bins, density=True, histtype="step", lw=1.5,
                       color=c, label=f"breakeven Xc*  (k_s={ks})")
        ax[0].set_xlabel("dt fusions per muon")
        ax[0].legend(fontsize=8)
        ax[0].set_title("Achieved Xc vs required Xc*")

        for ks, c in [(0.85, "C3"), (0.90, "C1"), (0.95, "C2")]:
            ax[1].hist(np.log10(results[ks][0]), bins=80, density=True, histtype="step",
                       lw=1.5, color=c, label=f"k_s={ks}")
        ax[1].axvline(0, color="k", ls="--", lw=0.8)
        ax[1].set_xlabel("log10 Q_eng")
        ax[1].legend(fontsize=8)
        ax[1].set_title("Posterior of Q_eng")
        fig.tight_layout()
        fig.savefig("mucf_bayes.png", dpi=120)
        print("\nSaved figure to mucf_bayes.png")
    except ImportError:
        pass

    if args.sweep:
        print("\n=== Prior sensitivity sweep (E_μ median × η_acc mean) ===")
        rows = sweep_plant_priors(xc_new, args.emu_grid, args.eta_grid, args.ks_grid,
                                  emu_sigma=args.emu_sigma)
        print_sweep(rows)
        save_sweep(rows)
        print("\nSaved table to mucf_sweep.csv and figure to mucf_sweep.png")


if __name__ == "__main__":
    main()
