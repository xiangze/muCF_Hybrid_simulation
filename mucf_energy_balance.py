"""
mucf_energy_balance.py — Energy balance Q_eng of a μCF fission hybrid (layer 3)

Inputs:
  from layer 1 (mucf_kinetics.py)        Xc     = dt fusions per muon
  from layer 2 (mucf_blanket_openmc.py)  E_heat = total blanket heating per source neutron [MeV]
                                         (or an analytic approximation from k_s)
and computes the engineering energy gain Q_eng of the whole plant.

The balance is taken per muon, in MeV:

    E_th     = Xc · (E_α + E_heat) + f_rec · E_μ                 thermal output
    E_gross  = η_th · E_th                                       gross electricity
    E_recirc = E_μ / η_acc + f_aux · E_gross                     recirculating power
    Q_eng    = E_gross / E_recirc

    E_heat (analytic) = E_n,dep + E_f · F,   F = k_s / (ν (1 - k_s))
        F is the number of fissions per source neutron (from νF = k_s/(1-k_s))

Breakeven (Q_eng = 1) is linear in Xc, so it has a closed form:

    Xc* = [ E_μ / (η_acc η_th (1 - f_aux)) - f_rec E_μ ] / (E_α + E_heat)

Neutron cost comparison (crossover with spallation ADS):
    μCF:         E_μ / Xc          [MeV/neutron]
    spallation:  E_p / Y_spall     [MeV/neutron]
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (public values)
# ---------------------------------------------------------------------------
E_ALPHA = 3.52     # MeV, d + t → α(3.52) + n(14.06)           [public: reaction Q value 17.59 MeV]
E_N_DT = 14.06     # MeV
Q_LI6_NT = 4.78    # MeV, Q value of ⁶Li(n,t)α                  [public]


@dataclass(frozen=True)
class PlantParams:
    # --- layer 1 ---
    Xc: float = 119.0            # dt fusions per muon (layer-1 baseline result)

    # --- muon source / accelerator ---
    E_mu: float = 5000.0         # MeV, beam energy required per muon                 [assumed: literature estimates are a few GeV]
    eta_acc: float = 0.25        # accelerator wall-plug-to-beam efficiency            [assumed: about 0.1–0.4]
    f_rec: float = 0.0           # fraction of beam energy in the production target recovered as heat [assumed]

    # --- blanket (layer 2) ---
    k_s: float = 0.95            # source multiplication factor (from the layer-2 fixed-source run)
    nu: float = 2.49             # neutrons per fission (U-233, near thermal)          [representative public nuclear-data value]
    E_f: float = 198.0           # MeV, recoverable energy per fission                 [representative public value]
    TBR: float = 1.05            # tritium breeding ratio (affects ⁶Li(n,t) heating)
    E_heat_override: float | None = None  # prefer the layer-2 heating tally when available [MeV per source neutron]

    # --- power conversion / house load ---
    eta_th: float = 0.40         # thermal-to-electric efficiency (molten salt, ~650°C) [assumed]
    f_aux: float = 0.05          # auxiliaries (cryogenics, tritium plant, ...) as a fraction of gross power [assumed]


def fissions_per_source_neutron(k_s: float, nu: float) -> float:
    return k_s / (nu * (1.0 - k_s)) if k_s > 0 else 0.0


def blanket_heat(p: PlantParams) -> float:
    """Blanket heating per source neutron [MeV]."""
    if p.E_heat_override is not None:
        return p.E_heat_override
    E_n_dep = E_N_DT + p.TBR * Q_LI6_NT  # neutron kinetic energy + exothermic reactions (mainly ⁶Li)
    return E_n_dep + p.E_f * fissions_per_source_neutron(p.k_s, p.nu)


def energy_balance(p: PlantParams) -> dict:
    """Per-muon energy balance [MeV] and Q_eng."""
    E_unit = E_ALPHA + blanket_heat(p)              # heat per dt fusion
    E_th = p.Xc * E_unit + p.f_rec * p.E_mu
    E_gross = p.eta_th * E_th
    E_recirc = p.E_mu / p.eta_acc + p.f_aux * E_gross
    return {
        "E_per_fusion_MeV": E_unit,
        "E_th_per_muon_MeV": E_th,
        "E_gross_per_muon_MeV": E_gross,
        "E_recirc_per_muon_MeV": E_recirc,
        "Q_eng": E_gross / E_recirc,
        "recirc_fraction": E_recirc / E_gross,
    }


def Q_eng(p: PlantParams) -> float:
    return energy_balance(p)["Q_eng"]


def breakeven_Xc(p: PlantParams, Q_target: float = 1.0) -> float:
    """Xc at which Q_eng = Q_target (closed form)."""
    E_unit = E_ALPHA + blanket_heat(p)
    a = p.eta_th * (1.0 - Q_target * p.f_aux)
    return (Q_target * p.E_mu / p.eta_acc - a * p.f_rec * p.E_mu) / (a * E_unit)


def plant_power(p: PlantParams, P_net_target_MWe: float) -> dict:
    """Muon supply rate and beam power required for a net electric output P_net."""
    eb = energy_balance(p)
    net_MeV = eb["E_gross_per_muon_MeV"] - eb["E_recirc_per_muon_MeV"]
    if net_MeV <= 0:
        return {"feasible": False}
    MeV_J = 1.602176634e-13
    R_mu = P_net_target_MWe * 1e6 / (net_MeV * MeV_J)
    return {
        "feasible": True,
        "muon_rate_per_s": R_mu,
        "beam_power_MW": R_mu * p.E_mu * MeV_J / 1e6,
        "thermal_power_MW": R_mu * eb["E_th_per_muon_MeV"] * MeV_J / 1e6,
        "fusion_neutron_rate_per_s": R_mu * p.Xc,
    }


# --- neutron cost and comparison with spallation ADS -------------------------

def neutron_cost_mucf(p: PlantParams) -> float:
    """MeV per neutron (based on beam energy)."""
    return p.E_mu / p.Xc


def neutron_cost_spallation(E_p: float = 1000.0, yield_n: float = 28.0) -> float:
    """[assumed] rough guide of 25–30 neutrons for a 1 GeV proton on a thick Pb target."""
    return E_p / yield_n


# --- uncertainty propagation --------------------------------------------------

def monte_carlo(p: PlantParams, n: int = 20000, seed: int = 0) -> dict:
    """Treat key parameters as distributions and return the distribution of Q_eng (distribution shapes are assumed)."""
    rng = np.random.default_rng(seed)
    samples = {
        "Xc": rng.uniform(80, 160, n),
        "E_mu": np.exp(rng.uniform(np.log(2000), np.log(10000), n)),  # log-uniform 2–10 GeV
        "eta_acc": rng.uniform(0.10, 0.40, n),
        "eta_th": rng.uniform(0.33, 0.45, n),
        "k_s": rng.uniform(0.90, 0.97, n),
    }
    q = np.array([Q_eng(replace(p, **{k: float(v[i]) for k, v in samples.items()}))
                  for i in range(n)])
    # simple sensitivity index: rank correlation between each input and Q
    ranks = lambda x: np.argsort(np.argsort(x))
    sens = {k: float(np.corrcoef(ranks(v), ranks(q))[0, 1]) for k, v in samples.items()}
    return {
        "Q_percentiles": dict(zip(["p5", "p50", "p95"], np.percentile(q, [5, 50, 95]))),
        "P(Q>1)": float((q > 1).mean()),
        "rank_correlation": sens,
    }


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    base = PlantParams()

    print("=== Pure fusion (no fission in blanket, k_s=0) vs hybrid ===")
    for label, ks in [("pure μCF", 0.0), ("hybrid k_s=0.90", 0.90),
                      ("hybrid k_s=0.95", 0.95), ("hybrid k_s=0.97", 0.97)]:
        p = replace(base, k_s=ks)
        eb = energy_balance(p)
        print(f" {label:22s} per fusion {eb['E_per_fusion_MeV']:7.1f} MeV  "
              f"Q_eng = {eb['Q_eng']:6.2f}   breakeven Xc* = {breakeven_Xc(p):8.1f}")

    print("\n=== Scale required for 1 GWe net (baseline k_s=0.95) ===")
    for k, v in plant_power(base, 1000.0).items():
        print(f" {k:28s} {v:.3e}" if isinstance(v, float) else f" {k:28s} {v}")

    print("\n=== Neutron cost (beam-energy basis) ===")
    print(f" μCF  : {neutron_cost_mucf(base):6.1f} MeV/n  (E_μ={base.E_mu:.0f} MeV, Xc={base.Xc:.0f})")
    print(f" spall: {neutron_cost_spallation():6.1f} MeV/n  (1 GeV protons, 28 n/p assumed)")

    print("\n=== Uncertainty propagation (assumed distributions) ===")
    mc = monte_carlo(base, n=5000)
    print(" Q_eng 5/50/95%:", {k: round(v, 2) for k, v in mc["Q_percentiles"].items()})
    print(" P(Q_eng > 1)  :", round(mc["P(Q>1)"], 3))
    print(" rank correlation:", {k: round(v, 2) for k, v in mc["rank_correlation"].items()})

    try:
        import matplotlib.pyplot as plt

        xc = np.linspace(20, 400, 200)
        emu = np.linspace(500, 10000, 200)
        X, Y = np.meshgrid(xc, emu)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for ax, ks, title in [(axes[0], 0.0, "pure μCF (k_s = 0)"),
                              (axes[1], 0.95, "hybrid (k_s = 0.95)")]:
            Z = np.vectorize(lambda a, b: Q_eng(replace(base, Xc=a, E_mu=b, k_s=ks)))(X, Y)
            cs = ax.contour(X, Y / 1000, Z, levels=[0.1, 0.3, 1, 3, 10], colors="k", linewidths=0.8)
            ax.clabel(cs, fmt="Q=%g", fontsize=8)
            ax.contourf(X, Y / 1000, Z >= 1, levels=[0.5, 1.5], colors=["#cfe8d6"], alpha=0.6)
            ax.axvspan(100, 150, color="#999999", alpha=0.25, label="achieved Xc (1980s)")
            # line where the neutron cost equals spallation: E_μ / Xc = 35.7 MeV
            ax.plot(xc, xc * neutron_cost_spallation() / 1000, "--", color="#b03030",
                    label="neutron cost = spallation")
            ax.set_xlabel("Xc (dt fusions per muon)")
            ax.set_title(title)
            ax.set_ylim(0.5, 10)
            ax.legend(fontsize=8, loc="upper left")
        axes[0].set_ylabel("E_μ (GeV of beam per muon)")
        fig.tight_layout()
        fig.savefig("mucf_Qeng_map.png", dpi=120)
        print("\nSaved figure to mucf_Qeng_map.png")
    except ImportError:
        pass
