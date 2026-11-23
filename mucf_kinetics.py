"""
mucf_kinetics.py — Rate-equation model of muon-catalyzed fusion (μCF) in D/T mixtures (layer 1)

The model structure follows the standard "muon cycle" rate equations, as reviewed in
Breunlich, Kammel, Cohen, Leon, Annu. Rev. Nucl. Part. Sci. 39 (1989) 311 and
Ponomarev, Contemp. Phys. 31 (1990) 219. Rate constants are representative literature values;
**check every value against the original sources before use** (see PARAMS_SOURCE).

Transient states (occupation probability per muon):
    0: dμ(1s)   1: tμ(1s)   2: ddμ   3: dtμ   4: ttμ
Absorbing (muon-loss) channels:
    decay λ0, (αμ) sticking, (³Heμ) sticking (ddμ neutron branch), ttμ sticking, capture by ³He

The model is linear with constant coefficients:  dN/dt = B N,  N(0) = b
  → expected number of fusions per muon is exact:  Y = λ_f [(-B)^{-1} b]_state
    (no time integration needed; numerically stable even for the stiff λ_f ~ 1e12 s^-1)
  → time evolution is obtained analytically from the eigendecomposition of B

Simplifications (extension points):
  - the excited-state cascade and excited-state d→t transfer are lumped into q1s (instantaneous)
  - hyperfine states of dμ and tμ are not resolved; spin-averaged effective rates are used
  - dtμ formation is temperature dependent (Vesman resonance) but is a constant input here
  - reactivation enters as ω_s = ω_s0 (1 - R), with R a density-independent constant input
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

LAMBDA0 = 1.0 / 2.197e-6  # free muon decay rate [s^-1] (τ_μ = 2.197 μs)

STATE_NAMES = ["dmu", "tmu", "ddmu", "dtmu", "ttmu"]
D, T, DD, DT, TT = range(5)


@dataclass(frozen=True)
class MuCFParams:
    """Rate constants are normalized to liquid hydrogen density (LHD, N0 = 4.25e22 atoms/cm^3) [s^-1].
    Collisional rates are multiplied by φ (density in LHD units) and the atomic fraction of the partner."""

    # --- target conditions ---
    phi: float = 1.0        # density (LHD units)
    c_t: float = 0.4        # tritium atomic fraction
    c_he: float = 0.0       # ³He atomic fraction (accumulates from tritium decay)

    # --- atomic processes ---
    kappa_q1s: float = 3.5     # q1s = 1/(1 + κ c_t): model for the probability that excited dμ reaches 1s
    lam_dt: float = 2.8e8      # dμ(1s) + t → tμ + d  transfer
    lam_d_he: float = 1.0e8    # dμ + ³He → capture (effective)
    lam_t_he: float = 1.0e8    # tμ + ³He → capture (effective)

    # --- molecular formation ---
    lam_dtmu: float = 6.0e8    # tμ + D → dtμ (effective; constant stand-in for the T-dependent resonance)
    lam_ddmu: float = 2.0e6    # dμ + D → ddμ (spin-averaged effective value)
    lam_ttmu: float = 2.0e6    # tμ + T → ttμ

    # --- intramolecular fusion ---
    lam_f_dt: float = 1.15e12  # dtμ → α + n + μ
    lam_f_dd: float = 4.0e8    # ddμ → (³He + n) or (t + p) + μ
    lam_f_tt: float = 1.5e7    # ttμ → α + 2n + μ

    # --- sticking ---
    omega_s0: float = 0.00857  # initial dt sticking ω_s^0 (Kamimura+ PRC 107, 034607 (2023))
    reactivation: float = 0.35 # reactivation probability R (ω_s = ω_s0 (1 - R))
    br_dd_n: float = 0.58      # ³He + n branching ratio of ddμ
    omega_dd: float = 0.12     # sticking probability in the ³He branch
    omega_tt: float = 0.14     # sticking probability in ttμ

    @property
    def c_d(self) -> float:
        return 1.0 - self.c_t - self.c_he

    @property
    def q1s(self) -> float:
        return 1.0 / (1.0 + self.kappa_q1s * self.c_t)

    @property
    def omega_s(self) -> float:
        return self.omega_s0 * (1.0 - self.reactivation)


# Provenance of each parameter (template for the parameter table of a report)
PARAMS_SOURCE = {
    "LAMBDA0": "verified: PDG, Navas+ PRD 110, 030001 (2024)",
    "lam_dt": "to be checked against the tables in Breunlich+ ARNPS 39 (1989); unverified",
    "lam_dtmu": "resonant formation: Cohen & Leon PRL 55, 52 (1985); measurement: Fujiwara+ PRL 85, 1642 (2000); "
                "high-density effect: Jones+ PRL 56, 588 (1986). Value is an effective guess; unverified, to be fitted",
    "lam_ddmu": "differs strongly between F=3/2 and F=1/2; spin-averaged guess; unverified",
    "lam_ttmu": "guess; unverified",
    "lam_f_dt": "verified: 1.15e12 s^-1, Kamimura+ PRC 107, 034607 (2023); Wu & Kamimura PRC 109, 054625 (2024)",
    "lam_f_dd": "guess; low sensitivity because formation is rate-limiting; unverified",
    "lam_f_tt": "guess; unverified",
    "omega_s0": "verified: 0.857%, Kamimura+ PRC 107, 034607 (2023)",
    "reactivation": "R=0.35 (high density): adopted by Kamimura+ 2023; Rafelski+ PPNP 22, 279 (1989)",
    "kappa_q1s": "effective model for q1s; to be matched to q1s fits such as Jones+ 1986",
    "lam_d_he, lam_t_he": "effective ³He capture rates; order-of-magnitude guesses only; unverified",
    "br_dd_n, omega_dd, omega_tt": "guesses; unverified",
}


def recapture_vector(p: MuCFParams) -> np.ndarray:
    """Initial distribution b of a stopped or released muon (cascade treated as instantaneous)."""
    b = np.zeros(5)
    b[D] = p.c_d * p.q1s           # dμ that reached the ground state
    b[T] = 1.0 - b[D]              # direct tμ + transfer from excited states (direct ³He capture neglected)
    return b


def generator(p: MuCFParams) -> np.ndarray:
    """Matrix B of dN/dt = B N (column = origin state, row = destination)."""
    phi, cd, ct, che = p.phi, p.c_d, p.c_t, p.c_he
    b = recapture_vector(p)
    B = np.zeros((5, 5))

    # dμ(1s)
    out_d = LAMBDA0 + phi * (p.lam_dt * ct + p.lam_ddmu * cd + p.lam_d_he * che)
    B[D, D] -= out_d
    B[T, D] += phi * p.lam_dt * ct
    B[DD, D] += phi * p.lam_ddmu * cd

    # tμ(1s)
    out_t = LAMBDA0 + phi * (p.lam_dtmu * cd + p.lam_ttmu * ct + p.lam_t_he * che)
    B[T, T] -= out_t
    B[DT, T] += phi * p.lam_dtmu * cd
    B[TT, T] += phi * p.lam_ttmu * ct

    # intramolecular fusion and recycling of muons that did not stick
    for mol, lam_f, p_free in (
        (DD, p.lam_f_dd, 1.0 - p.br_dd_n * p.omega_dd),
        (DT, p.lam_f_dt, 1.0 - p.omega_s),
        (TT, p.lam_f_tt, 1.0 - p.omega_tt),
    ):
        B[mol, mol] -= LAMBDA0 + lam_f
        B[:, mol] += lam_f * p_free * b

    return B


def analyze(p: MuCFParams) -> dict:
    """Expected number of fusions per muon and the breakdown of muon losses."""
    B = generator(p)
    b = recapture_vector(p)
    I = np.linalg.solve(-B, b)  # expected residence time in each state ∫N dt [s]

    Y_dt = p.lam_f_dt * I[DT]
    Y_dd = p.lam_f_dd * I[DD]
    Y_tt = p.lam_f_tt * I[TT]

    loss = {
        "decay": LAMBDA0 * I.sum(),
        "stick_dt": p.lam_f_dt * p.omega_s * I[DT],
        "stick_dd": p.lam_f_dd * p.br_dd_n * p.omega_dd * I[DD],
        "stick_tt": p.lam_f_tt * p.omega_tt * I[TT],
        "he_capture": p.phi * p.c_he * (p.lam_d_he * I[D] + p.lam_t_he * I[T]),
    }
    total = sum(loss.values())  # probability conservation check (should equal 1)

    # observed neutron disappearance rate = slowest eigenvalue (corresponds to λ0 + W λc)
    eigvals = np.linalg.eigvals(B)
    lam_n = -max(eigvals.real)

    return {
        "Xc": Y_dt, "Y_dd": Y_dd, "Y_tt": Y_tt,
        "loss": loss, "prob_check": total,
        "lam_n": lam_n,
        "lam_c_eff": Y_dt / I.sum(),  # effective cycling rate = fusions / muon lifetime
    }


def cycle_formula(p: MuCFParams) -> dict:
    """Textbook steady-state cycle approximation (ignores dd and tt branches and ³He), for comparison.
        1/λc = q1s c_d / (φ c_t λ_dt) + 1/(φ c_d λ_dtμ)
        Xc   = λc / (λ0 + ω_s λc)"""
    tau_c = (p.q1s * p.c_d / (p.phi * p.c_t * p.lam_dt)
             + 1.0 / (p.phi * p.c_d * p.lam_dtmu)
             + 1.0 / p.lam_f_dt)
    lam_c = 1.0 / tau_c
    return {"lam_c": lam_c, "Xc": lam_c / (LAMBDA0 + p.omega_s * lam_c)}


def neutron_time_spectrum(p: MuCFParams, t: np.ndarray) -> np.ndarray:
    """dt fusion neutron emission rate dY/dt(t) [s^-1] per muon, from the eigendecomposition."""
    B = generator(p)
    w, V = np.linalg.eig(B)
    coef = np.linalg.solve(V, recapture_vector(p))
    N_dt = (V[DT, :] * coef) @ np.exp(np.outer(w, t))
    return p.lam_f_dt * N_dt.real


if __name__ == "__main__":
    base = MuCFParams()

    print("=== Baseline φ=1.0, c_t=0.4, c_He=0 ===")
    r = analyze(base)
    c = cycle_formula(base)
    print(f"Xc (ODE)           = {r['Xc']:.1f}")
    print(f"Xc (cycle approx.) = {c['Xc']:.1f}")
    print(f"λc effective = {r['lam_c_eff']:.3e} s^-1,  approx. = {c['lam_c']:.3e} s^-1")
    print(f"neutron disappearance rate λn = {r['lam_n']:.3e} s^-1  (λ0 = {LAMBDA0:.3e})")
    print(f"ddμ fusions {r['Y_dd']:.2f},  ttμ fusions {r['Y_tt']:.3f}")
    print("muon loss breakdown:", {k: round(v, 4) for k, v in r["loss"].items()})
    print(f"probability conservation check Σ = {r['prob_check']:.6f}")

    print("\n=== Tritium fraction scan (φ=1.0) ===")
    print(" c_t   q1s    λc[1e8/s]  Xc     stick_loss decay_loss")
    for ct in np.arange(0.1, 0.95, 0.1):
        p = replace(base, c_t=float(ct))
        r = analyze(p)
        print(f" {ct:.1f}  {p.q1s:.3f}  {r['lam_c_eff']/1e8:8.2f}  {r['Xc']:6.1f}"
              f"   {r['loss']['stick_dt']:.3f}    {r['loss']['decay']:.3f}")

    print("\n=== Effect of ³He accumulation (φ=1.0, c_t=0.4) ===")
    for che in [0.0, 1e-4, 1e-3, 1e-2]:
        r = analyze(replace(base, c_he=che))
        print(f" c_He = {che:.0e}: Xc = {r['Xc']:6.1f},  He capture loss = {r['loss']['he_capture']:.3f}")

    print("\n=== Sticking-limited ceiling (infinite cycling rate) ===")
    print(f" 1/ω_s = {1/base.omega_s:.0f}")

    try:
        import matplotlib.pyplot as plt

        cts = np.linspace(0.05, 0.95, 60)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for phi in [0.5, 1.0, 1.2]:
            ax[0].plot(cts, [analyze(replace(base, phi=phi, c_t=float(x)))["Xc"] for x in cts],
                       label=f"φ = {phi}")
        ax[0].set_xlabel("tritium fraction c_t")
        ax[0].set_ylabel("Xc (dt fusions per muon)")
        ax[0].legend()
        ax[0].grid(alpha=0.3)

        t = np.linspace(0, 10e-6, 400)
        ax[1].semilogy(t * 1e6, neutron_time_spectrum(base, t))
        ax[1].set_xlabel("time after muon stop [μs]")
        ax[1].set_ylabel("dt neutron rate per muon [1/s]")
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig("mucf_kinetics.png", dpi=120)
        print("\nSaved figure to mucf_kinetics.png")
    except ImportError:
        pass
