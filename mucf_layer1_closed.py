"""
mucf_layer1_closed.py — Closed-form solution of the 5-state model of layer 1 (mucf_kinetics.py)

mucf_kinetics.analyze() solves the linear system (-B) I = b numerically. Because the cycle has a
one-directional structure ("recapture → dμ/tμ → molecule → fusion → recapture"), it can be written
in closed form using per-cycle probabilities (embedded jump chain):

    S_D = λ0 + φ(λ_dt c_t + λ_ddμ c_d + λ_dHe c_He)        total exit rate from dμ
    S_T = λ0 + φ(λ_dtμ c_d + λ_ttμ c_t + λ_tHe c_He)       total exit rate from tμ
    P_T = b_T + b_D · φ λ_dt c_t / S_D                      probability of reaching tμ
    R   = b_D p_DD s_DD (1 - β ω_dd)                        probability that the muon is recycled in one cycle
        + P_T [ p_DT s_DT (1 - ω_s) + p_TT s_TT (1 - ω_tt) ]
    Xc  = P_T p_DT s_DT / (1 - R)
    λc  = (P_T p_DT s_DT) / τ_cyc,  τ_cyc = mean residence time per cycle

Only arithmetic plus exp / sigmoid is used, so the same function works with
  numpy (synthetic data generation, posterior evaluation) and pytensor (PyMC likelihood).
Agreement with analyze() is checked in test_layer1_closed.py.

For Bayesian inference, the following phenomenological parameterizations are added
(they are not derived from theory):
    log λ_dtμ(T, φ) = b0 + b1 x + b2 x² + b3 (φ - 1),   x = log(T / 100 K)
        a smooth approximation of the temperature dependence (effective Vesman resonance) and
        the density dependence (three-body collision contribution)
    ω_s,eff(φ) = ω_s0 (1 - R(φ)),   R(φ) = sigmoid(r0 + r1 (φ - 1))
        density dependence of reactivation (faster α slowing-down and stopping power at high density)
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy.special import expit

from mucf_kinetics import LAMBDA0, MuCFParams

# Constants that are not inferred (low sensitivity) take the layer-1 defaults
_K = MuCFParams()
FIXED = {
    "lam_ddmu": _K.lam_ddmu, "lam_ttmu": _K.lam_ttmu,
    "lam_f_dt": _K.lam_f_dt, "lam_f_dd": _K.lam_f_dd, "lam_f_tt": _K.lam_f_tt,
    "lam_d_he": _K.lam_d_he, "lam_t_he": _K.lam_t_he,
    "br_dd_n": _K.br_dd_n, "omega_dd": _K.omega_dd, "omega_tt": _K.omega_tt,
}

T_REF = 100.0  # K

NP = SimpleNamespace(exp=np.exp, log=np.log, sigmoid=expit)


def lam_dtmu_model(theta, T, phi, xp=NP):
    x = xp.log(T / T_REF)
    return xp.exp(theta["b0"] + theta["b1"] * x + theta["b2"] * x * x
                  + theta["b3"] * (phi - 1.0))


def reactivation_model(theta, phi, xp=NP):
    return xp.sigmoid(theta["r0"] + theta["r1"] * (phi - 1.0))


def omega_eff_pct(theta, phi, xp=NP):
    """Effective sticking [%]. theta["omega_s0"] is also in %."""
    return theta["omega_s0"] * (1.0 - reactivation_model(theta, phi, xp))


def cycle_closed_form(rates: dict, cond: dict) -> dict:
    """rates: lam_dt, kappa_q1s, lam_dtmu, omega_s (effective, as a probability) + the FIXED values
    cond : T, phi, c_t, c_he
    Works with numpy arrays or pytensor tensors (arithmetic only)."""
    r = {**FIXED, **rates}
    phi, ct, che = cond["phi"], cond["c_t"], cond["c_he"]
    cd = 1.0 - ct - che

    q1s = 1.0 / (1.0 + r["kappa_q1s"] * ct)
    bD = cd * q1s
    bT = 1.0 - bD

    a = phi * r["lam_dt"] * ct
    d = phi * r["lam_ddmu"] * cd
    SD = LAMBDA0 + a + d + phi * r["lam_d_he"] * che
    e = phi * r["lam_dtmu"] * cd
    g = phi * r["lam_ttmu"] * ct
    ST = LAMBDA0 + e + g + phi * r["lam_t_he"] * che

    pTD, pDD = a / SD, d / SD
    pDT, pTT = e / ST, g / ST
    PT = bT + bD * pTD

    SDD = LAMBDA0 + r["lam_f_dd"]
    SDT = LAMBDA0 + r["lam_f_dt"]
    STT = LAMBDA0 + r["lam_f_tt"]
    sDD, sDT, sTT = r["lam_f_dd"] / SDD, r["lam_f_dt"] / SDT, r["lam_f_tt"] / STT

    R = (bD * pDD * sDD * (1.0 - r["br_dd_n"] * r["omega_dd"])
         + PT * (pDT * sDT * (1.0 - r["omega_s"]) + pTT * sTT * (1.0 - r["omega_tt"])))
    y_dt = PT * pDT * sDT
    tau_cyc = bD / SD + PT / ST + bD * pDD / SDD + PT * pDT / SDT + PT * pTT / STT

    return {
        "Xc": y_dt / (1.0 - R),
        "lam_c": y_dt / tau_cyc,
        "Y_dd": bD * pDD * sDD / (1.0 - R),
        "Y_tt": PT * pTT * sTT / (1.0 - R),
        "loss_per_cycle": 1.0 - R,
    }


def layer1_forward(theta: dict, cond: dict, xp=NP) -> dict:
    """Compute observables from the inferred parameters theta and experimental conditions cond."""
    lam_dtmu = lam_dtmu_model(theta, cond["T"], cond["phi"], xp)
    w_pct = omega_eff_pct(theta, cond["phi"], xp)
    out = cycle_closed_form(
        {"lam_dt": xp.exp(theta["log_lam_dt"]),
         "kappa_q1s": xp.exp(theta["log_kappa"]),
         "lam_dtmu": lam_dtmu,
         "omega_s": w_pct / 100.0},
        cond)
    out["lam_dtmu"] = lam_dtmu
    out["omega_eff_pct"] = w_pct
    return out
