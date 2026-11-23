"""Check that the closed-form solution (mucf_layer1_closed) matches the numerical layer-1 solution (mucf_kinetics.analyze)."""

from dataclasses import replace

import numpy as np

from mucf_kinetics import MuCFParams, analyze
from mucf_layer1_closed import cycle_closed_form


def test_closed_form_matches_matrix_solution(n_cases: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_cases):
        c_he = float(rng.choice([0.0, 10 ** rng.uniform(-5, -2)]))
        p = replace(
            MuCFParams(),
            phi=float(rng.uniform(0.05, 1.5)),
            c_t=float(rng.uniform(0.05, 0.9)),
            c_he=c_he,
            lam_dt=float(10 ** rng.uniform(7.5, 9.0)),
            kappa_q1s=float(rng.uniform(0.5, 8.0)),
            lam_dtmu=float(10 ** rng.uniform(7.5, 9.5)),
            omega_s0=float(rng.uniform(0.006, 0.010)),
            reactivation=float(rng.uniform(0.1, 0.6)),
        )
        ref = analyze(p)
        cf = cycle_closed_form(
            {"lam_dt": p.lam_dt, "kappa_q1s": p.kappa_q1s,
             "lam_dtmu": p.lam_dtmu, "omega_s": p.omega_s},
            {"phi": p.phi, "c_t": p.c_t, "c_he": p.c_he})
        for key_ref, key_cf in [("Xc", "Xc"), ("lam_c_eff", "lam_c"),
                                ("Y_dd", "Y_dd"), ("Y_tt", "Y_tt")]:
            rel = abs(cf[key_cf] - ref[key_ref]) / abs(ref[key_ref])
            worst = max(worst, rel)
            assert rel < 1e-8, (key_ref, ref[key_ref], cf[key_cf], p)
    return worst


if __name__ == "__main__":
    w = test_closed_form_matches_matrix_solution()
    print(f"OK: closed form matches the matrix solution in 300 cases (max relative error {w:.2e})")
