"""
mucf_blanket_openmc.py — Neutron transport model of a μCF neutron source + molten-salt blanket (layer 2)

Geometry (concentric spheres, a quasi-1D scoping model):
    r <  10 cm : μCF target (liquid D-T), 14.06 MeV isotropic point source at the center
    10 – 12 cm : vessel wall (Hastelloy-N equivalent)
    12 – 92 cm : molten-salt blanket (LiF-BeF2-ThF4(-UF4))
    92 –122 cm : graphite reflector
    > 122 cm   : vacuum (leakage)

Quantities computed (all per source neutron = per dt fusion in the μCF target):
    TBR         : tritium production (Li-6(n,t), Li-7(n,n't))
    U233 bred   : number of Th-232(n,γ) reactions (→ Pa-233 → U-233)
    F           : number of fissions,  E_heat : total heating [MeV]
    M           : energy multiplication = E_heat / 14.06 MeV
    k_s         : source multiplication factor = νF / (1 + νF) (passed to the Q_eng formula of layer 3)
    k_eff       : effective multiplication factor from an eigenvalue run (subcriticality check)

Requirements:
    - OpenMC (>= 0.14)              conda install -c conda-forge openmc
    - nuclear data (HDF5, free)     https://openmc.org/official-data-libraries/
      e.g. unpack ENDF/B-VIII.0 and point OPENMC_CROSS_SECTIONS to its cross_sections.xml
    - a depletion chain file for burnup calculations (distributed on the same page)

Note: this script was written in an environment where OpenMC could not be installed, so it has
      not been executed. The API targets the OpenMC 0.14–0.15 series.
"""

from __future__ import annotations

import argparse

import openmc

# ---------------------------------------------------------------------------
# 1. Materials
#    [public] = value available in the public literature, [assumed] = placeholder value
# ---------------------------------------------------------------------------

# (element or nuclide, number of atoms) for each compound
COMPOUNDS = {
    "LiF": [("Li", 1), ("F", 1)],
    "BeF2": [("Be", 1), ("F", 2)],
    "ThF4": [("Th", 1), ("F", 4)],
    "UF4": [("U233", 1), ("F", 4)],  # the fissile component is U-233 directly
}


def salt(name: str, mol_frac: dict[str, float], density: float,
         li6_enrichment: float = 7.59) -> openmc.Material:
    """Build a molten-salt material from mole fractions. li6_enrichment is Li-6 in atom% (natural: 7.59%)."""
    atoms: dict[str, float] = {}
    for comp, x in mol_frac.items():
        for sym, n in COMPOUNDS[comp]:
            atoms[sym] = atoms.get(sym, 0.0) + x * n

    m = openmc.Material(name=name)
    for sym, n in atoms.items():
        if sym == "Li":
            m.add_element("Li", n, percent_type="ao",
                          enrichment=li6_enrichment, enrichment_target="Li6",
                          enrichment_type="ao")
        elif sym[-1].isdigit():
            m.add_nuclide(sym, n, percent_type="ao")
        else:
            m.add_element(sym, n, percent_type="ao")
    m.set_density("g/cm3", density)
    return m


def flibe_density(T_kelvin: float) -> float:
    """[public] FLiBe (2LiF-BeF2) density correlation ρ = 2.413 - 4.884e-4 T [g/cm3] (Janz correlation,
    compiled in the property review by Romatoski & Hu, Ann. Nucl. Energy 109 (2017) 635)."""
    return 2.413 - 4.884e-4 * T_kelvin


def make_materials(u233_mol: float, li6_enrichment: float) -> dict[str, openmc.Material]:
    # μCF target: liquid D-T (D:T = 0.6:0.4, about LHD) [assumed]
    dt = openmc.Material(name="DT target")
    dt.add_nuclide("H2", 0.6)
    dt.add_nuclide("H3", 0.4)
    dt.set_density("g/cm3", 0.20)

    # Vessel wall: Hastelloy-N equivalent [public: representative ORNL alloy composition, wt%, simplified]
    hast = openmc.Material(name="Hastelloy-N")
    for el, w in [("Ni", 71.0), ("Mo", 16.0), ("Cr", 7.0), ("Fe", 5.0), ("Si", 0.5), ("Mn", 0.5)]:
        hast.add_element(el, w, percent_type="wo")
    hast.set_density("g/cm3", 8.86)

    # Blanket salt:
    # [public] blanket salt of the MSBR two-fluid design is LiF-BeF2-ThF4 ≈ 71-2-27 mol%
    #          (design value from ORNL-4541, 1971; check the number against the original)
    # Here UF4 (U-233) replaces u233_mol of the ThF4 to tune the subcriticality
    th = 0.27 - u233_mol
    blanket = salt("blanket salt",
                   {"LiF": 0.71, "BeF2": 0.02, "ThF4": th, "UF4": u233_mol},
                   density=4.3,  # [assumed] rough value for a Th-rich salt; should be computed from composition
                   li6_enrichment=li6_enrichment)
    blanket.depletable = True
    blanket.volume = 4.0 / 3.0 * 3.14159265 * (92.0**3 - 12.0**3)

    graphite = openmc.Material(name="graphite")
    graphite.add_element("C", 1.0)
    graphite.set_density("g/cm3", 1.85)
    graphite.add_s_alpha_beta("c_Graphite")

    return {"dt": dt, "wall": hast, "blanket": blanket, "graphite": graphite}


# ---------------------------------------------------------------------------
# 2. Geometry
# ---------------------------------------------------------------------------

R = {"target": 10.0, "wall": 12.0, "blanket": 92.0, "reflector": 122.0}


def make_geometry(mats):
    s = {k: openmc.Sphere(r=v) for k, v in R.items()}
    s["reflector"].boundary_type = "vacuum"
    cells = {
        "target": openmc.Cell(name="target", fill=mats["dt"], region=-s["target"]),
        "wall": openmc.Cell(name="wall", fill=mats["wall"], region=+s["target"] & -s["wall"]),
        "blanket": openmc.Cell(name="blanket", fill=mats["blanket"], region=+s["wall"] & -s["blanket"]),
        "reflector": openmc.Cell(name="reflector", fill=mats["graphite"],
                                 region=+s["blanket"] & -s["reflector"]),
    }
    return openmc.Geometry(list(cells.values())), cells, s


# ---------------------------------------------------------------------------
# 3. Fixed-source calculation (driven by the μCF neutron source)
# ---------------------------------------------------------------------------

E_DT = 14.06e6  # eV


def build_fixed_source_model(u233_mol=0.0, li6_enrichment=7.59, particles=20000, batches=20):
    mats = make_materials(u233_mol, li6_enrichment)
    geom, cells, surfs = make_geometry(mats)

    src = openmc.IndependentSource()
    src.space = openmc.stats.Point((0.0, 0.0, 0.0))
    src.angle = openmc.stats.Isotropic()
    # μCF is a cold intramolecular reaction, so there is almost no thermal broadening (unlike plasma fusion)
    src.energy = openmc.stats.Discrete([E_DT], [1.0])

    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.source = src
    settings.particles = particles
    settings.batches = batches
    settings.photon_transport = False  # photons deposited locally (uses heating-local)

    tallies = openmc.Tallies()

    t_bl = openmc.Tally(name="blanket")
    t_bl.filters = [openmc.CellFilter([cells["blanket"]])]
    t_bl.scores = ["H3-production", "fission", "nu-fission", "(n,2n)", "absorption"]
    tallies.append(t_bl)

    t_th = openmc.Tally(name="Th232 capture")
    t_th.filters = [openmc.CellFilter([cells["blanket"]])]
    t_th.nuclides = ["Th232"]
    t_th.scores = ["(n,gamma)", "fission"]
    tallies.append(t_th)

    t_heat = openmc.Tally(name="heating")
    t_heat.filters = [openmc.CellFilter(list(cells.values()))]
    t_heat.scores = ["heating-local"]
    tallies.append(t_heat)

    t_leak = openmc.Tally(name="leakage")
    t_leak.filters = [openmc.SurfaceFilter([surfs["reflector"]])]
    t_leak.scores = ["current"]
    tallies.append(t_leak)

    # Neutron spectrum in the blanket (to see how the 14 MeV component is moderated)
    t_spec = openmc.Tally(name="spectrum")
    t_spec.filters = [openmc.CellFilter([cells["blanket"]]),
                      openmc.EnergyFilter.from_group_structure("CCFE-709")]
    t_spec.scores = ["flux"]
    tallies.append(t_spec)

    return openmc.Model(geometry=geom, materials=openmc.Materials(mats.values()),
                        settings=settings, tallies=tallies)


def analyze_fixed_source(statepoint_path: str) -> dict:
    with openmc.StatePoint(statepoint_path) as sp:
        bl = sp.get_tally(name="blanket")
        val = lambda t, s: float(t.get_values(scores=[s]).sum())
        tbr = val(bl, "H3-production")
        fis = val(bl, "fission")
        nufis = val(bl, "nu-fission")
        n2n = val(bl, "(n,2n)")

        th = sp.get_tally(name="Th232 capture")
        u233_breed = float(th.get_values(scores=["(n,gamma)"]).sum())

        heat_eV = float(sp.get_tally(name="heating").get_values(scores=["heating-local"]).sum())
        leak = float(sp.get_tally(name="leakage").get_values(scores=["current"]).sum())

    return {
        "TBR": tbr,
        "U233_breeding_per_source_n": u233_breed,
        "fissions_per_source_n": fis,
        "n2n_per_source_n": n2n,
        "heat_MeV_per_source_n": heat_eV / 1e6,
        "M_energy": heat_eV / E_DT,                   # multiplication relative to the neutron energy
        "k_s": nufis / (1.0 + nufis) if nufis > 0 else 0.0,
        "leakage": leak,
    }


# ---------------------------------------------------------------------------
# 4. Eigenvalue calculation (subcriticality k_eff without the external source)
# ---------------------------------------------------------------------------

def build_eigenvalue_model(u233_mol, li6_enrichment=7.59, particles=20000, batches=60, inactive=20):
    model = build_fixed_source_model(u233_mol, li6_enrichment, particles, batches)
    model.tallies = openmc.Tallies()
    s = model.settings
    s.run_mode = "eigenvalue"
    s.inactive = inactive
    # Initial source distributed uniformly over the blanket region
    s.source = openmc.IndependentSource(
        space=openmc.stats.SphericalIndependent(
            r=openmc.stats.PowerLaw(R["wall"], R["blanket"], 2),
            cos_theta=openmc.stats.Uniform(-1.0, 1.0),
            phi=openmc.stats.Uniform(0.0, 2 * 3.14159265)),
        energy=openmc.stats.Watt(0.988e6, 2.249e-6),
    )
    return model


# ---------------------------------------------------------------------------
# 5. (Optional) Depletion: U-233 build-up and decay heat
#    Source strength S [n/s] = (muon supply rate [1/s]) × Xc (result of layer 1)
# ---------------------------------------------------------------------------

def run_depletion(model, chain_file, source_rate_n_per_s, days=(30,) * 12):
    import openmc.deplete as dep
    op = dep.CoupledOperator(model, chain_file, normalization_mode="source-rate")
    integ = dep.PredictorIntegrator(op, list(days), source_rates=source_rate_n_per_s,
                                    timestep_units="d")
    integ.integrate()
    # Results are written to depletion_results.h5. Time evolution of U233 atoms:
    #   res = dep.Results("depletion_results.h5"); res.get_atoms(mat, "U233")
    # Decay heat can be evaluated from the depleted material with Material.get_decay_heat() (version dependent)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fixed", "eigen", "scan"], default="fixed")
    ap.add_argument("--u233", type=float, default=0.0, help="mole fraction of UF4 (U-233)")
    ap.add_argument("--li6", type=float, default=7.59, help="Li-6 enrichment [atom%%]")
    ap.add_argument("--particles", type=int, default=20000)
    args = ap.parse_args()

    if args.mode == "fixed":
        sp = build_fixed_source_model(args.u233, args.li6, args.particles).run()
        for k, v in analyze_fixed_source(sp).items():
            print(f"{k:28s} {v:.4f}")

    elif args.mode == "eigen":
        sp = build_eigenvalue_model(args.u233, args.li6, args.particles).run()
        with openmc.StatePoint(sp) as s:
            print("k_eff =", s.keff)

    else:  # scan the U-233 concentration to relate k_eff to the multiplication
        print(" u233_mol  k_eff        k_s     M_energy  TBR    U233_breed")
        for x in [0.0, 0.002, 0.004, 0.006, 0.008, 0.010]:
            ke = build_eigenvalue_model(x, args.li6, args.particles).run(cwd=f"eig_{x}")
            with openmc.StatePoint(ke) as s:
                keff = s.keff
            fs = build_fixed_source_model(x, args.li6, args.particles).run(cwd=f"fix_{x}")
            r = analyze_fixed_source(fs)
            print(f" {x:.3f}    {keff.n:.4f}±{keff.s:.4f}  {r['k_s']:.3f}  {r['M_energy']:8.2f}"
                  f"  {r['TBR']:.3f}  {r['U233_breeding_per_source_n']:.3f}")
