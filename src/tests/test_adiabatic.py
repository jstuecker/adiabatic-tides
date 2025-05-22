import numpy as np
import pytest
import matplotlib.pyplot as plt
import adiabatic_tides as at

from .test_config import embed_plot, check_max_relative_error, standard_profiles, plot_relative_error

@pytest.mark.parametrize("profile", ["nfw", "plummer", "powerlaw0.5", "powerlaw1.0", "powerlaw1.4", "powerlaw1.8", "aniso0.2pow1.5", "aniso-0.2pow1.0"])
def test_single_step_tidal_convergence(profile, embed_plot):
    # np.seterr(all='raise')
    prof = standard_profiles(profile)

    rt = 5
    tprof = at.profiles.RadialTidalProfile(-prof.accr(rt)/rt)
    prof_t = at.profiles.CompositeProfile(dm=prof, tide=tprof)

    rperi, rapo, rlmax, rtid, ramax_of_rp = at.numerics.interpolate.define_paspace_boundaries(prof_t.potential, prof_t.accr, prof_t.daccdr, nbins=10000, rmax=prof.rmax())

    table2 = at.numerics.interpolate.define_limited_peri_apo_table(ramax_of_rp, 1e-10, rlmax, nbins=100)
    def fbelow(rp, ra):
        return prof.f_of_el(*prof.e_l_of_rperi_rapo(rp, ra))

    f_of_rperi_rapo = at.numerics.interpolate.setup_adiabatic_f_of_rperi_rapo(prof.f_of_jl, prof_t.potential, table2, k=3, fpa_below=fbelow, accr=prof_t.accr, daccdr=prof_t.daccdr)

    if profile == "plummer":
        r = np.logspace(-3,3,100)
    else:
        r = np.logspace(-10,3,100)

    rhoref = at.numerics.integrate.integrate_f_paspace(f_of_rperi_rapo, prof_t.potential, prof_t.accr, r, N=80, rperirange=(0, rlmax), raporange=(0, ramax_of_rp))
    rho = at.numerics.integrate.integrate_f_paspace(f_of_rperi_rapo, prof_t.potential, prof_t.accr, r, rperirange=(0, rlmax), raporange=(0, ramax_of_rp))

    rhoscale = prof.density(r)
    embed_plot(plot_relative_error(rho, rhoref, 5e-3, fscale=rhoscale))
    check_max_relative_error(rho, rhoref, 5e-3, fscale=rhoscale)

@pytest.mark.slow
@pytest.mark.parametrize("profile", ["nfw", "powerlaw0.5", "powerlaw1.0", "powerlaw1.4", "powerlaw1.8", "aniso0.2pow1.5", "aniso-0.2pow1.0"])
def test_multi_step_tidal_convergence(profile, embed_plot):
    np.seterr(all="raise", under='ignore')

    prof = standard_profiles(profile)

    if profile == "plummer":
        rpmin = 1e-11
        tide = np.abs(prof.accr(1e1)/1e1)
        lower_boundary = "constant"
    else:
        rpmin = 1e-11
        tide = np.abs(prof.accr(1.))
        lower_boundary = prof.density, prof.m_of_r, prof.potential
    
    rev = np.logspace(-12,12,100)

    def phi0(r): return prof.potential(r, zero_at_zero=True)

    rho, m, phi = prof.density, prof.m_of_r, phi0
    rho_hr, m_hr, phi_hr = prof.density, prof.m_of_r, phi0

    for i in range(0,5):
        rnew, rhonew = at.adiabatic.adiabatic_tidal_iteration(prof.f_of_jl, rho, m, phi, tide=tide, rpmin=rpmin, fpa_below=prof.f_of_rperi_rapo, nr=200, nintegrate=32, ninterp=50, G=prof.G, rmax=prof.rmax())
        assert  np.min(rhonew) >= 0
        rho, m, phi = at.numerics.integrate.solve_poisson_via_spline_with_smart_boundaries(rnew, rhonew, lower_boundary=lower_boundary, upper_boundary="vacuum", G=prof.G)

        rnew_hr, rhonew_hr = at.adiabatic.adiabatic_tidal_iteration(prof.f_of_jl, rho_hr, m_hr, phi_hr, tide=tide, rpmin=rpmin, fpa_below=prof.f_of_rperi_rapo, nr=177, nintegrate=43, ninterp=77, G=prof.G, rmax=prof.rmax())
        assert  np.min(rnew_hr) >= 0
        rho_hr, m_hr, phi_hr = at.numerics.integrate.solve_poisson_via_spline_with_smart_boundaries(rnew_hr, rhonew_hr, lower_boundary=lower_boundary, upper_boundary="vacuum", G=prof.G)

        check_max_relative_error(rho(rev), rho_hr(rev), 1e-2, fscale=prof.density(rev))
    embed_plot(plot_relative_error(rho(rev), rho_hr(rev), 1e-2, fscale=prof.density(rev)))

@pytest.mark.slow
@pytest.mark.parametrize("profile", ["nfw", "powerlaw0.5", "powerlaw1.8", "aniso0.2pow1.5"])
def test_adiabatic_class(profile, embed_plot):
    prof = standard_profiles(profile)
    aprof = at.adiabatic.AdiabaticTidalTransformation.from_rtid(prof, 1.).run(5).assemble_total_profile()

    print(repr(aprof))

@pytest.mark.slow
def test_assembly_consistency():
    prof = standard_profiles("powerlaw1.5")
    lam = np.abs(prof.accr(1.)/1.)

    atr = at.adiabatic.AdiabaticTidalTransformation(prof, lam, verbose=1).run(eps = 0, nitermax=5)
    pi0 = atr.assemble_total_profile(0)
    pi2 = atr.assemble_total_profile(2)
    pi5 = atr.assemble_total_profile(-1)
    atr2 = at.adiabatic.AdiabaticTidalTransformation(pi2, lam, verbose=1).run(eps = 0, nitermax=3)
    pi5b = atr2.assemble_total_profile(-1)
    atr3 = at.adiabatic.AdiabaticTidalTransformation(pi0, lam, verbose=1).run(eps = 0, nitermax=5)
    pi5c = atr3.assemble_total_profile(-1)

    print(pi0.rtid(), pi0.m_of_r(pi0.rtid(), mode="self"))
    print(pi2.rtid(), pi2.m_of_r(pi2.rtid(), mode="self"))
    print(pi5.rtid(), pi5.m_of_r(pi5.rtid(), mode="self"))
    print(pi5b.rtid(), pi5b.m_of_r(pi5b.rtid(), mode="self"))
    print(pi5c.rtid(), pi5c.m_of_r(pi5c.rtid(), mode="self"))

    assert np.allclose(pi5.rtid(), pi5b.rtid(), rtol=1e-3)
    assert np.allclose(pi5.rtid(), pi5c.rtid(), rtol=1e-3)
    assert np.allclose(pi5.m_of_r(pi5.rtid(), mode="self"), pi5b.m_of_r(pi5b.rtid(), mode="self"), rtol=5e-3)
    assert np.allclose(pi5.m_of_r(pi5.rtid(), mode="self"), pi5c.m_of_r(pi5c.rtid(), mode="self"), rtol=1e-3)

@pytest.mark.slow
def test_double_adiabatic():
    prof = standard_profiles("nfw")

    # We choose a weak tide to get faster convergence here
    lam = np.abs(prof.accr(50.)/50.)

    prof1 = at.adiabatic.AdiabaticTidalTransformation(prof, lam).run(eps = 1e-3).assemble_total_profile()
    prof2 = at.adiabatic.AdiabaticTidalTransformation(prof, 2*lam).run(eps = 1e-3).assemble_total_profile()
    prof2b = at.adiabatic.AdiabaticTidalTransformation(prof1, 2*lam).run(eps = 1e-3).assemble_total_profile()

    print(prof1.rtid(), prof1.m_of_r(prof1.rtid(), mode="self"))
    print(prof2.rtid(), prof2.m_of_r(prof2.rtid(), mode="self"))
    print(prof2b.rtid(), prof2b.m_of_r(prof2b.rtid(), mode="self"))

    assert np.allclose(prof2.rtid(), prof2b.rtid(), rtol=1e-3)
    assert np.allclose(prof2.m_of_r(prof2.rtid(), mode="self"), prof2b.m_of_r(prof2b.rtid(), mode="self"), rtol=1e-3)

@pytest.mark.slow
def test_tide_reduction():
    prof = standard_profiles("nfw")
    lam = np.abs(prof.accr(100.)/100.) # we use a weak tide for cheap convergence

    p1 = at.adiabatic.AdiabaticTidalTransformation(prof, lam, verbose=1).run(eps = 0, nitermax=15).assemble_total_profile()
    # Since the scales are not quite optimal for the tide reduction,
    # it may be necessary to use slightly more interpolation points
    p1.cfg.adiabatic.ninterp = 80
    p2 = at.adiabatic.AdiabaticTidalTransformation(p1, lam*1e-1, verbose=1).run(eps = 0, nitermax=5).assemble_total_profile()

    # Since mass is conserved and not additional mass is lost 
    # it should be m1 ~ m2
    m1, m2 = p1.m_of_r(10*p1.rtid(), mode="self"), p2.m_of_r(10*p1.rtid(), mode="self")
    print(m1, p1.rtid())
    print(m2, p2.rtid())
    
    assert np.allclose(m1, m2, rtol=1e-2)