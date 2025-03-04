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

    # Define Initial profile phase space
    table = at.numerics.interpolate.define_peri_apo_table(1e-12, 1e12, nbins=100, facmax=1e14)
    rp_ra_of_jl = at.numerics.interpolate.setup_rperi_rapo_of_jl(prof.potential, table, nsteps_newton=5)

    def f_of_jl(j, l):
        assert np.all(~np.isnan(l))
        rperi,rapo = rp_ra_of_jl(j,l)

        return prof.f_of_el(*prof.E_L_of_rperi_rapo(rperi, rapo))

    rperi, rapo, rlmax, rtid, ramax_of_rp = at.numerics.interpolate.define_paspace_boundaries(prof_t.potential, prof_t.accr, prof_t.daccdr, nbins=10000)

    table2 = at.numerics.interpolate.define_limited_peri_apo_table(ramax_of_rp, 1e-10, rlmax, nbins=100)
    def fbelow(rp, ra):
        return prof.f_of_el(*prof.E_L_of_rperi_rapo(rp, ra))

    f_of_rperi_rapo = at.numerics.interpolate.setup_adiabatic_f_of_rperi_rapo(f_of_jl, prof_t.potential, table2, k=3, fpa_below=fbelow)

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

    table = at.numerics.interpolate.define_peri_apo_table(1e-12, 1e12, nbins=200)
    rp_ra_of_jl = at.numerics.interpolate.setup_rperi_rapo_of_jl(prof.potential, table)

    def f_of_jl(j, l):
        rperi,rapo = rp_ra_of_jl(j,l)
        return prof.f_of_el(*prof.E_L_of_rperi_rapo(rperi, rapo))
    
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
        rnew, rhonew = at.adiabatic.adiabatic_tidal_iteration(f_of_jl, rho, m, phi, tide=tide, rpmin=rpmin, fpa_below=prof.f_of_rperi_rapo, nr=200, nintegrate=32, ninterp=50)
        assert  np.min(rhonew) >= 0
        rho, m, phi = at.numerics.integrate.solve_poisson_via_spline_with_smart_boundaries(rnew, rhonew, lower_boundary=lower_boundary)

        rnew_hr, rhonew_hr = at.adiabatic.adiabatic_tidal_iteration(f_of_jl, rho_hr, m_hr, phi_hr, tide=tide, rpmin=rpmin, fpa_below=prof.f_of_rperi_rapo, nr=177, nintegrate=43, ninterp=77)
        assert  np.min(rnew_hr) >= 0
        rho_hr, m_hr, phi_hr = at.numerics.integrate.solve_poisson_via_spline_with_smart_boundaries(rnew_hr, rhonew_hr, lower_boundary=lower_boundary)

        check_max_relative_error(rho(rev), rho_hr(rev), 1e-2, fscale=prof.density(rev))
    embed_plot(plot_relative_error(rho(rev), rho_hr(rev), 1e-2, fscale=prof.density(rev)))

@pytest.mark.slow
@pytest.mark.parametrize("profile", ["nfw", "powerlaw0.5", "powerlaw1.8", "aniso0.2pow1.5"])
def test_adiabatic_class(profile, embed_plot):
    prof = standard_profiles(profile)
    aprof = at.adiabatic.AdiabaticTidalTransformation.from_rtid(prof, 1.).run(5).assemble_total_profile()

    print(repr(aprof))