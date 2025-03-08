import numpy as np
import pytest
import matplotlib.pyplot as plt
import adiabatic_tides as at

from .test_config import embed_plot, check_max_relative_error, standard_profiles, plot_relative_error

@pytest.mark.parametrize("profile", ["nfw", "plummer", "powerlaw0.5", "powerlaw1.0", "powerlaw1.4"])
def test_peri_apo_finding(profile, embed_plot):
    np.random.seed(42)
    prof = standard_profiles(profile)
    
    rp = np.logspace(-10, 6, 1333)
    ra = rp * 10.**np.random.uniform(0,5,len(rp))
    r = rp * (ra/rp)**np.random.uniform(0,1,len(rp))
    
    e,l = prof.E_L_of_rperi_rapo(rp, ra)
    rp2, ra2, = prof.rperi_rapo_of_r_e_l(r, e, l)

    # embed_plot(plot_relative_error(rp, rp2, 1e-5))
    # embed_plot(plot_relative_error(ra, ra2, 1e-5))

    check_max_relative_error(rp, rp2, 1e-4)
    check_max_relative_error(ra, ra2, 1e-4)

@pytest.mark.parametrize("profile", ["nfw", "plummer", "powerlaw0.5", "powerlaw1.0", "powerlaw1.4"])
def test_actions_rel(profile, embed_plot):
    np.random.seed(42)
    prof = standard_profiles(profile)
    
    rp = np.logspace(-10, 6, 1333)
    ra = rp * 10.**np.random.uniform(0,5,len(rp))
    r = rp * (ra/rp)**np.random.uniform(0,1,len(rp))
    
    e,l = prof.E_L_of_rperi_rapo(rp, ra)
    j1 = prof.radial_action_of_r_e_l(r, e, l)
    j2 = prof.radial_action_of_rp_ra(rp, ra)

    # embed_plot(plot_relative_error(j1, j2, 1e-5))

    check_max_relative_error(j1, j2, 1e-4)

@pytest.mark.parametrize("profile", ["nfw", "plummer", "powerlaw0.5", "powerlaw1.0", "powerlaw1.4", "powerlaw1.8", "aniso0.2pow1.5", "aniso-0.2pow1.0"])
def test_actions_convergence(profile, embed_plot):
    np.seterr(all='raise')
    prof = standard_profiles(profile)

    np.random.seed(42)
    rptest = np.logspace(-4, 4, 10000)
    ratest = rptest * (1 + 10**np.random.uniform(-1,5,len(rptest)))

    jhr = at.numerics.integrate.calculate_radial_action_tanh_peri_apo(prof.potential, rptest, ratest, nintegrate=100, accr=prof.accr, daccdr=prof.daccdr)
    jlr = at.numerics.integrate.calculate_radial_action_tanh_peri_apo(prof.potential, rptest, ratest, accr=prof.accr, daccdr=prof.daccdr)

    sel = jhr > 0
    
    embed_plot(plot_relative_error(jlr[sel], jhr[sel], 1e-3))
    check_max_relative_error(jlr[sel], jhr[sel], 1e-3)
    
    ri = np.logspace(-10, 10, 4000)
    nprof = at.profiles.NumericalProfile(ri, prof.density(ri))

    jlr = at.numerics.integrate.calculate_radial_action_tanh_peri_apo(nprof.potential, rptest, ratest, accr=prof.accr, daccdr=prof.daccdr)
    embed_plot(plot_relative_error(jlr[sel], jhr[sel], 1e-2))
    check_max_relative_error(jlr[sel], jhr[sel], 1e-2)

@pytest.mark.parametrize("profile", ["nfw", "plummer", "powerlaw0.5", "powerlaw1.0", "powerlaw1.4", "powerlaw1.8", "aniso0.2pow1.5", "aniso-0.2pow1.0"])
def test_action_inversion(profile, embed_plot):
    np.seterr(all='raise', under='ignore')
    prof = standard_profiles(profile)

    np.random.seed(42)
    import time

    # Setup interpolator
    t0 = time.time()
    table = at.numerics.interpolate.define_peri_apo_table(1e-10, 1e10, nbins=100)
    # rp_ra_of_jl = at.numerics.interpolate.setup_rperi_rapo_of_jl(prof.potential, table)
    rp_ra_of_jl = at.numerics.interpolate.setup_rperi_rapo_of_jl_new(prof.potential, table, nsteps_newton=1, accr=prof.accr, daccdr=prof.daccdr)
    print(f"Setup time {time.time()-t0:.2f}s")

    # Test against actual values
    rptest = np.logspace(-5,5,4312)
    for logramin, tolerance in ((-3,1e-4), (-5,1e-2)):
        print("Test tolerance: %.1e" % tolerance)
        ratest = rptest * (1 + 10**np.random.uniform(logramin,5,len(rptest)))
        j = at.numerics.integrate.calculate_radial_action_tanh_peri_apo(prof.potential, rptest, ratest, accr=prof.accr, daccdr=prof.daccdr)
        l = np.sqrt(2.*(prof.potential(ratest) - prof.potential(rptest))/(rptest**-2 - ratest**-2))

        rpn, ran = rp_ra_of_jl(j,l)

        print("nans:", np.sum(np.isnan(rpn)), np.sum(np.isnan(ran)))

        if logramin > -4:
            embed_plot(plot_relative_error(rpn, rptest, tolerance))
            embed_plot(plot_relative_error(ran, ratest, tolerance))

            check_max_relative_error(rpn, rptest, tolerance)
            check_max_relative_error(ran, ratest, tolerance)

            # Check energies
            En, Ln = prof.E_L_of_rperi_rapo(rpn, ran)
            E, L = prof.E_L_of_rperi_rapo(rptest, ratest)

            embed_plot(plot_relative_error(En, E, tolerance))
            embed_plot(plot_relative_error(Ln, L, tolerance))

            check_max_relative_error(En, E, tolerance)
            check_max_relative_error(Ln, L, tolerance)

        # For very circular orbits we only care to reconstruc the phase space distr.
        # properly:
        f1, f2 = prof.f_of_el(E,L), prof.f_of_el(En,Ln)
        embed_plot(plot_relative_error(f1[f2>0], f2[f2>0], tolerance))

        check_max_relative_error(f1[f2>0], f2[f2>0], tolerance)


@pytest.mark.parametrize("profile", ["nfw", "plummer", "powerlaw0.5", "powerlaw1.0", "powerlaw1.4", "powerlaw1.8", "aniso0.2pow1.5", "aniso-0.2pow1.0"])
def test_f_jl_reconstruction(profile, embed_plot):
    np.random.seed(42)
    np.seterr(all='raise', under="ignore")
    prof = standard_profiles(profile)

    # table = at.numerics.interpolate.define_peri_apo_table(1e-10, 1e10, nbins=100)
    # rp_ra_of_jl = at.numerics.interpolate.setup_rperi_rapo_of_jl_new(prof.potential, table, nsteps_newton=1, accr=prof.accr)
    # def f_of_jl(j, l): 
    #     rp,ra = rp_ra_of_jl(j,l)
    #     return prof.f_of_rperi_rapo(rp,ra)

    table2 = at.numerics.interpolate.define_peri_apo_table(1e-9, 1e9, nbins=133, facmax=1e8)
    f_of_rperi_rapo = at.numerics.interpolate.setup_adiabatic_f_of_rperi_rapo(prof.f_of_jl, prof.potential, table2, accr=prof.accr, daccdr=prof.daccdr)

    # Test
    rp = np.logspace(-5,5,4000)
    for logramin, tolerance in ((-1,1e-3), (-3,0.5)):
        ra = rp * (1. + 10**np.random.uniform(logramin,5,len(rp)))
        

        f = f_of_rperi_rapo(rp, ra)
        fref = prof.f_of_el(*prof.E_L_of_rperi_rapo(rp, ra))

        embed_plot(plot_relative_error(f, fref, tolerance))

        check_max_relative_error(f, fref, tolerance)

@pytest.mark.parametrize("profile", ["plummer", "powerlaw0.5", "powerlaw1.0", "powerlaw1.4", "powerlaw1.8", "aniso0.2pow1.5", "aniso-0.2pow1.0"])
def test_rho_reconstruction(profile, embed_plot):
    np.random.seed(42)
    np.seterr(all='raise')
    prof = standard_profiles(profile)

    table2 = at.numerics.interpolate.define_peri_apo_table(1e-9, 1e9, nbins=133, facmax=1e8)
    f_of_rperi_rapo = at.numerics.interpolate.setup_adiabatic_f_of_rperi_rapo(prof.f_of_jl, prof.potential, table2, accr=prof.accr, daccdr=prof.daccdr)

    if profile == "plummer":
        r = np.logspace(-0.8,6,200)
    else:
        r = np.logspace(-7,7,200)
    rho = at.numerics.integrate.integrate_f_paspace(f_of_rperi_rapo, prof.potential, prof.accr, r)

    embed_plot(plot_relative_error(rho, prof.density(r), 1e-3))

    check_max_relative_error(rho, prof.density(r), 1e-3)

#"plummer", 
@pytest.mark.parametrize("profile", ["nfw", "powerlaw0.5", "powerlaw1.0", "powerlaw1.4", "powerlaw1.8", "aniso0.2pow1.5", "aniso-0.2pow1.0"])
def test_numerical_rho_reconstruction(profile, embed_plot):
    np.random.seed(42)
    np.seterr(divide='raise', invalid="raise", over="raise", under="ignore")
    prof = standard_profiles(profile)
    r = np.logspace(-16, 16, 3000)
    prof = at.profiles.NumericalProfile(r, prof.density(r))

    table = at.numerics.interpolate.define_peri_apo_table(1e-12, 1e12, nbins=400, facmax=1e11)
    rp_ra_of_jl = at.numerics.interpolate.setup_rperi_rapo_of_jl(prof.potential, table, accr=prof.accr, daccdr=prof.daccdr)
    # rp_ra_of_jl = at.numerics.interpolate.setup_rperi_rapo_of_jl_new(prof.potential, table, accr=prof.accr, nsteps_newton=1)

    def f_of_jl(j, l):
        rperi,rapo = rp_ra_of_jl(j,l)

        f = np.zeros(np.broadcast(j,l).shape)
        valid = ~np.isnan(rperi) & ~np.isnan(rapo)
        f[valid] = prof.f_of_el(*prof.E_L_of_rperi_rapo(rperi[valid], rapo[valid]))

        return f

    table2 = at.numerics.interpolate.define_peri_apo_table(1e-9, 1e9, nbins=140, facmax=1e8)
    def potential(r):
        return prof.potential(r)

    f_of_rperi_rapo = at.numerics.interpolate.setup_adiabatic_f_of_rperi_rapo(f_of_jl, prof.potential, table2, accr=prof.accr, daccdr=prof.daccdr)

    if profile == "plummer":
        r = np.logspace(-0.8,6,200)
    else:
        r = np.logspace(-7,7,200)
    rho = at.numerics.integrate.integrate_f_paspace(f_of_rperi_rapo, potential, prof.accr, r)

    embed_plot(plot_relative_error(rho, prof.density(r), 1e-2))

    check_max_relative_error(rho, prof.density(r), 1e-2)
