import adiabatic_tides as at
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

from .test_config import check_perc_relative_error, check_max_relative_error, standard_profiles, embed_plot, plot_relative_error

@pytest.mark.parametrize("profile", ["nfw", "plummer", "powerlaw0.5", "powerlaw1.0", "powerlaw1.4", "powerlaw1.8"])
def test_sample_radii_fixed(profile, embed_plot):
    np.random.seed(42)

    prof = standard_profiles(profile)
    ri = np.logspace(-10, 10, 1000)

    rmax = 1.
    
    rs = at.numerics.sample.sample_radii(ri, prof.m_of_r(ri), 1000000, rmax=rmax)
    ms = np.ones_like(rs) * prof.m_of_r(rmax) / len(rs)
    pprof = at.profiles.ParticleProfile((rs, ms), rbins=np.logspace(-3,3,100))

    # uniform mass sampling makes it hard to get the density right at small radii
    ritest = np.logspace(-1.,-0.1, 100)
    m, mref = pprof.m_of_r(ritest), prof.m_of_r(ritest)

    embed_plot(plot_relative_error(m, mref, 1e-1))
    check_max_relative_error(m, mref, 1e-1)

@pytest.mark.parametrize("profile", ["nfw", "plummer", "powerlaw0.5", "powerlaw1.0", "powerlaw1.4", "powerlaw1.8"])
def test_sample_radii_adaptive(profile, embed_plot):
    np.random.seed(42)

    prof = standard_profiles(profile)

    ri = np.logspace(-10, 10, 1000)

    # These are the optimal weights for the sampling
    # We have a lot more (lower mass) particles at small radii
    weights = 1. / (prof.density(ri)*ri**2 * ri) 
    rs,ms = at.numerics.sample.sample_rimi_from_density(ri, prof.density(ri), 1000000, weights=weights )
    pprof = at.profiles.ParticleProfile((rs, ms), rbins=np.logspace(-10,10,1000))

    if profile == "plummer":
        # At large radii the histogram needed to calculate the
        # density profile gets inaccurate that is because of 
        # cancellation errors
        ritest = np.logspace(-9,5, 100)
    else: 
        ritest = np.logspace(-9,9, 100)
    rho, rhoref = pprof.density(ritest), prof.density(ritest)

    embed_plot(plot_relative_error(rho, rhoref, 1e-1))
    check_max_relative_error(rho, rhoref, 1e-1)

# @pytest.mark.veryslow
# @pytest.mark.parametrize("profile", ["nfw", "powerlaw1.0", "powerlaw1.4", "powerlaw1.8"])
# def test_sample_energy_distr(profile, embed_plot):
#     np.random.seed(42)

#     prof0 = standard_profiles(profile)

#     ri = np.logspace(-10, 10, 1000)
#     prof = at.profiles.NumericalProfile(ri, prof0.density(ri))

#     weights = 1. / (prof.density(ri)*ri**2 * ri)
#     rs,ms = at.numerics.sampling.sample_rimi_from_density(ri, prof.density(ri), 200000, weights=weights)
#     Es = at.numerics.sampling.sample_conditional_energy_adaptive_batched(prof.potential(rs), prof.f_of_e, nintegrate=400)

#     n_of_e,ebins  = np.histogram(Es, weights=ms, bins=prof.potential(np.logspace(-5,5,100)))
#     n_of_e = n_of_e / (ebins[1:] - ebins[:-1])
#     ei = np.sqrt(ebins[1:] * ebins[:-1])

#     nref = prof.n_of_e(ei)

#     embed_plot(plot_relative_error(n_of_e, nref, 0.2))
#     check_perc_relative_error(n_of_e, nref, 0.2, 80)


@pytest.mark.slow
@pytest.mark.parametrize("profile", ["nfw", "powerlaw1.0", "powerlaw1.4", "powerlaw1.8"])
def test_sample_and_integrate(profile, embed_plot):
    np.random.seed(55)
    np.seterr(all="warn")

    prof = standard_profiles(profile)

    rs,rps,Ls,vrs,ms,ri,rhoi = prof.sample_particles(10000, mode="r_rp_l_vr_m_rrho_rho", rpmin=1, rpmax=50., rmax=5e2)

    rbins = np.logspace(-0.2, 2, 40)
    rtest = np.sqrt(rbins[1:] * rbins[:-1])

    # First integrate a bit to make sure we are uncorrelated with initial position
    tmax = prof.tcirc(rps)
    # rs, vrs = at.numerics.sampling.integrate_radial_orbits(prof.accr, rs, vrs, Ls, tmax, nsteps=1000)
    # rs = np.abs(rs)

    # Now integrate and measure time-averaged mass profile
    # (The time-averaging helps enhance the statistics at same costs)
    rho, n = 0, 0
    for rs, vrs in at.numerics.sample.integrate_radial_orbits_with_snaps(prof.accr, rs, vrs, Ls, tmax, nsnaps=100, nsteps_per_snap=10):
        rhoinew, minew = at.numerics.sample.get_mass_profile(rs, ms, rbins)
        rho += rhoinew
        n += 1
    rho /= n

    rhoref = np.interp(rtest, ri, rhoi)
    sel = rhoref > 0.
    embed_plot(plot_relative_error(rho[sel], rhoref[sel], 0.4))
    check_perc_relative_error(rho[sel], rhoref[sel], 0.4, 80)

@pytest.mark.parametrize("profile", ["nfw", "powerlaw1.0", "powerlaw1.4", "powerlaw1.8", "aniso-0.3pow1", "aniso0pow1", "aniso0.3pow1"])
def test_ks_statistic_convergence(profile):
    np.random.seed(42)
    prof = standard_profiles(profile)

    nsamp = 5000
    pref = prof.sample_particles(nsamp, rpmin=0.1, rpmax=1., rmax=1e4, nsteps_metropolis=256)

    def compare_particles(p1, p2, test_label="", minp=0.):
        res = []
        for i, lab in enumerate(["r", "E", "L", "vr"]):
            ks_stat, p_value = ks_2samp(p1[i], p2[i])
            print(f"{test_label}{lab}: KS Statistic: {ks_stat:.2g}, p-value: {p_value:.2g}")
            
            assert p_value > minp
        print("---")

    for nsteps_chain in 64,32,16,8:
        p = prof.sample_particles(nsamp, rpmin=0.1, rpmax=1., rmax=1e4, nsteps_metropolis=nsteps_chain)
        compare_particles(p, pref, test_label="Nchain=%d: " % nsteps_chain, minp = 0.01 if nsteps_chain >= 64 else 0)