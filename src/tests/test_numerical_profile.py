import adiabatic_tides as at
import numpy as np
import pytest

#from .test_config import check_max_relative_error, standard_profiles
from . import test_config as tc
from .test_config import embed_plot

def compare_profiles(pnum, pref, rtest, rtol=1e-3):
    tc.check_max_relative_error(pnum.density(rtest), pref.density(rtest), rtol)
    tc.check_max_relative_error(pnum.m_of_r(rtest), pref.m_of_r(rtest), rtol)
    tc.check_max_relative_error(pnum.accr(rtest), pref.accr(rtest), rtol)
    assert np.allclose(pnum.potential(rtest) - pnum.potential(rtest[0]), pref.potential(rtest) - pref.potential(rtest[0]), rtol=rtol)

def plot_profile_err(pnum, pref, rtest, rtol=1e-3):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2,2, figsize=(10,10))

    def plot_difference(ax, f1, f2):
        ax.axhline(rtol, color='black', ls = "dashed")
        ax.loglog(rtest, np.abs(f1/f2 - 1.))
    
    plot_difference(axs[0,0], pnum.density(rtest), pref.density(rtest))
    plot_difference(axs[0,1], pnum.m_of_r(rtest), pref.m_of_r(rtest))
    plot_difference(axs[1,0], pnum.accr(rtest), pref.accr(rtest))
    plot_difference(axs[1,1], pnum.potential(rtest), pref.potential(rtest, zero_at_zero=True))

    return fig

@pytest.mark.parametrize("profile", ["nfw", "plummer", "powerlaw0.5", "powerlaw1.0", "powerlaw1.4", "powerlaw1.8"])
def test_poisson(profile, embed_plot):
    np.seterr(all='raise')

    for n, tolerance in ((200,1e-2),(400,2e-3)):
        prof = tc.standard_profiles(profile)

        rsetup = np.logspace(-10, 10, n)
        rtest = np.logspace(-8, 8, 231)

        nprof = at.profiles.NumericalProfile(rsetup, prof.density(rsetup))

        embed_plot(plot_profile_err(nprof, prof, rtest, tolerance))

        compare_profiles(nprof, prof, rtest, tolerance)

# @pytest.mark.parametrize("profile", ["plummer"])
# def test_cored_poisson(profile, embed_plot):
#     np.seterr(all='raise')
#     tolerance = 1e-2

#     prof = tc.standard_profiles(profile)

#     rsetup = np.linspace(1e-10, 5, 1000)
#     rtest = np.linspace(1e-2, 2, 73)
    
#     nprof = at.profiles.NumericalProfile(rsetup, prof.density(rsetup))

#     embed_plot(plot_profile_err(nprof, prof, rtest, tolerance))
#     compare_profiles(nprof, prof, rtest, tolerance)

@pytest.mark.parametrize("profile", ["nfw", "plummer", "powerlaw0.5", "powerlaw1.0", "powerlaw1.4", "powerlaw1.8"])
def test_boundaries(profile, embed_plot):
    np.seterr(divide='raise', invalid="raise", over="raise", under="ignore")

    prof = tc.standard_profiles(profile)

    rsetup = np.logspace(-10, 5, 6000)
    rtest = np.logspace(-8, 8, 231)

    rtid0 = 1e2
    tide = np.abs(prof.accr(rtid0)/rtid0)
    tprof = at.profiles.RadialTidalProfile(tide)
    
    nprof = at.profiles.NumericalProfile(rsetup, prof.density(rsetup))
    
    prof_t = at.profiles.CompositeProfile(prof, tprof)
    nprof_t = at.profiles.CompositeProfile(nprof, tprof)

    for p in (prof, nprof):
        rlmax, rtid = at.mathtools.find_rlmax(p.accr, p.daccdr), at.mathtools.find_rphimax(p.accr)
        print("Without tide: boundary: %.5e %.5e" % (rlmax, rtid))
        assert (rlmax == np.infty) and (rtid == np.infty)
    print("With Tide:")
    for p in (prof_t, nprof_t):
        rlmax, rtid = at.mathtools.find_rlmax(p.accr, p.daccdr), at.mathtools.find_rphimax(p.accr)
        print("With tide: boundary: %.5e %.5e" % (rlmax, rtid))
        assert np.abs(rtid/rtid0 - 1.) < 1e-3

        # Check that rlmax corresponds to the maximum of the circular angular momentum
        assert np.all(p.vcirc(rtest)*rtest <= p.vcirc(rlmax)*rlmax)

        # Check that rtid corresponds to the maximum of the potential
        assert np.all(p.potential(rtest) <= p.potential(rtid))

    rperi, rapo, rlmax, rtid, ramax_of_rp = at.mathtools.define_paspace_boundaries(prof_t.potential, prof_t.accr, prof_t.daccdr, nbins=555)
    nrperi, nrapo, nrlmax, nrtid, nramax_of_rp = at.mathtools.define_paspace_boundaries(nprof_t.potential, nprof_t.accr, nprof_t.daccdr, nbins=555)

    rtest = np.logspace(-8, -0.1, 33) * rlmax
    tc.check_max_relative_error(ramax_of_rp(rtest), nramax_of_rp(rtest), 1e-2)

# add back "plummer" later
@pytest.mark.parametrize("profile", ["nfw", "powerlaw0.5", "powerlaw1.0", "powerlaw1.4", "powerlaw1.8"])
def test_boundaries_orbits(profile, embed_plot):
    np.seterr(all='raise')
    np.random.seed(42)
    prof = tc.standard_profiles(profile)
    
    rtid0 = 1e2
    tide = np.abs(prof.accr(rtid0)/rtid0)
    tprof = at.profiles.RadialTidalProfile(tide)
    
    rsetup = np.logspace(-10, 5, 6000)
    nprof = at.profiles.NumericalProfile(rsetup, prof.density(rsetup))
    
    prof_t = at.profiles.CompositeProfile(prof, tprof)
    nprof_t = at.profiles.CompositeProfile(nprof, tprof)

    rperi, rapo, rlmax, rtid, ramax_of_rp = at.mathtools.define_paspace_boundaries(prof_t.potential, prof_t.accr, prof_t.daccdr, nbins=555)
    nrperi, nrapo, nrlmax, nrtid, nramax_of_rp = at.mathtools.define_paspace_boundaries(nprof_t.potential, nprof_t.accr, nprof_t.daccdr, nbins=555)

    js = at.mathtools.calculate_radial_action_tanh_peri_apo(prof_t.potential, rperi[:-1], rapo[:-1], invalid_vr_to_zero=False)
    assert np.all(~np.isnan(js))
    njs = at.mathtools.calculate_radial_action_tanh_peri_apo(nprof_t.potential, nrperi[:-1], nrapo[:-1], invalid_vr_to_zero=True)
    assert np.all(js > 0)
    
    for p in (prof_t, nprof_t):
        print("----")
        rperi, rapo, rlmax, rtid, ramax_of_rp = at.mathtools.define_paspace_boundaries(p.potential, p.accr, p.daccdr, nbins=555)

        rp = 10**np.random.uniform(-5, np.log10(rlmax), 1000)
        ra = rp * 10 ** np.random.uniform(0, np.log10(ramax_of_rp(rp)/rp), rp.shape)
        
        valid = at.mathtools.rperi_rapo_valid(p.potential, p.accr, rp, ra)
        assert np.all(valid)

        js = at.mathtools.calculate_radial_action_tanh_peri_apo(p.potential, rp, ra, invalid_vr_to_zero=True)
        assert np.all(js > 0)