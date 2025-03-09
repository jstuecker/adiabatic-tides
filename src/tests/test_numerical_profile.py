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
    
    prof_t = at.profiles.CompositeProfile(dm=prof, tide=tprof)
    nprof_t = at.profiles.CompositeProfile(dm=nprof, tide=tprof)

    for p in (prof, nprof):
        rlmax, rtid = at.numerics.search.find_rlmax(p.accr, rmin=p.rmin(), rmax=p.rmax()), at.numerics.search.find_rphimax(p.potential, rmin=p.rmin(), rmax=p.rmax())
        print("Without tide: boundary: %.5e %.5e" % (rlmax, rtid))
        assert (rlmax == np.infty) and (rtid == np.infty)
    print("With Tide:")
    for p in (prof_t, nprof_t):
        rlmax, rtid = at.numerics.search.find_rlmax(p.accr, rmin=p.rmin(), rmax=p.rmax()), at.numerics.search.find_rphimax(p.potential, rmin=p.rmin(), rmax=p.rmax())
        print("With tide: boundary: %.5e %.5e" % (rlmax, rtid))
        assert np.abs(rtid/rtid0 - 1.) < 1e-3

        # Check that rlmax corresponds to the maximum of the circular angular momentum
        assert np.all(p.vcirc(rtest)*rtest <= p.vcirc(rlmax)*rlmax)

        # Check that rtid corresponds to the maximum of the potential
        assert np.all(p.potential(rtest) <= p.potential(rtid))

    rperi, rapo, rlmax, rtid, ramax_of_rp = at.numerics.interpolate.define_paspace_boundaries(prof_t.potential, prof_t.accr, prof_t.daccdr, nbins=555, rmax=prof_t.rmax())
    nrperi, nrapo, nrlmax, nrtid, nramax_of_rp = at.numerics.interpolate.define_paspace_boundaries(nprof_t.potential, nprof_t.accr, nprof_t.daccdr, nbins=555, rmax=nprof_t.rmax())

    rtest = np.logspace(-8, -0.1, 33) * rlmax
    tc.check_max_relative_error(ramax_of_rp(rtest), nramax_of_rp(rtest), 1e-2)

# !!! add back "plummer" later and "powerlaw1.8"
@pytest.mark.parametrize("profile", ["nfw", "powerlaw0.5", "powerlaw1.0", "powerlaw1.4"])
def test_boundaries_orbits(profile, embed_plot):
    np.seterr(all='raise')
    np.random.seed(42)
    prof = tc.standard_profiles(profile)
    
    rtid0 = 1e2
    tide = np.abs(prof.accr(rtid0)/rtid0)
    tprof = at.profiles.RadialTidalProfile(tide)
    
    rsetup = np.logspace(-10, 5, 6000)
    nprof = at.profiles.NumericalProfile(rsetup, prof.density(rsetup))
    
    prof_t = at.profiles.CompositeProfile(dm=prof, tide=tprof)
    nprof_t = at.profiles.CompositeProfile(dm=nprof, tide=tprof)

    rperi, rapo, rlmax, rtid, ramax_of_rp = at.numerics.interpolate.define_paspace_boundaries(prof_t.potential, prof_t.accr, prof_t.daccdr, nbins=555, rmax=prof_t.rmax())
    nrperi, nrapo, nrlmax, nrtid, nramax_of_rp = at.numerics.interpolate.define_paspace_boundaries(nprof_t.potential, nprof_t.accr, nprof_t.daccdr, nbins=555, rmax=nprof_t.rmax())

    # js = at.numerics.integrate.calculate_radial_action_tanh_peri_apo(prof_t.potential, rperi[:-1], rapo[:-1], invalid_vr_to_zero=False)
    # assert np.all(~np.isnan(js))
    js = at.numerics.integrate.calculate_radial_action_tanh_peri_apo(nprof_t.potential, nrperi[:-1], nrapo[:-1])
    assert np.all(js > 0)
    
    for p in (prof_t, nprof_t):
        print("----")
        rperi, rapo, rlmax, rtid, ramax_of_rp = at.numerics.interpolate.define_paspace_boundaries(p.potential, p.accr, p.daccdr, nbins=555, rmax=p.rmax())

        rp = 10**np.random.uniform(-5, np.log10(rlmax), 1000)
        ra = rp * 10 ** np.random.uniform(0, np.log10(ramax_of_rp(rp)/rp), rp.shape)
        
        valid = at.numerics.search.rperi_rapo_valid(p.potential, p.accr, rp, ra)
        assert np.all(valid)

        js = at.numerics.integrate.calculate_radial_action_tanh_peri_apo(p.potential, rp, ra)
        assert np.all(js > 0)

@pytest.mark.parametrize("profile", ["nfw", "plummer", "powerlaw0.5", "powerlaw1.0", "powerlaw1.4", "powerlaw1.8"])
def test_tidal_boundary_detection(profile, embed_plot):
    np.seterr(all='raise', under="ignore")
    
    prof0 = tc.standard_profiles(profile)
    r = np.logspace(-10, 10, 1000)
    nprof = at.profiles.NumericalProfile(r, prof0.density(r))

    for prof in prof0, nprof:
        assert not np.isfinite(prof.rtid())
        assert not np.isfinite(prof.rlmax())

        rmax, vmax = prof.rmax_vmax()
        print(f"rmax {rmax:.2g}, vmax: {vmax:.2g}")

        if isinstance(prof0, at.profiles.PowerlawProfile):
            # Powerlaw profiles should have undefined rmax vmax
            assert (not np.isfinite(rmax)) and (not np.isfinite(vmax))
        else:
            assert rmax > 0 and vmax > 0

        for rt in 1e-2,1,1e2:
            # With tide boundaries should be defined
            tprof = at.profiles.RadialTidalProfile(np.abs(prof0.accr(rt)/rt))
            cprof = at.profiles.CompositeProfile(mass=prof, tide=tprof, external=("tide",))

            assert np.abs(cprof.rtid() / rt - 1) < 1e-3
            assert cprof.rlmax() <= cprof.rtid()
            assert cprof.rmax_vmax()[0] <= cprof.rtid()

@pytest.mark.parametrize("profile", ["nfw", "plummer", "powerlaw0.5", "powerlaw1.0", "powerlaw1.4", "powerlaw1.8"])
def test_rcirc_finding(profile, embed_plot):
    np.seterr(all='raise', under="ignore")
    
    prof0 = tc.standard_profiles(profile)
    r = np.logspace(-10, 10, 1000)
    nprof = at.profiles.NumericalProfile(r, prof0.density(r))

    for prof in prof0, nprof:
        r = np.logspace(-5,5,100)
        lcirc = prof.vcirc(r)*r
        ecirc = 0.5*prof.vcirc(r)**2 + prof.potential(r)
        
        rphi = prof.r_of_potential(prof.potential(r))
        rlcirc = prof.r_of_lcirc(lcirc)
        recirc = prof.r_of_ecirc(ecirc)

        tc.check_max_relative_error(rphi, r, 1e-5)
        tc.check_max_relative_error(rlcirc, r, 1e-5)
        tc.check_max_relative_error(recirc, r, 1e-5)

@pytest.mark.parametrize("profile", ["nfw", "plummer", "powerlaw0.5", "powerlaw1.0", "powerlaw1.4", "powerlaw1.8"])
def test_rcirc_finding_nonmonotoneous(profile, embed_plot):
    np.seterr(all='raise', under="ignore")
    
    prof0 = tc.standard_profiles(profile)
    r = np.logspace(-15, 15, 2000)
    nprof = at.profiles.NumericalProfile(r, prof0.density(r))

    for prof in prof0, nprof:
        for rt in 1,1e2:
            # With tide boundaries should be defined
            tprof = at.profiles.RadialTidalProfile(np.abs(prof0.accr(rt)/rt))
            cprof = at.profiles.CompositeProfile(mass=prof, tide=tprof, external=("tide",))

            r = np.geomspace(1e-2,0.99,133)*rt
            
            rphi = cprof.r_of_potential(cprof.potential(r))
            tc.check_max_relative_error(cprof.potential(rphi), cprof.potential(r), 1e-5)

            ecirc = 0.5*cprof.vcirc(r)**2 + cprof.potential(r)
            
            lcirc = cprof.vcirc(r)*r
            for mode in "asc", "desc":
                rlcirc = cprof.r_of_lcirc(lcirc, mode=mode)
                tc.check_max_relative_error(cprof.vcirc(rlcirc)*rlcirc, lcirc, 1e-5)

                recirc = cprof.r_of_ecirc(ecirc, mode=mode)
                tc.check_max_relative_error(0.5*cprof.vcirc(recirc)**2 + cprof.potential(recirc), ecirc, 1e-5)


def test_string_repr():
    nfw = at.profiles.NFWProfile(1., m200c=1e12)
    ppow = at.profiles.PowerlawProfile(1.1, anisotropy=0.1)
    ptide = at.profiles.RadialTidalProfile(1.2)
    peinasto = at.profiles.EinastoProfile()
    pplum = at.profiles.PlummerProfile()
    piso = at.profiles.IsothermalSphere()
    cprof = at.profiles.CompositeProfile(dm=nfw, tide=ptide, external=("tide",))

    r = np.logspace(-10,10,133)
    nprof = at.profiles.NumericalProfile(r, nfw.density(r))
    nprofb = at.profiles.NumericalProfile(r, nfw.density(r))

    # pprof
    part = ppow.sample_particles(1000, mode="dict", rmax=1.)
    pprof = at.profiles.ParticleProfile(part, rbins=r)

    # Mimic result of adiabatic calculation
    ares = (nprof.ri, nprof.rhoi, nprof.density, nprof.m_of_r, nprof.potential)
    aprof = at.adiabatic.AdiabaticResultProfile(ares, nprof.f_of_rperi_rapo)

    for prof in (nfw, ppow, ptide, peinasto, pplum, piso, cprof, nprof, pprof, aprof):
        print(prof)

    print("\n")

    assert repr(nprof) == repr(nprofb)

    for prof in (nfw, ppow, ptide, peinasto, pplum, piso, cprof, nprof, pprof, aprof):
        print(repr(prof))
        print()

def test_yaml_cfg():
    # Check that example config is consistent with default config
    cfg = at.Config.from_yaml("example_config.yaml")
    default_cfg = at.config.Config()
    
    print(cfg)
    cfg.print_modified()

    for c1, c2 in zip(cfg.configs, default_cfg.configs):
        assert c1 == c2

def test_config_constructors():
    cfg1 = at.Config.from_yaml("example_config.yaml")
    cfg2 = at.Config()

    # Reference init
    cfg3 = at.Config.flexible_init(cfg2)
    assert cfg3 == cfg2
    cfg3.general.rmin =  2
    assert cfg3 == cfg2 

    # Init with default
    cfg4 = at.Config.flexible_init(None, cfg1)
    assert cfg4 == cfg1

    # Dict[Dict] Init
    cfg5 = at.Config.flexible_init(dict(units = dict(length=cfg1.units.length)), cfg1)
    assert(cfg5 == cfg1)
    cfg6 = at.Config.flexible_init(dict(units = dict(length=cfg1.units.length*2)), cfg1)
    assert(cfg6 != cfg1)

    # Dict[SubConfig] Init
    units = at.config.UnitConfig(length=cfg1.units.length)
    cfg7 = at.Config.flexible_init(dict(units = units), cfg1)
    assert(cfg7 == cfg1)
    units = at.config.UnitConfig(length=cfg1.units.length*2)
    cfg8 = at.Config.flexible_init(dict(units = units), cfg1)
    assert(cfg8 != cfg1)