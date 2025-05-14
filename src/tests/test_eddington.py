import numpy as np
import pytest
import matplotlib.pyplot as plt
import adiabatic_tides as at

from .test_config import embed_plot, check_max_relative_error, standard_profiles, plot_relative_error

def test_nfw_galpy(record_property):
    from galpy import potential
    from galpy.df import isotropicNFWdf

    np.seterr(all='raise', under='ignore')

    prof_nfw = standard_profiles("nfw")
    # r = np.logspace(-7, 7, 1000)
    # prof_num = at.profiles.NumericalProfile(r, prof_nfw.density(r))
    prof_num = prof_nfw

    galpy_nfw = potential.NFWPotential(amp=1,a=1) # these don't matter, we normalize the profile anyway
    galpy_df_widrow = isotropicNFWdf(pot=galpy_nfw, widrow=True, rmax=np.inf)
    galpy_df_lane = isotropicNFWdf(pot=galpy_nfw, widrow=False, rmax=np.inf)

    rtest = np.logspace(-5, 5, 500)
    f_nfw = prof_num.f_of_e(prof_num.potential(rtest, zero_at_zero=True))
    f_galpy_widrow = galpy_df_widrow.fE(galpy_nfw(rtest,0.))
    f_galpy_lane = galpy_df_lane.fE(galpy_nfw(rtest,0.))

    f_nfw *= f_galpy_lane[250]/f_nfw[250] # fit normalization

    # The widrow and lane functions have different inaccuracies
    # we always compare with the more close one to get the maximal error
    err1 = np.abs(f_nfw - f_galpy_widrow)/np.abs(f_galpy_widrow)
    err2 = np.abs(f_nfw - f_galpy_lane)/np.abs(f_galpy_lane)
    
    max_rel_err = np.max(np.min([err1,err2], axis=0))

    print("Maximal relative error:", max_rel_err)

    assert max_rel_err < 1e-2

# def test_pss():
#     pprof = at.profiles.PowerlawProfile(-1, rhoc=1.)

#     pss = at.phasespace.IsotropicPhaseSpaceSolver(pprof) # , rbins=2000, rnorm=1., rmax=1e6, rmin=1e-6, nbinsE = 100, dlog_emin=-3
    
#     rtest = np.logspace(-8, 8, 7*33)
#     etest = pprof.potential(rtest)

#     f1 = pprof.f_of_e(etest)
#     f2 = pss.f_of_e(etest)

#     #plot_phasespace(etest, f1, f2, "tests/plots/f_of_e_pss.pdf")

#     assert np.allclose(f1, f2, rtol=1e-3)

#test_pss()

@pytest.mark.parametrize("profile", ["plummer", "powerlaw0.5", "powerlaw1.0", "powerlaw1.4", "powerlaw1.8"])
def test_known_f_of_e(profile, embed_plot):
    np.seterr(all='raise')

    prof = standard_profiles(profile)
    # nprof = at.profiles.NumericalProfile(rsetup, prof.density(rsetup), ancorphi="rmin")
    prof.cfg.eddington.nr = 4000
    phasespace = at.phasespace.EddingtonPhaseSpace(prof.density, prof.potential, prof.cfg.general, prof.cfg.eddington, anisotropy=0.)

    rtest = np.geomspace(prof.rmin()*1e2, prof.rmax()/100, 7*33)
    
    e1 = prof.potential(rtest)
    e2 = prof.potential(rtest, zero_at_zero=True)

    f1 = prof.f_of_e(e1)
    f2 = phasespace.f_of_e(e2)

    embed_plot(plot_relative_error(f2, f1, 2e-3))
    check_max_relative_error(f2, f1, 2e-3)

def integrate_rho_f_rho(ri, rho):
    e, f = at.numerics.integrate.eddington_inversion(ri, rho)

    rho_new = at.numerics.integrate.integrate_f_to_density(e, f)
    
    return rho,f,rho_new

@pytest.mark.parametrize("profile", ["nfw", "plummer", "powerlaw0.5", "powerlaw1.0", "powerlaw1.4", "powerlaw1.8"])
def test_rho_f_rho(profile, embed_plot):
    np.seterr(all='raise')
    rsetup = np.logspace(-12, 12, 10000)

    prof = standard_profiles(profile)

    rho, f, rho_new = integrate_rho_f_rho(rsetup, prof.density(rsetup))

    sel = (rsetup > 1e-6) & (rsetup < 1e6)

    # embed_plot(plot_relative_error(rho_new[sel], rho[sel], 2e-3))
    check_max_relative_error(rho_new[sel], rho[sel], 2e-3)

@pytest.mark.parametrize("profile", ["plummer"])
def test_rho_f_rho_cored(profile, embed_plot):
    np.seterr(all='raise')
    rsetup = np.linspace(0, 40, 4000)

    prof = standard_profiles(profile)

    rho, f, rho_new = integrate_rho_f_rho(rsetup, prof.density(rsetup))

    sel = (rsetup > 1e-6) & (rsetup < 2)

    # embed_plot(plot_relative_error(rho_new[sel], rho[sel], 2e-3))
    check_max_relative_error(rho_new[sel], rho[sel], 2e-3)

@pytest.mark.parametrize("profile", ["nfw", "plummer", "powerlaw0.5", "powerlaw1.0", "powerlaw1.4", "powerlaw1.8"])
def test_rho_f_rho_adaptive(profile, embed_plot):
    np.seterr(all='raise', under="ignore")

    if profile == "plummer":
        tolerance = 1e-2
    else:
        tolerance = 2e-3

    prof = standard_profiles(profile)
    prof.phase_space = at.phasespace.EddingtonPhaseSpace(prof.density, prof.potential, prof.cfg.general, prof.cfg.eddington, anisotropy=0.)
    # prof = at.profiles.NumericalProfile(rsetup, prof.density(rsetup))

    rev = np.geomspace(max(1e-10, prof.rmin()*1e2), min(1e10, prof.rmax()/1e2), 7*31)
    # prof.f_of_e(E=(0.1,0.2), nintegrate=400)
    def phi(r): return prof.potential(r, zero_at_zero=True)
    # rho_new = at.numerics.integrals.integrate_f_paspace(prof.f_of_el, phi, prof.accr, rev, N=200) # , rperirange=(rsetup[0], rsetup[-1]), raporange=(rsetup[0], rsetup[-1])
    rho_new = at.numerics.integrate.integrate_f_to_density_adaptive(prof.f_of_e, phi(rev), nintegrate=400)
    rho = prof.density(rev)

    sel = np.ones_like(rev, dtype=bool)

    embed_plot(plot_relative_error(rho_new[sel], rho[sel], tolerance))
    check_max_relative_error(rho_new[sel], rho[sel], tolerance)

@pytest.mark.parametrize("profile", ["nfw", "plummer", "powerlaw0.5", "powerlaw1.0", "powerlaw1.4", "powerlaw1.8"])
def test_aniso_vs_iso_inv(profile):
    np.seterr(all='raise')

    rsetup = np.logspace(-10, 10, 2000)
    sel = (rsetup > 1e-9) & (rsetup < 1e3)

    prof = standard_profiles(profile)

    E,fa = at.numerics.integrate.anisotropic_inversion(rsetup, prof.density(rsetup), prof.potential(rsetup, zero_at_zero=True), beta=0.)
    E,fb = at.numerics.integrate.eddington_inversion(rsetup, prof.density(rsetup), prof.potential(rsetup))

    check_max_relative_error(fa[sel], fb[sel], 1e-2)


@pytest.mark.parametrize("profile", ["aniso-0.49pow1", "aniso-0.3pow1", "aniso0pow1", "aniso0.3pow1", "aniso0.4pow1.4"])
def test_known_aniso_fel(profile, embed_plot):
    np.seterr(all='raise')

    prof = standard_profiles(profile)
    beta = prof.anisotropy

    ri = np.logspace(-15,15,2000)
    E,f1 = at.numerics.integrate.anisotropic_inversion(ri, prof.density(ri), prof.potential(ri), beta=beta, nintegrate=100)
    
    sel = (ri > 1e-8) & (ri < 1e4)
    phi = prof.potential(ri[sel])
    Esamp = phi*np.random.uniform(1.,10., ri[sel].shape)
    Lsamp = np.sqrt(2.*Esamp - 2.*phi)/ri[sel]**2

    from scipy.interpolate import PchipInterpolator
    fip = PchipInterpolator(E, f1)
    fa = fip(Esamp) * Lsamp**(-2*beta)

    fa = np.interp(Esamp, E, f1) * Lsamp**(-2*beta)
    fb = prof.f_of_el(Esamp, Lsamp)

    # embed_plot(plot_relative_error(fa, fb, 2e-2))
    check_max_relative_error(fa, fb, 2e-2)

@pytest.mark.parametrize("profile", ["aniso-0.3pow1", "aniso0pow1", "aniso0.3pow1", "aniso0.4pow1.4"])
def test_known_aniso_fel_profile(profile, embed_plot):
    np.seterr(all='raise', under='ignore')

    prof = standard_profiles(profile)
    ri = np.logspace(-15,15,2000)
    nprof = at.profiles.NumericalProfile(ri, prof.density(ri), anisotropy=prof.anisotropy)
    
    sel = (ri > 1e-8) & (ri < 1e4)
    phi = prof.potential(ri[sel])
    Esamp = phi*np.random.uniform(1.,10., ri[sel].shape)
    Lsamp = np.sqrt(2.*Esamp - 2.*phi)/ri[sel]**2

    # from scipy.interpolate import PchipInterpolator
    # fip = PchipInterpolator(E, f1)
    # fa = fip(Esamp) * Lsamp**(-2*beta)

    fa = nprof.f_of_el(Esamp, Lsamp)
    fb = prof.f_of_el(Esamp, Lsamp)

    embed_plot(plot_relative_error(fa, fb, 1e-2))
    check_max_relative_error(fa, fb, 1e-2)

@pytest.mark.parametrize("profile", ["plummer", "aniso-0.49pow1", "aniso-0.3pow1", "aniso0pow1", "aniso0.3pow1", "aniso0.4pow1.4"])
def test_known_fel_integration(profile, embed_plot):
    np.seterr(all='raise', under='ignore')

    prof = standard_profiles(profile)

    r = np.logspace(-4,4,400)
    # rho = at.numerics.integrate.integrate_fofel_adaptive(prof.f_of_el, prof.potential, r)
    # rho = at.numerics.integrate.integrate_f_paspace(prof.f_of_rperi_rapo, prof.potential, prof.accr, r)

    if profile == "plummer":
        # For plummmer, it is not so great to integrate in logarithmic radial space
        # However, it does still converge with more integration points
        tolerance = 2e-2
        rho = prof.compute_pa_space_integral(r, nintegrate=100)
    else:
        tolerance = 1e-3
        rho = prof.compute_pa_space_integral(r)

    rhoref = prof.density(r)

    # embed_plot(plot_relative_error(rho, rhoref, tolerance))
    check_max_relative_error(rho, rhoref, tolerance)