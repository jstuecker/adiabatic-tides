import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from . import numerics
from . import adiabatic

def plot_profile_rel(axs, rbins, rhoi, rhoref, **kwargs):
    rcent = np.sqrt(rbins[1:]*rbins[:-1])
    axs[0].loglog(rcent, rhoi, **kwargs)
    rhor = rhoref(rcent)
    axs[1].semilogx(rcent, rhoi/rhor, **kwargs)
    axs[0].set_xlim(rbins[0], rbins[-1])
    axs[1].set_xlim(rbins[0], rbins[-1])
    axs[0].set_ylim(np.min(rhor)/2, np.max(rhor)*2.)
    axs[1].set_ylim(-0.1,1.5)
    axs[1].set_xlabel("r")
    axs[0].set_ylabel(r"$\rho$")
    axs[1].set_ylabel(r"$\rho/\rho_0$")

def plot_perisplit_integration(prof, npart=100000, nsteps_metropolis=64, rpmin=0.1, rpmax=1.0, norb=100, steps_per_orb=100):

    rs0,Es0,Ls0,vrs0,ms,ri,rho = prof.sample_particles(npart, mode="r_rp_l_vr_m_rrho_rho", rpmin=rpmin, rpmax=rpmax, nsteps_metropolis=nsteps_metropolis, rmax=1e4)

    tmax = prof.tcirc(rpmin)
    
    rbins = np.geomspace(rpmin/2., rpmax*1e2, 100)

    fig, axs = plt.subplots(2,1, figsize=(6,6), sharex=True)
    fig.subplots_adjust(hspace=0.05)

    rhos,rs,vrs = [],rs0,vrs0
    for i in range(norb):
        rhos.append(numerics.sample.get_mass_profile(rs, ms, rbins)[0])
        rs, vrs = numerics.sample.integrate_radial_orbits(prof.accr, rs, vrs, Ls0, tmax, nsteps=steps_per_orb)

    plot_profile_rel(axs, rbins, rhos[0], prof.density, label="initial")
    plot_profile_rel(axs, rbins, rhos[-1], prof.density, label="final")
    plot_profile_rel(axs, rbins, np.mean(rhos[norb//2:], axis=0), prof.density, label="averaged (%d)" % (norb//2))

    axs[0].loglog(rbins, prof.density(rbins), color="black", ls="dashed", label="profile")
    axs[1].axhline(1., color="black", ls="dashed", label="profile")
    axs[0].loglog(ri, rho, color="black", ls="dotted", label="split")
    axs[1].semilogx(ri, rho/prof.density(ri), color="black", ls="dotted", label="split")

    axs[0].legend()
    
    return fig,axs

def plot_perimultisplit_integration(prof, n_per_split=100000, nsteps_metropolis=64, norb=40, steps_per_orb=100, rpsplits=None):
    if rpsplits is None:
        rpsplits = np.insert(np.logspace(-3,3,7), 0, 1e-6)
    rs0,Es0,Ls0,vrs0,ms = prof.sample_particles_perisplits(n_per_split, rpsplits=rpsplits, flat=False, nsteps_metropolis=nsteps_metropolis)
    iperilow = np.arange(rs0.shape[0])[:,np.newaxis] * np.ones(rs0.shape[1], dtype=np.int64)
    tmax = prof.tcirc(rpsplits[iperilow.flat].reshape(rs0.shape))

    rbins, rbinsaniso = np.logspace(-3, 3, 61), np.logspace(-3, 3, 21)
    rcent, rcentaniso = np.sqrt(rbins[1:] * rbins[:-1]), np.sqrt(rbinsaniso[1:] * rbinsaniso[:-1])
    rhoref = prof.density(rcent)

    rs,vrs = rs0,vrs0
    rhos = np.zeros((norb, len(rpsplits), len(rbins)-1))
    rhoxvr2, rhoxvt2, rhoxvr2red, rhoxvt2red = [], [], [], []
    for i in range(norb):
        for j in range(len(rpsplits)-1):
            rhos[i,j] = numerics.sample.get_mass_profile(rs[j], ms[j], rbins)[0]
        rhoxvr2.append(np.histogram(rs[rs >0.], weights=np.float128((vrs**2*ms)[rs > 0.]), bins=rbinsaniso)[0])
        rhoxvt2.append(np.histogram(rs[rs >0.], weights=((Ls0/rs)**2*ms)[rs > 0.], bins=rbinsaniso)[0])
        v2 = vrs**2 + (Ls0/rs)**2
        rhoxvr2red.append(np.histogram(rs[rs >0.], weights=np.float128((vrs**2*ms/v2)[rs > 0.]), bins=rbinsaniso)[0])
        rhoxvt2red.append(np.histogram(rs[rs >0.], weights=((Ls0/rs)**2*ms/v2)[rs > 0.], bins=rbinsaniso)[0])

        rs, vrs = numerics.sample.integrate_radial_orbits(prof.accr, rs, vrs, Ls0, tmax, nsteps=steps_per_orb)
    betas = 1-np.array(rhoxvt2)/np.array(rhoxvr2)/2.
    betamean = 1-np.mean(rhoxvt2, axis=0)/np.mean(rhoxvr2, axis=0)/2.
    betamean_red = 1-np.mean(rhoxvt2red, axis=0)/np.mean(rhoxvr2red, axis=0)/2.

    if np.sum(rs <= 0.) > 0:
        print("Warning: fraction %.2e particles have r <= 0" % np.mean(rs <= 0.))

    fig, axs = plt.subplots(3,1, figsize=(6,8), sharex=True)
    fig.subplots_adjust(hspace=0.02)

    def plot_all(rhos,beta, label=None, label_peris=False, **kwargs):
        for i in range(0,len(rpsplits)-1):
            plabel= r"$r_p \in ($%.1g,%.1g$)$" % (np.log10(rpsplits[i]), np.log10(rpsplits[i+1])) if label_peris else None
            axs[0].loglog(rcent, rhos[i], color="C%d" % i, label=plabel,**kwargs)
            axs[1].semilogx(rcent, rhos[i]/rhoref, color="C%d" % i, **kwargs)
        
        axs[0].loglog(rcent, np.sum(rhos,axis=0), color="black", **kwargs)
        axs[1].semilogx(rcent, np.sum(rhos,axis=0)/rhoref, color="black", **kwargs)

        axs[2].semilogx(rcentaniso, beta, color="black", label=label, **kwargs)

    axs[2].axhline(prof.phase_space.anisotropy, color="red", ls="dashed", alpha=0.5, label="True")

    plot_all(rhos[0],betas[0], alpha=0.5, label="Initial", label_peris=True)
    plot_all(rhos[-1],betas[-1], alpha=0.5, ls="dashed", label="Final", lw=2)
    plot_all(np.mean(rhos, axis=0),betamean, alpha=1.0, ls="dotted", label="Averaged (%d)" % norb, lw=2)
    axs[2].semilogx(rcentaniso, betamean_red, color="green", label=r"Reduced, Av(%d)" % norb, lw=2, ls="dotted")

    for i in range(0,len(rpsplits)-1):
        rhotrue = numerics.integrate.integrate_fofel_adaptive_rperi_lim(prof.f_of_el, prof.potential, rcent, N=200, rp1=rpsplits[i], rp2=rpsplits[i+1])
        axs[0].loglog(rcent, rhotrue, color="black", ls="dashed", alpha=0.6, label="true" if i == 0 else None)
        axs[1].semilogx(rcent, rhotrue/rhoref, color="black", ls="dashed", alpha=0.6)

    axs[1].axhline(1, color="black", ls="dashed", alpha=0.5)

    axs[0].set_ylim(np.min(rhoref), np.max(rhoref)*2)
    axs[1].set_ylim(-0.1,1.5)
    axs[2].set_ylim(prof.phase_space.anisotropy-0.45,prof.phase_space.anisotropy+0.45)

    axs[0].set_ylabel(r"$\rho(r)$")
    axs[1].set_ylabel(r"$\rho(r)/\rho_0(r)$")
    axs[2].set_ylabel(r"$\beta(r)$")

    axs[0].legend(ncol=2, fontsize=9)
    axs[2].legend(ncol=2, fontsize=9)

    for ax in axs:
        ax.grid("on")

    res = {"r": rcent, "rbeta":rcentaniso, "rhos": rhos, "betas": betas, "betamean": betamean}
    
    return fig,axs,res

def plot_poisson_convergence(prof, spline_class=PchipInterpolator, title="Interpolation Convergence", **kwargs):
    """Makes a convergence plot of the poisson solver + interpolators against a given profile
    prof: a Radial profile
    spline_class: can be any interpolator from scipy.interpolate that implements .antiderivative,
        for example Akima1DInterpolator, CubicSpline, UnivariateSpline
    """
    rtest = np.logspace(-13,12,733)

    fig, axs=plt.subplots(2,3, figsize=(8,5), sharex=True)

    for n in 50,100,200,400:
        ri = np.logspace(-10,10,n)
        rho,m,phi = numerics.integrate.solve_poisson_via_spline_with_smart_boundaries(ri, prof.density(ri), spline_class, **kwargs)

        axs[0,0].loglog(rtest, rho(rtest), label="N=%d"%n)
        axs[1,0].loglog(rtest, np.abs(rho(rtest)/prof.density(rtest)-1.))

        axs[0,1].loglog(rtest, m(rtest), label="N=%d"%n)
        axs[1,1].loglog(rtest, np.abs(m(rtest)/prof.m_of_r(rtest)-1.))

        axs[0,2].loglog(rtest, phi(rtest), label="N=%d"%n)
        axs[1,2].loglog(rtest, np.abs(phi(rtest)/prof.potential(rtest, zero_at_zero=True) -1))

    axs[0,0].plot(rtest, prof.density(rtest), ls="dashed", color="black", label="ip. domain")
    axs[0,1].plot(rtest, prof.m_of_r(rtest), ls="dashed", color="black")
    axs[0,2].plot(rtest, prof.potential(rtest, zero_at_zero=True), ls="dashed", color="black")
    for j in range(0,3):
        axs[1,j].set_ylim(1e-6, 2)
        for i in(0,1):
            axs[i,j].axvline(ri[0], color="black", ls="dotted")
            axs[i,j].axvline(ri[-1], color="black", ls="dotted")
            axs[i,j].grid("on")

    axs[0,0].set_title("density")
    axs[0,1].set_title("mass")
    axs[0,2].set_title("potential")

    axs[0,0].set_ylabel("Value")
    axs[1,0].set_ylabel("Relative Error")
    axs[0,0].legend()

    if title is not None:
        fig.suptitle(title)

    return fig,axs

def plot_adiabatic_iterations(prof, rt0, title=None, verbose=1, **kwargs):
    # res = adiabatic.adiabatic_tidal_reconstruction(prof, np.abs(prof.accr(rt0)/rt0), rpmin=1e-20, eps=1e-3, get_all=True, verbose=verbose, **kwargs)
    att = adiabatic.AdiabaticTidalTransformation.from_rtid(prof, rt0, **kwargs)
    res = att.run().history

    fig,ax = plt.subplots(1,1, figsize=(6,5))
    for i in (0,1) + tuple(range(5, len(res), 5)):
        r, rho, frho, fm, fphi = res[i]
        eps = np.abs((res[i-1][2](r) - rho)/prof.density(r)).max()
        ax.semilogx(r, rho/prof.density(r), color=plt.get_cmap("rainbow")(i/len(res)), label=f"i = {i} eps = {eps:.1%}")
    ax.set_xlabel("r")
    ax.set_ylabel(r"$\rho/\rho_0$")
    ax.set_title(title)
    ax.legend()
    ax.axhline(1., color='black', ls='dotted')

    r, rho, frho, fm, fphi = res[-1]
    reldiff0 = np.abs((rho/prof.density(r) - 1))
    rsel = np.max(r[reldiff0 < 1e-3])
    ax.set_xlim(rsel, rt0*2)

    return fig,ax