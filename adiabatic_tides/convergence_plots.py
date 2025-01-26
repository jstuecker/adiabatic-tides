import adiabatic_tides as at
import numpy as np
import matplotlib.pyplot as plt

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

def plot_perisplit_integration(prof, npart=100000, nsteps_chain=100, rpmin=0.1, rpmax=1.0, norb=100, steps_per_orb=100):

    rs0,Es0,Ls0,vrs0,ms,ri,rho = prof.sample_r_E_L_vr_m_metropolis(npart, rpmin=rpmin, rpmax=rpmax, nsteps_chain=nsteps_chain, get_rho=True, rmax=1e4)
    tmax = prof.tcirc(rpmin)
    
    rbins = np.geomspace(rpmin/2., rpmax*1e2, 100)

    fig, axs = plt.subplots(2,1, figsize=(6,6), sharex=True)
    fig.subplots_adjust(hspace=0.05)

    rhos,rs,vrs = [],rs0,vrs0
    for i in range(norb):
        rhos.append(at.mathtools.get_mass_profile(rs, ms, rbins)[0])
        rs, vrs = at.mathtools.integrate_radial_orbits(prof.accr, rs, vrs, Ls0, tmax, nsteps=steps_per_orb)

    plot_profile_rel(axs, rbins, rhos[0], prof.density, label="initial")
    plot_profile_rel(axs, rbins, rhos[-1], prof.density, label="final")
    plot_profile_rel(axs, rbins, np.mean(rhos[norb//2:], axis=0), prof.density, label="averaged (%d)" % (norb//2))

    axs[0].loglog(rbins, prof.density(rbins), color="black", ls="dashed", label="profile")
    axs[1].axhline(1., color="black", ls="dashed", label="profile")
    axs[0].loglog(ri, rho, color="black", ls="dotted", label="split")
    axs[1].semilogx(ri, rho/prof.density(ri), color="black", ls="dotted", label="split")

    axs[0].legend()
    
    return fig,axs

def plot_perimultisplit_integration(prof, n_per_split=100000, nsteps_chain=100, norb=40, steps_per_orb=100, rpsplits=None):
    if rpsplits is None:
        rpsplits = np.insert(np.logspace(-3,3,7), 0, 1e-6)
    rs0,Es0,Ls0,vrs0,ms = prof.sample_r_E_L_vr_m_metropolis_perisplits(n_per_split, rpsplits=rpsplits, flat=False, nsteps_chain=nsteps_chain)
    iperilow = np.arange(rs0.shape[0])[:,np.newaxis] * np.ones(rs0.shape[1], dtype=np.int64)
    tmax = prof.tcirc(rpsplits[iperilow.flat].reshape(rs0.shape))

    rbins, rbinsaniso = np.logspace(-3, 3, 61), np.logspace(-3, 3, 21)
    rcent, rcentaniso = np.sqrt(rbins[1:] * rbins[:-1]), np.sqrt(rbinsaniso[1:] * rbinsaniso[:-1])
    rhoref = prof.density(rcent)

    rs,vrs = rs0,vrs0
    rhos = np.zeros((norb, len(rpsplits), len(rbins)-1))
    rhoxvr2, rhoxvt2 = [], []
    for i in range(norb):
        for j in range(len(rpsplits)-1):
            rhos[i,j] = at.mathtools.get_mass_profile(rs[j], ms[j], rbins)[0]
        rhoxvr2.append(np.histogram(rs[rs >0.], weights=np.float128((vrs**2*ms)[rs > 0.]), bins=rbinsaniso)[0])
        rhoxvt2.append(np.histogram(rs[rs >0.], weights=((Ls0/rs)**2*ms)[rs > 0.], bins=rbinsaniso)[0])

        rs, vrs = at.mathtools.integrate_radial_orbits(prof.accr, rs, vrs, Ls0, tmax, nsteps=steps_per_orb)
    betas = 1-np.array(rhoxvt2)/np.array(rhoxvr2)/2.
    betamean = 1-np.mean(rhoxvt2, axis=0)/np.mean(rhoxvr2, axis=0)/2.

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

    axs[2].axhline(prof.anisotropy(), color="red", ls="dashed", alpha=0.5, label="True")

    plot_all(rhos[0],betas[0], alpha=0.5, label="Initial", label_peris=True)
    plot_all(rhos[-1],betas[-1], alpha=0.5, ls="dashed", label="Final", lw=2)
    plot_all(np.mean(rhos, axis=0),betamean, alpha=1.0, ls="dotted", label="Averaged (%d)" % norb, lw=2)

    for i in range(0,len(rpsplits)-1):
        rhotrue = at.mathtools.integrate_fofel_adaptive_rperi_lim(prof.f_of_el, prof.potential, rcent, N=200, rp1=rpsplits[i], rp2=rpsplits[i+1])
        axs[0].loglog(rcent, rhotrue, color="black", ls="dashed", alpha=0.6, label="true" if i == 0 else None)
        axs[1].semilogx(rcent, rhotrue/rhoref, color="black", ls="dashed", alpha=0.6)

    axs[1].axhline(1, color="black", ls="dashed", alpha=0.5)

    axs[0].set_ylim(np.min(rhoref), np.max(rhoref)*2)
    axs[1].set_ylim(-0.1,1.5)
    axs[2].set_ylim(prof.anisotropy()-0.45,prof.anisotropy()+0.45)

    axs[0].set_ylabel(r"$\rho(r)$")
    axs[1].set_ylabel(r"$\rho(r)/\rho_0(r)$")
    axs[2].set_ylabel(r"$\beta(r)$")

    axs[0].legend(ncol=2, fontsize=9)
    axs[2].legend(fontsize=9)

    for ax in axs:
        ax.grid("on")

    res = {"r": rcent, "rbeta":rcentaniso, "rhos": rhos, "betas": betas, "betamean": betamean}
    
    return fig,axs,res