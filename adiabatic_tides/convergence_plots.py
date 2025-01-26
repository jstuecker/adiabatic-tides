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
    axs[1].set_ylim(-0.1,2.1)
    axs[1].set_xlabel("r")
    axs[0].set_ylabel(r"$\rho$")
    axs[1].set_ylabel(r"$\rho/\rho_0$")

def plot_perisplit_integration(prof, npart=100000, nsteps_chain=64, rpmin=0.1, rpmax=1.0, norb=100):

    rs0,Es0,Ls0,vrs0,ms,ri,rho = prof.sample_r_E_L_vr_m_metropolis(npart, rpmin=rpmin, rpmax=rpmax, nsteps_chain=nsteps_chain, get_rho=True, rmax=1e4)
    tmax = prof.tcirc(rpmin)
    
    rbins = np.geomspace(rpmin/2., rpmax*1e2, 100)

    fig, axs = plt.subplots(2,1, figsize=(6,6), sharex=True)
    fig.subplots_adjust(hspace=0.05)

    rhos,rs,vrs = [],rs0,vrs0
    for i in range(norb):
        rhos.append(at.mathtools.get_mass_profile(rs, ms, rbins)[0])
        rs, vrs = at.mathtools.integrate_radial_orbits(prof.accr, rs, vrs, Ls0, tmax, nsteps=40)

    plot_profile_rel(axs, rbins, rhos[0], prof.density, label="initial")
    plot_profile_rel(axs, rbins, rhos[-1], prof.density, label="final")
    plot_profile_rel(axs, rbins, np.mean(rhos[norb//2:], axis=0), prof.density, label="averaged (%d)" % (norb//2))

    axs[0].loglog(rbins, prof.density(rbins), color="black", ls="dashed", label="profile")
    axs[1].axhline(1., color="black", ls="dashed", label="profile")
    axs[0].loglog(ri, rho, color="black", ls="dotted", label="split")
    axs[1].semilogx(ri, rho/prof.density(ri), color="black", ls="dotted", label="split")

    axs[0].legend()
    
    return fig,axs