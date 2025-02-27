import numpy as np
from . import integrate
from .utility import cosh_space, Jacobian_ldlde_drpdra
from .interpolate import vectorized_interp

# ========= Utilitys functions for binning sampled particles =============== #

def get_mass_profile(ri, mi, rbins):
    """Returns the mass profile, given some particle radii
    
    ri : radii of particles
    mi : masses of the particles
    rbins : radial bins to use
    
    returns : (rho, mprof) where rho is the density profile and 
              mprof is the cumulative mass profile
              each has shape len(rbins) -1
    """
    vbins = 4./3.*np.pi*(rbins[1:]**3 - rbins[:-1]**3)
    m,_ = np.histogram(ri, bins=rbins, weights=mi)

    # Estimate the error in the mass profile that results
    # from the numpy histogram being based on a cumulative sum
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_inc = np.nanmin(np.clip(m,np.min(mi),None)/np.cumsum(m))
    if rel_inc <= 1e-13:
        print("Warning: I expect cancellation in the mass-profile calculation, worst mass ratio = %.2e" % (rel_inc))

    rho = m / vbins
    mprof = np.concatenate([[0.], np.cumsum(m)])

    return rho, mprof

def get_anisotropy_profile(ri, mi, vri, li, rbins, reduced=False):
    """Returns the mass profile, given some particle radii
    
    ri : radii of particles
    vri : radial velocities of the particles
    li : angular momentum of the particles
    mi : masses of the particles
    rbins : radial bins to use
    reduced : If true, then we replace e.g. simga_vt2 by expect(vt**2/v**2)
            instead of expect(vt**2). For profiles with constant anisotropy
            this leads to the same parameter, but it seems more robustly defined
            especially, for profiles with shallow energy distributions
    
    returns : r, beta  where the anisotropy beta is 
              given by the value of 1-simga_vt**2/sigma_vr**2/2.
    """
    if reduced:
        v2 = (vri**2 + (li/ri)**2)
        rhoxvr2 = np.histogram(ri.flatten(), weights=np.float128(vri**2*mi/v2).flatten(), bins=rbins)[0]
        rhoxvt2 = np.histogram(ri.flatten(), weights=np.float128((li/ri)**2*mi/v2).flatten(), bins=rbins)[0]
    else:
        rhoxvr2 = np.histogram(ri.flatten(), weights=np.float128(vri**2*mi).flatten(), bins=rbins)[0]
        rhoxvt2 = np.histogram(ri.flatten(), weights=np.float128((li/ri)**2*mi).flatten(), bins=rbins)[0]

    beta = 1-rhoxvt2/rhoxvr2/2.

    rcent = np.sqrt(rbins[1:]*rbins[:-1])
    return rcent, beta

# ================== Generic functions for sampling ======================== #

def random_direction(size, ndim):
    """Samples random unit vectors
    
    size : number of vectors to sample, can be tuple
    ndim : the dimension of the space
    
    returns : array with shape (*size, ndim)
    """
    x = np.random.normal(size=tuple(size) + (ndim,))
    r = np.sqrt(np.sum(x**2,axis=-1))
    return x/r[...,np.newaxis]

def sample_metropolis_hastings(f, x0, stepsize=1., nsteps=1000, nhalf=None):
    """Does an mcmc sampling of a probability distribution function
    only returns the last step of each chain.
    
    f : a pdf to sample
    x0 : start locations
    sig : standard deviation(s) for step sizes
    nsteps : total number of steps to perform
    """
    
    x = np.copy(x0)
    f0 = f(x)

    if np.sum(f0 <= 0) > 0:
        print("Warning, I am starting with invalid points... %d" % np.sum(f0 <= 0))
        #raise ValueError("Invalid starting points")

    for i in range(0, nsteps):
        if nhalf is not None:
            if (i % nhalf == 0) & (i > 0):
                stepsize = np.array(stepsize) / 2.

        dx = np.random.normal(loc=0., scale=stepsize, size=x.shape)

        f1 = f(x + dx)

        alpha = f1 / f0
        u = np.random.uniform(0., 1., size=x.shape[0])
        accept = u <= alpha

        x[accept] = (x+dx)[accept]
        f0[accept] = f1[accept]

    if np.sum(f0 <= 0) > 0:
        print("Warning, I ended with invalid points... %d" % np.sum(f0 <= 0))
        #raise ValueError("Invalid starting points")
        
    return x

def sample_from_Finv(Finv, size):
    Fs = np.random.uniform(0., 1., size)

    return Finv(Fs)

# ================= Functions for sampling particles ======================= #

def sample_radii(ri, mi, size=1, rmax=None):
    """sample radii from a given mass profile

    returns sampled radii and total mass
    """
    mi = mi-mi[0]

    if rmax is None:
        Mmax = mi[-1]
    else:
        Mmax = np.interp(rmax, ri, mi)

    def FcumInv(f):
        return np.interp(f, mi/Mmax, ri)
    
    return sample_from_Finv(FcumInv, size)

def sample_rimi_from_density(ri, rhoi, size=1, rmax=None, weights=None):
    """ sample radii and masses from a density profile
        weights : can be provided to sample more particles (with lower weights)
                  at different radii. The number of particles at a radius will
                  be proportional to rhoi*weights, but the masses to 1/weights
    """
    mi = integrate.trapez_integral_cumulative(ri, 4.*np.pi*rhoi*ri**2)
    if rmax is None:
        Mmax = mi[-1]
    else:
        Mmax = np.interp(rmax, ri, mi)

    if weights is None:
        rsamp = sample_radii(ri, mi, size=size, rmax=rmax)
        msamp = np.ones_like(rsamp) * Mmax / len(rsamp)
    else:
        miwi = integrate.trapez_integral_cumulative(ri, 4.*np.pi*rhoi*ri**2*weights)
        rsamp = sample_radii(ri, miwi, size=size, rmax=rmax)
        msamp = 1. / np.interp(rsamp, ri, weights)
        msamp *= Mmax / np.sum(msamp) # normaliz

    return rsamp, msamp

def sample_conditional_energy(phisamp, ei, fi, emaxsamp=None):
    """Samples the energy, given that the particle is at a radius where the potential is phi
    phisamp : potential energies of sampled particles
    ei, fi: phase space distribution as function of energy
    """
    print("Warning this method is deprecated, use adaptive one!")
    
    rho_phi_e = integrate.integrate_fiso_cumulative_phi_e(ei, fi)
    with np.errstate(divide='ignore', invalid='ignore'):
        Fcum = rho_phi_e / rho_phi_e[:,-1:]
    Fsamp = np.random.uniform(0., 1., phisamp.shape)

    itab = np.interp(phisamp, ei, np.arange(len(ei)))

    esamp = np.zeros_like(phisamp)
    for i in range(0, len(phisamp)):
        #Fcum_sel = np.interp(phisamp, ei, Fcum) # select the correct row of our table
        #i0 = np.floor(itab[i]).astype(int)
        #di = itab[i] - i0
        Fcum_sel = Fcum[itab[i].astype(int)]
        #Fcum_sel = Fcum[i0] * (1-di) + Fcum[i0+1] * di
        esamp[i] = np.interp(Fsamp[i], Fcum_sel, ei) # do  the inversion sampling
    
    return esamp

def sample_conditional_energy_adaptive(phisamp, f_of_e, emax=None, nintegrate=1000):
    """Samples the energy, given that the particle is at a radius where the potential is phi
    phisamp : potential energies of sampled particles
    ei, fi: phase space distribution as function of energy
    """

    assert np.min(phisamp) > 0, "Please normalize potential to zero at zero"
    
    Fsamp = np.random.uniform(0., 1., phisamp.shape)

    if emax is None:
        emax = np.max(phisamp)*1e3

    Esamp = np.zeros_like(phisamp)
    for i,phi in enumerate(phisamp):
        eeval = phi * cosh_space(emax/phi, nintegrate, 2)
        assert ~np.isnan(np.max(eeval))
        
        integrand = f_of_e(eeval) * np.sqrt(np.clip(eeval - phi, 0, None)) 
        fcum = integrate.trapez_integral_cumulative(eeval, integrand)

        Esamp[i] = np.interp(Fsamp[i], fcum/fcum[-1], eeval)
    
    return Esamp

def sample_conditional_energy_perisplit_adaptive(rsamp, pot, f_of_e, rp1=0, rp2=np.infty, nintegrate=1000, rmaxfac=1e10):
    """Samples the energy, given that the particle is at a radius where the potential is phi
    but limits to orbits which have pericenters in rp1 < rp < rp2
    phisamp : potential energies of sampled particles
    ei, fi: phase space distribution as function of energy
    """
    Fsamp = np.random.uniform(0., 1., rsamp.shape)

    phip1, phip2 = pot(rp1), pot(rp2)

    rfacs = cosh_space(rmaxfac, nintegrate, 2)

    Esamp = np.zeros_like(rsamp)
    for i,r in enumerate(rsamp):
        reval = r * rfacs

        phi, eeval = pot(r), pot(reval)

        q1 = np.clip(eeval - phi - np.clip(eeval - phip1, 0, None) * (rp1**2 / r**2), 0, None) * (r >= rp1)
        q2 = np.clip(eeval - phi - np.clip(eeval - phip2, 0, None) * (rp2**2 / r**2), 0, None) * (r >= rp2)
        q2 = np.nan_to_num(q2, 0)
        integrand = f_of_e(eeval) * (np.sqrt(q1) - np.nan_to_num(np.sqrt(q2),0))

        fcum = integrate.trapez_integral_cumulative(eeval, integrand)

        Esamp[i] = np.interp(Fsamp[i], fcum/fcum[-1], eeval)
    
    return Esamp

def sample_conditional_energy_adaptive_batched(phisamp, f_of_e, emax=None, nintegrate=1000, batchsize=500):
    """Samples the energy, given that the particle is at a radius where the potential is phi
    phisamp : potential energies of sampled particles
    ei, fi: phase space distribution as function of energy
    """

    assert np.min(phisamp) > 0, "Please normalize potential to zero at zero"
    
    Fsamp = np.random.uniform(0., 1., phisamp.shape)

    if emax is None:
        emax = np.max(phisamp)*1e3

    facspace = cosh_space(emax/np.min(phisamp), nintegrate, 2)

    Esamp = np.zeros_like(phisamp)

    nlow = np.arange(0, len(phisamp), batchsize)
    nup = np.clip(nlow + batchsize, 0, len(phisamp))

    for ilow,iup in zip(nlow, nup):
        phi = phisamp[ilow:iup,np.newaxis]
        eeval = phi * facspace
        assert ~np.isnan(np.max(eeval))
        
        integrand = f_of_e(eeval) * np.sqrt(np.clip(eeval - phi, 0, None)) 
        fcum = integrate.trapez_integral_cumulative(eeval, integrand)

        Esamp[ilow:iup] = vectorized_interp(Fsamp[ilow:iup], fcum/fcum[:,-1:], eeval)
    
    return Esamp

def sample_conditional_vr_L_isotropic(r, dE):
    """Input: dE=E-phi(r)
    Output: vr, L
    """
    vel= random_direction(dE.shape, 3) * np.sqrt(2.*dE)[...,np.newaxis]
    # assume r = x-axis
    vr = vel[...,0]

    L = np.linalg.norm([0.*r, -r * vel[...,2], r * vel[...,1]], axis=0)

    return vr, L

def sample_conditional_L_vr_perisplit(r, E, pot, rp1, rp2):
    """Samples angular momenta for a given (r,E) and
    assuming that the peri center has to be in the range (rp1, rp2)
    """
    Fsamp = np.random.uniform(0., 1., r.shape)

    phip1, phip2 = pot(rp1), pot(rp2)
    phi = pot(r)

    assert np.all(E >= phi)
    assert np.all(r >= rp1), "Your input radii are not quite consistent. Maybe a numerical error from radial bins not lining up with peri center radii?"

    def Fu(L2): # Parent function of ang-mom distr. (for isotropic systems)
        return -np.sqrt(np.clip(2.*(E - phi) - L2/r**2,0,None))
    
    Lmin2 = np.clip(2.*(E - phip1) * rp1**2, 0, None)
    Lmax2 = 2.*(E - phi) * r**2
    Lmaxb2 = 2.*(E - phip2) * rp2**2
    Lmax2[r >= rp2] = Lmaxb2[r >= rp2]

    # Cumulative Function is F = (Fu(L) - Fu(Lmin))/(Fu(Lmax) - Fu(Lmin))
    # Invert this
    FuL = Fsamp * (Fu(Lmax2) - Fu(Lmin2)) + Fu(Lmin2)
    L2 = (2. * (E-phi) - FuL**2) * r**2

    vr = np.sqrt(np.clip(2.*(E - phi) - L2/r**2,0,None))
    vr = vr * np.sign(np.random.uniform(-1,1, r.shape))

    return np.sqrt(L2), vr

def sample_E_L_vr_given_r_metropolis(f_of_el, pot, vcirc, rs, nsteps_chain=100):
    phis = pot(rs)
    vref = vcirc(rs)
    
    def likelihood_of_vel_given_r(logvtheta):
        # Likelihood in polar coordinates in velocity space
        vs,thetas = np.exp(logvtheta[...,0]), logvtheta[...,1]

        es = phis + 0.5*vs**2
        ls = vs * rs * np.abs(np.sin(thetas))
        
        return f_of_el(es,ls) * vs**3 * np.abs(np.sin(thetas))

    logv0 = np.random.uniform(-2., 2., rs.shape) + np.log(vref)
    theta0 = np.random.uniform(0., np.pi, rs.shape)
    
    logvtheta = np.stack([logv0,theta0], axis=-1)
    stepsize = np.stack([2.0, 0.2*np.pi], axis=-1)
    
    logvtheta = sample_metropolis_hastings(likelihood_of_vel_given_r, logvtheta, stepsize=stepsize, nsteps=nsteps_chain)
    vs, thetas = np.exp(logvtheta[...,0]), logvtheta[...,1]
    Ls = vs *rs * np.abs(np.sin(thetas))
    vrs = vs * np.cos(thetas)
    Es = phis + 0.5*vs**2

    return Es, Ls, vrs

def sample_E_L_vr_given_r_metropolis_perisplit(f_of_el, pot, accr, rs, nsteps_chain=40, rp1=None, rp2=None, phimax=np.infty):
    # we have to sample from
    # f(E,L) v^2 sin(theta) dv dtheta
    # L = v r sin(theta)
    # vr = v cos(theta)

    # parameterize v in terms of u_p = r_p/r
    # where r_p is the pericenter radius
    # this way it is easy to predict the relevant
    # also substitute mu = sin(theta)
    # additionally transform to a hyperbolic space
    # where the boundaries are rapidly approached at infinity

    def dv_du_overv(u, sintheta2, rs, phis):
        rp = rs * u
        f = u**-3 * sintheta2 / (u**-2 * sintheta2 - 1)
        f = f + accr(rp)*rs / (2*phis - 2*pot(rp))
        return f

    assert rp2 > rp1
    umax = np.clip(rp2/rs,None,1.)
    umin = rp1 / rs
    
    def mu_of_t(t, a=umin, b=1.):
        return 0.5*(b+a) + 0.5*(b-a)*np.tanh(t)
    def dmudt_of_mu(x, a=umin, b=1.):
        return 2./(b-a) * (b-x)*(x-a)
    def t_of_mu(x, a=umin, b=1.):
        return np.arctanh((x - 0.5*(b+a))/(0.5*(b-a)))

    def u_of_s(s, a=umin, b=umax):
        return 0.5*(b+a) + 0.5*(b-a)*np.tanh(s)
    def du_ds_of_u(x, a=umin, b=umax):
        return 2./(b-a) * (b-x)*(x-a)
    def s_of_u(x, a=umin, b=umax):
        return np.arctanh((x - 0.5*(b+a))/(0.5*(b-a)))

    phis = pot(rs)
    def likelihood_of_vel_given_r(s_t):
        # Likelihood in polar coordinates in velocity space
        us,mus = u_of_s(s_t[...,0]), mu_of_t(s_t[...,1])

        rp = rs * us
        vs2 = 2.*(phis - pot(rp)) / (us**-2 * mus**2 - 1.)

        valid = (rp >= rp1) & (rp <= rs) & (rp <= rp2) & (vs2 > 0) & (mus <= 1.) & (mus >= 0.)

        es = phis[valid] + 0.5*vs2[valid]
        ls = rs[valid] * np.sqrt(vs2[valid]) * mus[valid]

        dvol = dmudt_of_mu(mus)[valid] * du_ds_of_u(us)[valid]
        dvol *= vs2[valid] * mus[valid] / np.sqrt(1. - mus[valid]**2)
        dvol *= dv_du_overv(us[valid], mus[valid]**2, rs[valid], phis[valid]) * np.sqrt(vs2[valid])

        f = np.zeros_like(rs)
        f[valid] = f_of_el(es,ls) * dvol
        
        return f
    
    s0 = np.random.uniform(-1, 1, rs.shape)
    if phimax < np.infty:
        u0 = u_of_s(s0)
        mumin = u0 * np.sqrt((phimax-pot(u0*rs))/(phimax-phis))
    else:
        mumin = u_of_s(s0)
    # dmu = 1. - mumin
    # mu0 = np.random.uniform(1.-0.66*dmu, 1.-0.33*dmu, rs.shape)
    mu0 = np.random.uniform(mumin, 1., rs.shape)

    s_t = np.stack([s0,t_of_mu(mu0)], axis=-1)
    stepsize = np.stack([8./np.cbrt(nsteps_chain), 8./np.cbrt(nsteps_chain)], axis=-1)
    
    s_t = sample_metropolis_hastings(likelihood_of_vel_given_r, s_t, stepsize=stepsize, nsteps=nsteps_chain)

    # Transform back
    us,mus = u_of_s(s_t[...,0]), mu_of_t(s_t[...,1])
    vs = np.sqrt(2.*(phis - pot(us*rs)) / (us**-2 * mus**2 - 1.))

    es = phis + 0.5*vs**2
    ls = rs * vs * mus
    vrs = vs * np.sqrt(1. - mus**2) * np.sign(np.random.uniform(-1,1,rs.shape))

    return es, ls, vrs

def sample_rp_ra_given_r_metropolis_perisplit(f_of_rp_ra, pot, accr, rs, nsteps_chain=64, rperirange=(0., np.infty)):
    """Samples particle's peri-apo-centers given their radii and an allowed range of peri-center"""
    assert (np.min(rs) >= rperirange[0]) & (rperirange[1] >= rperirange[0])

    phis = pot(rs)
    def likelihood_rpra(rp, ra):
        e,l,ldlde = Jacobian_ldlde_drpdra(pot, accr, rp, ra, get_el=True)
        vr = np.sqrt(np.clip(2*e - 2*phis - l**2/rs**2, 0, None))

        valid = (ldlde > 0.) & (vr > 0.) & (l > 0.)

        f = np.zeros_like(e)
        f[valid] = f_of_rp_ra(rp[valid], ra[valid])

        return np.divide(f * ldlde, vr, out=np.zeros_like(f), where=valid)

    # Morph space to a uniform space in (u,v) going from (-inf, inf) each
    rpmin, rpmax = np.clip(rperirange[0], 0, rs), np.clip(rperirange[1], 0, rs)
    def rp_ra(u,v):
        rp = 0.5*(rpmax+rpmin) + 0.5*(rpmax-rpmin) * np.tanh(u)
        ra = rs * (1. + np.exp(v))
        return rp, ra
    
    def drp_x_dra_dudv(rp,ra):
        drpdu = 2 * (rpmax - rp) * (rp - rpmin) / (rpmax - rpmin)
        dradv = ra-rs

        return drpdu * dradv

    def likelihood_uv(uv):
        rp, ra = rp_ra(uv[...,0], uv[...,1])
        jacobian = drp_x_dra_dudv(rp, ra)

        return likelihood_rpra(rp, ra) * jacobian
    
    uv0 = np.random.uniform(-2,2, size=rs.shape + (2,))
    uv = sample_metropolis_hastings(likelihood_uv, uv0, stepsize=(4., 8.), nsteps=nsteps_chain, nhalf=nsteps_chain//4) # , nhalf=nsteps_chain//4
    
    return rp_ra(uv[...,0], uv[...,1])

def E_L_vr_from_rp_r_ra(pot, rperi, r, rapo):
    phip, phi, phia = pot(rperi), pot(r), pot(rapo)
    
    e = phip + (phia - phip)*(rapo**2) / (rapo**2 - rperi**2)
    l = np.sqrt(2. * (phia - phip) / (rperi**-2 - rapo**-2))
    
    vr = np.sqrt(np.clip(2*e - 2*phi - l**2/r**2, 0, None))
    vr = vr * np.sign(np.random.uniform(-1,1,size=vr.shape))
    
    return e, l, vr

# ================= Functions for integrating orbits ======================= #

def integrate_radial_orbits(acc_func, r, vr, L, t, nsteps=1000, time_dependent_acc=False, t0=0.):
    # Hamiltonian = phi(r) + 0.5 vr**2 + 0.5 L**2 / r**2
    # dvr/dt = -dphi/dr - L**2 / r**3
    
    dt = t/nsteps

    for i in range(nsteps):
        # Drift Kick Drift Integrator
        r = r + vr*dt*0.5
        if time_dependent_acc:
            vr = vr + (acc_func(r, t=t0+i*dt) + L**2/r**3) * dt
        else:
            vr = vr + (acc_func(r) + L**2/r**3) * dt
        r = r + vr*dt*0.5

    return r, vr

def integrate_radial_orbits_with_snaps(acc_func, r, vr, L, t, nsnaps=10, nsteps_per_snap=100, time_dependent_acc=False):
    for i in range(0, nsnaps):
        r, vr = integrate_radial_orbits(acc_func, r, vr, L, t/nsnaps, t0=i*(t/nsnaps), nsteps=nsteps_per_snap, time_dependent_acc=time_dependent_acc)
        yield r, vr
