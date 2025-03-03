import numpy as np
from .import RadialProfile
from ..phasespace import AnalyticPhaseSpace
from scipy.special import gamma as GammaF
from scipy.optimize import brentq

# Helper functions

def RvirOfMvir(mvir, mode="crit", delta=200., h=0.679, omega_m=0.30):
    """Returns the virial radius of a halo with a given virial mass
    
    It is assumed that mvir is the mass enclosed inside that radius and
    that the object is 'delta' times as dense as the critical/mean
    density of the universe
    
    mvir : virial mass in solar masses
    mode : can be 'crit' or 'mean' to use the critical or mean density
    delta : the over-density. Typically 200 is used here
    h : reduced hubble parameter. If set to 1 units will change to
        mass in Msol/h and radius in Mpc/h
    omega_m : matter density parameter, only relevant when using mode='mean'
    
    returns : virial radius in Mpc
    """
    G = 43.0071057317063 * 1e-10  #  Grav. constant in Mpc (km/s)^2 / Msol
    rhocrit = 3.0 / (8.0 * np.pi * G) * (1e2*h)**2
    
    if mode == "crit":
        rhoref = rhocrit
    elif mode == "mean":
        rhoref = omega_m * rhocrit
    else:
        raise ValueError("Unknown mode=%s, can be 'crit' or 'mean'" % mode)

    return  np.cbrt(mvir / (rhoref * 4.*np.pi/3. * delta))

def MvirOfRvir(rvir, mode="crit", delta=200., h=0.679, omega_m=0.30):
    """Returns the virial mass of a halo with a given virial radius
    
    It is assumed that mvir is the mass enclosed inside rvir and
    that the object is 'delta' times as dense as the critical/mean
    density of the universe
    
    rvir : virial radius in Mpc
    mode : can be 'crit' or 'mean' to use the critical or mean density
    delta : the over-density. Typically 200 is used here
    h : reduced hubble parameter. If set to 1 units will change to
        mass in Msol/h and radius in Mpc/h
    omega_m : matter density parameter, only relevant when using mode='mean'
    
    returns : virial mass in Msol
    """
    G = 43.0071057317063 * 1e-10  #  Grav. constant in Mpc (km/s)^2 / Msol
    rhocrit = 3.0 / (8.0 * np.pi * G) * (1e2*h)**2
    
    if mode == "crit":
        rhoref = rhocrit
    elif mode == "mean":
        rhoref = omega_m * rhocrit
    else:
        raise ValueError("Unknown mode=%s, can be 'crit' or 'mean'" % mode)
        
    return rhoref * 4.*np.pi/3. * delta * rvir**3

def rhoc_rs_to_conc_m200c(rhoc, rs, h=0.679, delta=200.):
    """Converts the central density and scale radius to concentration and virial mass"""
    def rho_ratio_of_c(c): # mean enclosed density at r200c in units of rhoc
        return 3*(-c/(c + 1) + np.log(c + 1))/c**3
    
    G = 43.0071057317063 * 1e-10  #  Grav. constant in Mpc (km/s)^2 / Msol
    rhocrit = 3.0 / (8.0 * np.pi * G) * (1e2*h)**2

    # Function that has zero-point at the correct concentration
    def f(c): return rho_ratio_of_c(c)*rhoc - delta*rhocrit

    assert rhoc > rhocrit, f"The central density seems unreasonably low... rhoc/rhocrit={rhoc/rhocrit}"

    c = brentq(f, 1e-10, 1e10) # Find the zero-point
    M = 4*np.pi*rhoc*rs**3 * (np.log(1+c) - c/(1+c))
    
    return c, M


class NFWProfile(RadialProfile):
    def __init__(self, conc, m200c=None, r200c=None, h=0.679, anisotropy=0., rminrs=1e-15, rmaxrs=1e15, **config):
        """Set up an NFW profile with a given mass and concentration
        
        conc : concentration -- so that the scale radius is rs = r200c / c
        m200c : virial mass of the halo in units of Msol, r200c can be 
                provided instead
        r200c : virial radius of the halo in units of Mpc, m200c can be
                provided instead
        h : reduced hubble parameter. Set to 1 to use units where masses
            are measured in Msol/h and lengths in units of Mpc/h
        """

        self.conc = conc
        
        if m200c is not None:
            self.m200c = m200c
            self.r200c = RvirOfMvir(m200c, h=h)
        elif r200c is not None:
            assert m200c is None, "You provided both m200c and r200c, please only provide one"
            self.r200c = r200c
            self.m200c = MvirOfRvir(r200c, h=h)
        else:
            raise ValueError("You have to provide either m200c or r200c")

        self.rs = self.r200c / self.conc
        super().__init__(anisotropy=anisotropy, rmin=rminrs*self.rs, rmax=rmaxrs*self.rs, **config)

        self.rhoc = self.m200c/(4.*np.pi*self.rs**3 * (np.log(1.+self.conc) - self.conc/(1.+self.conc)))
        self.phi0 = - 4.*np.pi*self.G*self.rhoc*self.rs**2
        
        self.phasespace_initialized =  False

    @classmethod
    def from_rhoc_rs(cls, rhoc, rs, rminrs=1e-15, rmaxrs=1e15, h=0.679, **kwargs):
        """Create an NFW profile from characteristic density and scale radius"""
        conc, m200c = rhoc_rs_to_conc_m200c(rhoc, rs, h=h)
        return cls(conc, m200c=m200c, rminrs=rminrs, rmaxrs=rmaxrs, h=h, **kwargs)

    def density(self, r):
        """Density in Msol/Mpc**3"""
        a = r/self.rs

        return self.rhoc/(a * (1 + a)**2 )
    
    def drhodr(self, r):
        """Radial derivative of the density"""
        a = r/self.rs

        return self.rhoc/self.rs * (-(3*a**2 + 4*a + 1)  /(a * (1 + a)**2)**2)
    
    def m_of_r(self, r):
        """The mass contained inside radius r"""
        x = np.array(r) / self.rs
        M0 = 4.*np.pi*self.rs**3*self.rhoc
        
        m = np.zeros_like(r)
        sel = x > 1e-5
        m[sel] = M0 * (np.log(1 + x[sel]) + 1. / (1. + x[sel]) - 1.)
        m[~sel] = 0.5 * M0 * x[~sel]**2
        
        return m
    
    def potential(self, r, zero_at_zero=True):
        """The gravitational  potential. 0 at r -> infty.
        zero_at_zero: if True, norm to phi(r->0)=0. This can be useful
        to avoid problems caused by roundoff errors as r->0"""
        phi = np.zeros_like(r)
        x = np.array(r) / self.rs
        sel = x > 1e-4
        if zero_at_zero:
            phi[sel] = self.phi0 * (np.log(1. + x[sel]) / x[sel] - 1.)
            phi[~sel] = self.phi0 * (- x[~sel]/2. + x[~sel]**2/3.)
        else:
            phi[sel] = self.phi0 * np.log(1. + x[sel]) / x[sel]
            phi[~sel] = self.phi0 * (1. - x[~sel]/2. + x[~sel]**2/3.)
        return phi
    
    def phimax(self):
        return 0.
    
    def r0(self):
        """The virial radius"""
        return self.r200c
    
    def daccdr(self, r):
        """Radial derivative of the acceleration"""
        a = r/self.rs
        log_deriv = (1./self.rs/(1. + a))
        daccr_dr = -self.phi0*self.rs * (2*r**-3 * np.log(1. + a) - 2*r**-2 * log_deriv
                          + r**-1 * (-1./self.rs**2/(1. + a)**2))
        
        return daccr_dr
    
    def to_string(self):
        return "nfw_conc=%.5g_r200c=%.5e_m200c=%.5e" % (self.conc, self.r200c, self.m200c)
    
    def to_dict(self):
        d = {}
        
        d["conc"] = self.conc
        d["r200c"] = self.r200c
        
        return d

class EinastoProfile(RadialProfile):
    def __init__(self, rhom2=1., rm2=1., alpha=0.16):
        """Set up an Einasto profile
        
        rm2 : radius where the slope is -2
        rhom2 : Density at the radius where the slope is -2
        alpha : curvature parameter of the Einasto Profile. Wang et al (2020) suggest 0.16
        """
        super().__init__()
        
        self.rhom2 = rhom2
        self.rm2 = rm2
        self.alpha = alpha

    def density(self, r):
        """Density in Msol/Mpc**3"""
        
        return self.rhom2*np.exp(- 2./self.alpha * ((r/self.rm2)**self.alpha - 1.))
    
    def drhodr(self, r):
        """Radial derivative of the density"""
        assert 0
    
    def m_of_r(self, r):
        """The mass contained inside radius r"""
        from scipy.special import gamma, gammaincc
        
        N = self.rhom2 * np.exp(2./self.alpha)
        A = self.rm2**(-self.alpha) / self.alpha
        alpha = self.alpha
        #rho = N * np.exp(-2.*A*r**self.alpha)
        
        def gamma_wolfram(a, x): # incomplete gamma function as defined in wolfram alpha
            return gammaincc(a, x) * gamma(a)
        
        def m_indef(r):
            return - 4.*np.pi* N/alpha * ( 8**(-1./alpha) * r**3 * (A*r**alpha)**(-3/alpha)
                                          * gamma_wolfram(3./alpha, 2*A*r**alpha))
        
        return m_indef(r) - m_indef(self.r0()*1e-15)
    
    def potential(self, r, zero_at_zero=False):
        """The gravitational  potential. 0 at r -> infty.
        zero_at_zero: if True, norm to phi(r->0)=0. This can be useful
        to avoid problems caused by roundoff errors as r->0"""
       
        assert 0
    
    def r0(self):
        """The scale radius"""
        return self.rm2

class PowerlawProfile(RadialProfile):
    def __init__(self, alpha=None, anisotropy=0., gamma=None, rhoc=1.):
        """
        Initialize a powerlaw profile with the given parameters.

        density profile: rho = rhoc * r**(-alpha)
        phase space profile: f(E,L) ~ E**-gamma L**-beta
        where alpha is the slope and beta the anisotropy

        free variables: either alpha or gamma, and beta
        """
        super().__init__(phase_space=None)
        
        def gamma_of_alpha_beta(alpha, beta=0.):
            return (3 - 0.5*alpha - 4.*beta + alpha*beta)/(2. - alpha)

        def alpha_of_gamma_beta(gamma, beta=0.):
            return (2.*gamma + 4*beta - 3.)/(gamma + beta - 0.5)

        if alpha is None and gamma is None:
            raise ValueError("Please provide either alpha or gamma")

        if alpha is None:
            alpha = alpha_of_gamma_beta(gamma, anisotropy)
        elif gamma is None:
            assert anisotropy < alpha/2.
            gamma = gamma_of_alpha_beta(alpha, anisotropy)
        else:
            raise ValueError("Please provide either alpha or gamma, not both")
        
        assert alpha < 2., "Potential is not well defined for alpha>=2, have to check this"
        
        self.alpha = alpha
        self.anisotropy = anisotropy
        self.gamma = gamma

        # Normalization constants:
        self.rhoc = rhoc
        self.phic = 4.*np.pi * self.G * self.rhoc / ( (3. - self.alpha) * (2. - self.alpha)  )

        Cby = 2**(1.5 - anisotropy) * np.pi**1.5 * GammaF(1. - anisotropy) * GammaF(gamma + anisotropy - 1.5) / GammaF(gamma)
        
        self.fc = self.rhoc / Cby / self.phic**(-gamma-anisotropy+1.5)

        def f_of_el(e, l):
            return self.fc * e**-self.gamma * l**(-2.*self.anisotropy)

        self.set_phase_space(AnalyticPhaseSpace(f_of_el=f_of_el, anisotropy=self.anisotropy))

    @classmethod
    def from_rscale(cls, rscale, rhoscale, alpha=None, anisotropy=None):
        """Alternative initilization so that rho(rs) = rhoscale"""
        rhoc = rhoscale * rscale**alpha
        
        return cls(alpha=alpha, anisotropy=anisotropy, rhoc=rhoc)

    def density(self, r):
        return self.rhoc*r**(-self.alpha)
    
    def m_of_r(self, r):
        return 4.*np.pi * self.rhoc / (3. - self.alpha) * r**(3.-self.alpha)
    
    def potential(self, r, zero_at_zero=True):
        return self.phic * r**(2.-self.alpha)

    def r0(self):
        return 1.0

    def to_string(self):
        return "alpha=%.3f_beta=%.5e_rhoc=%.5e" % (self.alpha, self.anisotropy, self.rhoc)
    
    def to_dict(self):
        d = {}
        
        d["alpha"] = self.alpha
        d["beta"] = self.anisotropy
        d["rhoc"] = self.rhoc
        
        return d

    
class IsothermalSphere(RadialProfile):
    def __init__(self, rho0=1., r0=1.):
        """Set up an Isotrhermal Sphere profile
        
        https://arxiv.org/pdf/2011.07077.pdf
        """
        super().__init__()
        
        self.rho0 = rho0
        self.rad0 = r0
        self.v0 = np.sqrt(4.*np.pi*self.rho0*self.rad0**2*self.G)

    def density(self, r):
        return self.rho0 * (r/self.rad0)**-2
    
    def m_of_r(self, r):
        """The mass contained inside radius r"""
        return 4.*np.pi*self.rho0*self.rad0**2 * r
    
    def potential(self, r, zero_at_zero=False):
        return self.v0**2 * np.log(r/self.rad0)
    
    def r0(self):
        """The scale radius"""
        return self.rad0
    
    def daccdr(self, r):
        """Radial derivative of the acceleration"""

        return self.v0**2 /r**2

class PlummerProfile(RadialProfile):
    def __init__(self, M=1, a=1):
        """Set up a Plummer profile
        """
        super().__init__(phase_space=None)
        
        self.M = M
        self.a = a

        self.phi0 = - self.G * self.M / self.a

        def f_of_e(e):
            e = e+self.phi0
            f = np.zeros_like(e)
            f[e < 0] = 24.* np.sqrt(2.) / (7. * np.pi**3) * self.a**2 / (self.G**5 * self.M**4) * (-e[e < 0])**3.5
            
            return f

        self.set_phase_space(AnalyticPhaseSpace(f_of_e=f_of_e, anisotropy=0.))

    def density(self, r):
        return 3*self.M/(4*np.pi) * (1 + (r/self.a)**2)**-2.5
    
    def m_of_r(self, r):
        return r**3 / (r**2 + self.a**2)**1.5 * self.M

    def potential(self, r, zero_at_zero=True):
        phi = - self.G * self.M / np.sqrt(r**2 + self.a**2)

        if zero_at_zero:
            phi -= self.phi0
            # At small radii put expansion to avoid cancelation erros
            phi_expansion = 0.5 * self.G * self.M * r**2 / self.a**3
            phi = np.where(r < 1e-3*self.a, phi_expansion, phi)
        return phi
    
    def r0(self):
        return self.a
    
    def phimax(self):
        return 0.



class RadialTidalProfile(RadialProfile):
    def __init__(self, alpha=0.):
        """A repulsive potential of form phi = -0.5*alpha*r**2
        
        alpha : eigenvalue of the tidal tensor. alpha>0 corresponds to a field
                stretching the mass distribution and leading to disruption.
                alpha < 0 does not make much sense in this context"""

        super().__init__(phase_space=None)

        if alpha < 0:
            raise ValueError("Probably you want to use a positive alpha... If you don't, just comment this!")
        
        self.alpha = alpha
        self.rhoalpha = - 3.* self.alpha / (4.*np.pi*self.G)
        self.warned = False
        
    def density(self, r):
        """Density in Msol/Mpc**3"""
        return self.rhoalpha * np.ones_like(r)
    def drhodr(self, r):
        """Radial derivative of the density"""
        return np.zeros_like(r)
    def m_of_r(self, r):
        """The mass contained inside radius r"""
        return - self.alpha/self.G * r**3
    def potential(self, r, zero_at_zero=True):
        """The gravitational potential"""
        return - 0.5 * self.alpha* r**2
    def daccdr(self, r):
        """The radial derivative of the acceleration"""
        if not self.warned:
            print("Warning: this function was wrong previously... I have to check some things again! ")
            self.warned = True
        # return 3.*self.alpha/self.G * r**2 -- previous wrong version...
        return self.alpha * np.ones_like(r)
    def to_string(self):
        return "tid_alpha=%.5e" % self.alpha

def find_boundary(profile, getphi=False, rguess=None, maxiter=100, eps=1e-4, warning=True, mode="phimax"):
    """Finds a special boundary (e.g. tidal radius or vmax radius) of a RadialProfile
    
    The tidal radius is a saddle point in the potential or the zero-point of
    the radial acceleartion. This function assumes that the profile has
    profile.accr(r -> 0) < 0  and  profile.accr(r -> inf) > 0. So it is
    attractive at close range and repulsive at long range -- e.g. due to
    a tidal field
    
    profile : the profile, has to be an instance of RadialProfile
    getphi : if true the result will also include the saddle-point potential
    rguess : a guess of the radius. A good guess might reduce number of evaluations
    maxiter : maximum number of iterations of the binary earch
    eps : relative accuracy of the radius when to stop
    warning : if True, throws a warning if rtid->infty
    mode : can be "phimax" (for tidal radius) or "lmax", "vmaxself" for getting rmax
    
    returns : rtid or (rtid, phitid) if getphi is set
    """
    assert 0
    
    if mode == "phimax":
        def acc(r):
            return profile.accr(r)
    elif mode == "vmax":
        #vcirc = sqrt(G*M(r)/r)
        #dv/dr = 0.5 (G*M(r)/r)^{3/2} * (G*dMdr/r - G*M(r)/r**2)
        # dMdr*r - M(r)
        # dMdr = 4 pi rho(r) * r**2
        def acc(r):
            return -4.*np.pi*profile.density(r)*r**3 + profile.m_of_r(r)
    elif mode == "lmax":
        def acc(r):
            #return - 1./profile.vcirc(r) * (-3.*r*profile.accr(r) - profile.daccdr(r)*r**2)/2.
            return - (-3.*r*profile.accr(r) - profile.daccdr(r)*r**2)/2.
    else:
        assert 0
            
    
    if rguess is None:
        r = profile.r0()
    else:
        r = rguess

    # Find a radius where the sign of the acceleration is positive and negative
    if acc(r) == 0.:
        return r
    elif acc(r) > 0.:
        rpos = r
        for i in range(0, maxiter):
            r = r / 2.
            if acc(r) < 0.:
                rneg = r
                break
            if i == maxiter-1:
                raise ValueError("I couldn't find any radius where the profile is attractive")
    else: #  profile.accr(r) < 0.
        rneg = r
        for i in range(0, maxiter):
            r = r * 2.
            if acc(r) > 0.:
                rpos = r
                break
            if i == maxiter-1:
                if warning:
                    print("Warning, I couldn't find any radius where the profile is repulsive, rtid=infty")
                if getphi:
                    return np.infty, 0.
                else:
                    return np.infty
                #raise ValueError("I couldn't find any radius where the profile is repulsive")
    
    for i in range(0, maxiter):
        r = 0.5*(rpos + rneg)
        if acc(r) > 0.:
            rpos = r
        else:
            rneg = r
            
        if (rpos-rneg)/r < eps:
            break
            
    if getphi:
        return r, profile.potential(r)
    else:
        return r