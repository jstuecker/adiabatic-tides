import numpy as np
from .import RadialProfile
from ..config import Config, GeneralConfig
from ..phasespace import AnalyticPhaseSpace
from scipy.special import gamma as GammaF, gammaincc
from scipy.optimize import brentq

# Helper functions

def RvirOfMvir(mvir, mode="crit", delta=200., h=0.679, omega_m=0.30, G=43.0071057317063e-10):
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
    rhocrit = 3.0 / (8.0 * np.pi * G) * (1e2*h)**2
    
    if mode == "crit":
        rhoref = rhocrit
    elif mode == "mean":
        rhoref = omega_m * rhocrit
    else:
        raise ValueError("Unknown mode=%s, can be 'crit' or 'mean'" % mode)

    return  np.cbrt(mvir / (rhoref * 4.*np.pi/3. * delta))

def MvirOfRvir(rvir, mode="crit", delta=200., h=0.679, omega_m=0.30, G=43.0071057317063e-10):
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
    rhocrit = 3.0 / (8.0 * np.pi * G) * (1e2*h)**2
    
    if mode == "crit":
        rhoref = rhocrit
    elif mode == "mean":
        rhoref = omega_m * rhocrit
    else:
        raise ValueError("Unknown mode=%s, can be 'crit' or 'mean'" % mode)
        
    return rhoref * 4.*np.pi/3. * delta * rvir**3

def rhoc_rs_to_conc_m200c(rhoc, rs, h=0.679, delta=200., G=43.0071057317063e-10):
    """Converts the central density and scale radius to concentration and virial mass"""
    def rho_ratio_of_c(c): # mean enclosed density at r200c in units of rhoc
        return 3*(-c/(c + 1) + np.log(c + 1))/c**3
    
    rhocrit = 3.0 / (8.0 * np.pi * G) * (1e2*h)**2

    # Function that has zero-point at the correct concentration
    def f(c): return rho_ratio_of_c(c)*rhoc - delta*rhocrit

    assert rhoc > rhocrit, f"The central density seems unreasonably low... rhoc/rhocrit={rhoc/rhocrit}"

    c = brentq(f, 1e-10, 1e10) # Find the zero-point
    M = 4*np.pi*rhoc*rs**3 * (np.log(1+c) - c/(1+c))
    
    return c, M


class NFWProfile(RadialProfile):
    default_config = Config(general=GeneralConfig(rmin=1e-15, rmax=1e15))

    def __init__(self, conc, m200c=None, r200c=None, h=0.679, anisotropy=0., config : Config | None = None):
        """Set up an NFW profile with a given mass and concentration
        
        conc : concentration -- so that the scale radius is rs = r200c / c
        m200c : virial mass of the halo in units of Msol, r200c can be 
                provided instead
        r200c : virial radius of the halo in units of Mpc, m200c can be
                provided instead
        h : reduced hubble parameter. Set to 1 to use units where masses
            are measured in Msol/h and lengths in units of Mpc/h
        """

        super().__init__(anisotropy=anisotropy, config=config)

        self.conc = conc
        
        if m200c is not None:
            self.m200c = m200c
            self.r200c = RvirOfMvir(m200c, h=h) * (self.cfg.units.length/1e6) # Convert from Mpc units
        elif r200c is not None:
            assert m200c is None, "You provided both m200c and r200c, please only provide one"
            self.r200c = r200c
            self.m200c = MvirOfRvir(r200c * (self.cfg.units.length/1e6), h=h) * self.cfg.units.mass
        else:
            raise ValueError("You have to provide either m200c or r200c")

        self.rs = self.r200c / self.conc

        self.cfg.scale_base_radius(self.rs) # Make rmin and rmax be given in units of rs

        self.rhoc = self.m200c/(4.*np.pi*self.rs**3 * (np.log(1.+self.conc) - self.conc/(1.+self.conc)))
        self.phi0 = - 4.*np.pi*self.G*self.rhoc*self.rs**2
        
        self.phasespace_initialized =  False

    @classmethod
    def from_rhoc_rs(cls, rhoc, rs, h=0.679, config=None, **kwargs):
        """Create an NFW profile from characteristic density and scale radius"""
        config = Config.flexible_init(config, cls.default_config)
        rhoc_msol_ov_mpc3 = rhoc * 1e9 * config.units.mass / config.units.length**3
        rs_mpc = rs * (1e6/config.units.length)
        conc, m200c = rhoc_rs_to_conc_m200c(rhoc_msol_ov_mpc3, rs_mpc, h=h)
        return cls(conc, m200c=m200c,  h=h, config=config, **kwargs)

    def density(self, r):
        a = r/self.rs

        return self.rhoc/(a * (1 + a)**2 )
    
    def m_of_r(self, r):
        x = np.array(r) / self.rs
        M0 = 4.*np.pi*self.rs**3*self.rhoc
        
        m = np.zeros_like(r)
        sel = x > 1e-5
        m[sel] = M0 * (np.log(1 + x[sel]) + 1. / (1. + x[sel]) - 1.)
        m[~sel] = 0.5 * M0 * x[~sel]**2
        
        return m
    
    def potential(self, r, zero_at_zero=True):
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

    def to_dict(self):
        d = {}
        
        d["conc"] = self.conc
        d["r200c"] = self.r200c
        d["anisotropy"] = self.anisotropy
        
        return d
    
    @classmethod
    def from_dict(cls, d):
        """Load a state extracted from a previous '.to_dict()' call"""
        return cls.__init__(conc=d["conc"], r200c=d["r200c"], anisotropy=d["anisotropy"])
    
    def __str__(self):
        return "NFWProfile(conc=%.5g, r200c=%.5g, anisotropy=%.5g)" % (self.conc, self.r200c, self.anisotropy)

class EinastoProfile(RadialProfile):
    def __init__(self, rhom2=1., rm2=1., alpha=0.16, anisotropy=0., config : Config | None = None):
        """Set up an Einasto profile
        
        rm2 : radius where the slope is -2
        rhom2 : Density at the radius where the slope is -2
        alpha : curvature parameter of the Einasto Profile. Wang et al (2020) suggest 0.16
        """
        super().__init__(anisotropy=anisotropy, config=config)
        
        self.rhom2 = rhom2
        self.rm2 = rm2
        self.alpha = alpha

    def density(self, r):
        return self.rhom2*np.exp(- 2./self.alpha * ((r/self.rm2)**self.alpha - 1.))

    def m_of_r(self, r):
        N = self.rhom2 * np.exp(2./self.alpha)
        A = self.rm2**(-self.alpha) / self.alpha
        alpha = self.alpha

        def gamma_wolfram(a, x): # incomplete gamma function as defined in wolfram alpha
            return gammaincc(a, x) * GammaF(a)
        
        def m_indef(r):
            return - 4.*np.pi* N/alpha * ( 8**(-1./alpha) * r**3 * (A*r**alpha)**(-3/alpha)
                                          * gamma_wolfram(3./alpha, 2*A*r**alpha))
        
        return m_indef(r) - m_indef(self.r0()*1e-15)
    
    def potential(self, r, zero_at_zero=False):
        raise NotImplementedError("Potential of Einasto profile is not implemented yet")
    
    def __str__(self):
        return f"EinastoProfile(rhom2={self.rhom2:.5g}, rm2={self.rm2:.5g}, alpha={self.alpha:.5g}, anisotropy={self.anisotropy:.5g})"

class PowerlawProfile(RadialProfile):
    def __init__(self, alpha=None, anisotropy=0., gamma=None, rhoc=1., config : Config | None = None):
        """
        Initialize a powerlaw profile with the given parameters.

        density profile: rho = rhoc * r**(-alpha)
        phase space profile: f(E,L) ~ E**-gamma L**-beta
        where alpha is the slope and beta the anisotropy

        free variables: either alpha or gamma, and beta
        """
        super().__init__(phase_space=None, config=config)
        
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

    def to_dict(self):
        d = {}
        
        d["alpha"] = self.alpha
        d["beta"] = self.anisotropy
        d["rhoc"] = self.rhoc
        
        return d
    
    def __str__(self):
        return f"PowerlawProfile(alpha={self.alpha:.5g}, rhoc={self.rhoc:.5g}, anisotropy={self.anisotropy:.5g})"

class IsothermalSphere(RadialProfile):
    def __init__(self, rho0=1., r0=1., config : Config | None = None):
        """Set up an Isotrhermal Sphere profile
        
        https://arxiv.org/pdf/2011.07077.pdf
        """
        super().__init__(config=config)
        
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
    
    def __str__(self):
        return f"IsothermalSphere(rho0={self.rho0:.5g}, r0={self.rad0:.5g})"

class PlummerProfile(RadialProfile):
    def __init__(self, M=1, a=1, config : Config | None = None):
        """Set up a Plummer profile
        """
        super().__init__(phase_space=None, config=config)
        
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

    def __str__(self):
        return f"PlummerProfile(m={self.M:.5g}, a={self.a:.5g})"

class RadialTidalProfile(RadialProfile):
    def __init__(self, tide=0., config : Config | None = None):
        """A repulsive potential of form phi = -0.5*alpha*r**2
        
        alpha : eigenvalue of the tidal tensor. alpha>0 corresponds to a field
                stretching the mass distribution and leading to disruption.
                alpha < 0 does not make much sense in this context"""

        super().__init__(phase_space=None, anisotropy=None, config=config)

        if tide < 0:
            raise ValueError("Probably you want to use a positive alpha... If you don't, just comment this!")
        
        self.tide = tide
        
    def density(self, r):
        return (- 3.* self.tide / (4.*np.pi*self.G)) * np.ones_like(r)
    
    def m_of_r(self, r):
        return - self.tide/self.G * r**3
    
    def potential(self, r, zero_at_zero=True):
        return - 0.5 * self.tide* r**2

    def __str__(self):
        return f"RadialTidalProfile(tide={self.tide:.5g})"

    def __repr__(self):
        return self.__str__()