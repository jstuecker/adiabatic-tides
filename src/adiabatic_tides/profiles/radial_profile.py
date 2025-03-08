import numpy as np
from ..phasespace import PhaseSpace, EddingtonPhaseSpace, AnalyticPhaseSpace, ActionMap, InterpolatorActionMap
from ..config import Config
from .. import numerics
import functools
import copy

def deprecated(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        raise DeprecationWarning("This function is deprecated, please use a different one")
        return func(*args, **kwargs)
    return new_func

class RadialProfile():
    # A default config... this may be overwritten by subclasses to provide a more specific default
    default_config : Config = Config() 

    def __init__(self, phase_space="eddington", anisotropy=0., config = None):
        """This is an abstract class defining the interface of RadialProfiles
        
        config : can be a yaml file name, a Config object or a dictionary
            if it is a dictionary, the subconfigs 
            "units", "general", "eddington", "actions", "adiabatic", "sampling"
            can be specified through their respective config classes or as dictionaries

        To see config options print(profile.cfg) and to modify them e.g. use
        profile.cfg.eddington.nintegrate = 200 
        profile.cfg.scale_accuracy(2.0)
        Only units cannot be modified after initialization
        """

        self.cfg = Config.flexible_init(config, default=self.default_config)
        self.G = self.cfg.G()

        self.set_phase_space(phase_space, anisotropy=anisotropy)
        
        self.action_map = InterpolatorActionMap(self)
    
    #----------- Abstract methods  --------------#
    # These methods have to be implemented by any subclass

    def density(self, r):
        """Abstract: The density profile"""
        raise NotImplementedError("This is an abstract class, please implement a subclass")

    def m_of_r(self, r):
        """Abstract: The mass contained inside radius r"""
        raise NotImplementedError("This is an abstract class, please implement a subclass")

    def potential(self, r, zero_at_zero=False):
        """Abstract: The gravitational potential. By default normed to 0 at infinity"""
        raise NotImplementedError("This is an abstract class, please implement a subclass")
    
    # Optional methods that can be implemented by subclasses
    def to_dict(self):
        raise NotImplementedError("This is an abstract class, please implement a subclass")
    
    @classmethod
    def from_dict(cls, d):
        raise NotImplementedError("This is an abstract class, please implement a subclass")
    
    # ------------------ Geometrical Scales ------------------ #

    # Numerical scales that are defined by the configuration
    def rmin(self):
        return self.cfg.general.rmin
    
    def rmax(self):
        return self.cfg.general.rmax
    
    # Scales that depend on the potential structure
    def rtid(self):
        """Tidal radius corresponding to the maximum of the potential"""
        opt = numerics.search.maximize_scalar(lambda r: self.potential(r), (self.rmin(), self.rmax()))
        return opt.x
    
    def rlmax(self):
        """Radius with the maximum possible angular momentum"""
        opt = numerics.search.maximize_scalar(lambda r: self.m_of_r(r)*r, (self.rmin(), self.rapo_max()))
        return opt.x
    
    def rmax_vmax(self):
        """Radius and velocity where the circular velocity is maximal"""
        opt = numerics.search.maximize_scalar(lambda r: self.m_of_r(r)/r, (self.rmin(), self.rmax()))
        return opt.x, self.vcirc(opt.x)
    
    def rapo_max(self):
        """The maximal radius at which orbital apo-centers can lie
        Either corresponds to rmax or to the tidal radius, depending on the profile
        """
        rtid, rmax = self.rtid(), self.rmax()
        return rtid if rtid < rmax else rmax
    
    #----------- Potential related functions ----------# 
    # These functions follow directly from the ones above and do not
    # need to be implemented by subclasses

    def accr(self, r):
        """Radial Acceleration (negative means pull towards center)"""
        return  -self.G * self.m_of_r(r) / r**2

    def daccdr(self, r):
        """ accr = -G m(r) / r^2
        daccr/dr = 2 G m(r) / r^3 - G m'(r) / r^2 = -2 G accr(r) / r - G rho(r) 4 pi
        """
        return -2 * self.accr(r) / r - 4.*np.pi * self.density(r) * self.G
    
    def tdyn(self, r):
        """Dynamical Time-scale r / vcirc(r)"""
        return r / self.vcirc(r)
    
    def tcirc(self, r, inyears=False):
        """Time needed for a circular orbit at radius r. Default unit is (mpc/km) s
        
        inyears : transform to years
        """
        if inyears:
            assert 0, "***Have to fix this"
            si_parsec, si_year = 3.085677581491367e+16, 31536000.0
            tunit = (1e6*si_parsec / (1e3)) / si_year
            return(2.*np.pi*r / self.vcirc(r) * tunit)
        else:
            return 2.*np.pi*r / self.vcirc(r)
    
    def vcirc(self, r):
        """Circular velocity at radius r"""
        return np.sqrt(np.clip(-self.accr(r) * r, 0., None))

    def tidal_tensor(self, x, x0=(0.,0.,0.)):
        """The Tidal Tensor Tij = - d2phi/(dxi dxy)"""
        dx = x-np.array(x0)
        r = np.sqrt(np.sum(dx**2, axis=-1))
        
        accr = self.accr(r)
        daccr_drr = self.daccdr(r)
        
        tid = np.zeros(x.shape[:-1] + (3,3))

        for i in range(0,3):
            for j in range(0,3):
                if i == j:
                    # this is  d(xi/r)/dxj
                    der_xr = 1. / r - x[...,i]**2/r**3
                else:
                    der_xr = -x[...,i]*x[...,j]/r**3

                tid[...,i,j] = accr * der_xr + daccr_drr * (x[...,i]/r * x[...,j]/r)

        return tid

    def tidal_eigval(self, r):
        """Eigenvalues of the Tidal tensor"""
        accr = self.accr(r)
        daccr_drr = self.daccdr(r)
        
        lam_r = daccr_drr
        lam_phi = accr / r

        return np.array((lam_r, lam_phi, lam_phi))
    
    def E_L_of_rperi_rapo(self, rperi, rapo):
        "Maps peri- and apo-center radii to energy and angular-momentum"
        return numerics.utility.e_l_of_rp_ra(self.potential, rperi, rapo, accr=self.accr)
    
    def posvel_to_rEL(self, pos, vel):
        """Calculates the radius, energy and angular momentum of particles
        
        pos : positions of the particles
        vel : velocities of the particles
        
        returns : (r, E, L)  with  the radius, energy and angular momentum
        """
        r = np.sqrt(np.sum(pos**2, axis=-1))
        E = self.potential(r) + 0.5*np.sum(vel**2, axis=-1)
        L = np.sqrt(np.sum(np.cross(pos, vel)**2, axis=-1))
        
        return r, E, L
    
    #----------- Search related functions --------------#
    # These functions help to find e.g. a radius where a given condition is true
    
    def _search_radius(self, f, rlow=None, rup=None, niter=None, logspace=True, invalid_val=np.nan):
        rlow = rlow or self.rmin()
        rup = rup or self.rmax()
        niter = niter or self.cfg.actions.niter_pa

        return numerics.search.ridders_method(f, rlow, rup, mode="positive", niter=niter, logspace=logspace, invalid_val=invalid_val)
    
    def r_of_potential(self, phi):
        "Find radius where the potential is phi (if non-monotoneous considering only ascending part)"
        return self._search_radius(lambda r: self.potential(r) - phi, rup=self.rapo_max())

    def r_of_ecirc(self, ecirc, mode="asc"):
        "Find radius where the circular energy is ecirc (if non-monotoneous mode can be 'asc' or 'desc')"
        def f(r): return self.potential(r) + 0.5*self.vcirc(r)**2 - ecirc
        rlmax = self.rlmax() # radius where circular energy is maximal

        if mode == "asc":
            rup = self.rmax() if np.isnan(rlmax) else rlmax
            return self._search_radius(f, rup=rup)
        elif mode == "desc":
            if np.isnan(rlmax):
                raise ValueError("Cannot search for descending part, as there is no maximum")
            return self._search_radius(f, rlow=rlmax)
        else:
            raise ValueError("Unknown mode %s" % mode)
        
    def r_of_lcirc(self, lcirc, mode="asc"):
        def f(r): return self.vcirc(r)*r - lcirc
        rlmax = self.rlmax() # radius where circular angular momentum is maximal

        if mode == "asc":
            rup = self.rmax() if np.isnan(rlmax) else rlmax
            return self._search_radius(f, rup=rup)
        elif mode == "desc":
            if np.isnan(rlmax):
                raise ValueError("Cannot search for descending part, as there is no maximum")
            return self._search_radius(f, rlow=rlmax, rup=self.rtid())
        else:
            raise ValueError("Unknown mode %s" % mode)

    def radius_of_f(self, f, l=1., rlow=None, rup=None):
        "Radius where the phase space density f(phi(r), l) = f"
        def func(r): return np.log(self.f_of_el(self.potential(r), l)/f)
        return self._search_radius(func, rlow=rlow or self.rmin()*2, rup=rup or self.rmax()/2)
    
    def rperi_rapo_of_r_e_l(self, r, e, l, search_method=None, rlow=None, rup=None, niter=None, return_err=False, exceptions=True):
        def energy_permitted(r):
            return e - 0.5*l**2/r**2 - self.potential(r)

        if niter is None: niter = self.cfg.actions.niter_pa
        if rlow is None: rlow = self.rmin()
        if rup is None: rup = self.rmax()
        if search_method is None: search_method = self.cfg.actions.search_method

        if search_method == "binary":
            rp = numerics.search.vectorized_binary_search(energy_permitted, rlow*np.ones_like(r), r, niter=niter, return_err=return_err, exceptions=exceptions, xfallback=r)
            ra = numerics.search.vectorized_binary_search(energy_permitted, r, rup*np.ones_like(r), niter=niter, return_err=return_err, exceptions=exceptions, xfallback=r)
        elif search_method == "ridders":
            rp = numerics.search.ridders_method(energy_permitted, rlow*np.ones_like(r), r, mode="positive", niter=niter, logspace=True)
            ra = numerics.search.ridders_method(energy_permitted, r, rup*np.ones_like(r), mode="positive", niter=niter, logspace=True)
        else:
            raise ValueError("Unknown mode %s" % search_method)
        
        return rp, ra
    
    #----------- Phase Space Distribution --------------#

    def set_phase_space(self, phase_space="eddington", anisotropy=0.):
        if phase_space == "eddington":
            if anisotropy is None:
                self.phase_space = None
            else:
                self.phase_space = EddingtonPhaseSpace(self.density, self.potential, self.cfg.general, self.cfg.eddington, anisotropy=anisotropy)
        else:
            self.phase_space = phase_space
        
        if self.phase_space is not None:
            self.anisotropy = self.phase_space.anisotropy
        else:
            self.anisotropy = anisotropy

    def f_of_e(self, E):
        assert self.phase_space is not None, "No phase space defined"
        return self.phase_space.f_of_e(E)
    
    def f_of_el(self, E, L):
        assert self.phase_space is not None, "No phase space defined"
        return self.phase_space.f_of_el(E, L)
    
    def f_of_rperi_rapo(self, rp, ra):
        E, L = self.E_L_of_rperi_rapo(rp, ra)
        return self.f_of_el(E, L)
    
    def f_of_jl(self, j, l):
        rp,ra = self.action_map.rp_ra_of_jl(j, l)
        return self.f_of_rperi_rapo(rp, ra)

    #----------- Action Calculation Methods --------------#

    def radial_action_of_r_e_l(self, r, e, l):
        """Numerically infer the radial action Jr as in Binney and Tremaine (2008) eq 3.224"""
        rp, ra = self.rperi_rapo_of_r_e_l(r, e, l)
        return self.radial_action_of_rp_ra(rp, ra)

    def radial_action_of_rp_ra(self, rp, ra, nintegrate=None):
        """Numerically infer the radial action Jr as in Binney and Tremaine (2008) eq 3.224"""
        nintegrate = nintegrate or self.cfg.actions.nintegrate
        return numerics.integrate.calculate_radial_action_tanh_peri_apo(self.potential, rp, ra, nintegrate=nintegrate)

    def radial_period_of_rp_ra(self, rperi, rapo, nintegrate=None):
        """Numerically infer the radial orbital period time"""
        nintegrate = nintegrate or self.cfg.actions.nintegrate
        djde = numerics.integrate.calculate_dj_de_tanh_peri_apo(self.potential, rperi, rapo, nintegrate=nintegrate)
        return djde*2.*np.pi
    
    #----------- Integrals and Moments --------------#

    def compute_pa_space_integral(self, r, f_of_rp_ra=None, vrmoment=0, vtmoment=0, vmoment=0, nintegrate=40):
        """Integrates a function over velocity-space through peri-apo-space discretization
        f_of_rp_ra : the phase space density (dM/d3x/d3v) with peri and apo centers as arguments
        """
        if f_of_rp_ra is None:
            f_of_rp_ra = self.f_of_rperi_rapo

        is_limited = numerics.search.profile_is_limited(self.accr, rpmin=self.rmin())
        if is_limited:
            rperi, rapo, rlmax, rtid, ramax_of_rp = numerics.interpolate.define_paspace_boundaries(self.potential, self.accr, self.daccdr, rpmin=self.rmin())
            rperirange, raporange = (self.rmin(), rlmax), (self.rmin(), ramax_of_rp)
        else:
            rperirange, raporange = (self.rmin(), np.infty), (self.rmin(), np.infty)

        return numerics.integrate.integrate_f_paspace(f_of_rp_ra, self.potential, self.accr, r, N=nintegrate,
                                                      rperirange=rperirange, raporange=raporange, 
                                                      vrmoment=vrmoment, vtmoment=vtmoment, vmoment=vmoment)
    
    def compute_vr2_vt2(self, r, nintegrate=40):
        """Returns the velocity dispersions vr2 and vt2 as a function of radius"""
        rho_x_vr2 = self.compute_pa_space_integral(r, vrmoment=2, nintegrate=nintegrate)
        rho_x_vt2 = self.compute_pa_space_integral(r, vtmoment=2, nintegrate=nintegrate)
        rho = self.compute_pa_space_integral(r)

        return rho_x_vr2/rho, rho_x_vt2/rho
    
    def integral_density_squared(self, rmin=None, rmax=None, nintegrate=100):
        if rmin is None: rmin = self.rmin()
        if rmax is None: rmax = self.rmax()

        def integrand(r):  return self.density(r)**2 * 4*np.pi*r**2
        
        if np.min(rmin) == 0.:
            return numerics.integrate.integrate_double_exponential_a_b(integrand, rmin, rmax, N=nintegrate)
        else:
            return numerics.integrate.integrate_exp_double_exp_a_b(integrand, rmin, rmax, N=nintegrate)
    
    @deprecated
    def density_of_states(self, energy):
        """Returns the density of states g(E) associated with some energy.
        
        This is normalized so that for an isotropic profile, the actual
        number of particles with energy level E will be proportional to
        f(e) * g(e) where f(e) is the phase space density.
        
        This is fiven by 
        (4 pi)**2 integral( r**2 sqrt(2(E-phi)))
        The integral that has to be evaluated is very similar to the action
        integral for L = 0: 
        J(E, L=0) = integral(sqrt(2(E-phi)) / pi)  
        (which doesn't have the factor r**2), so that we use  the same routine
        for evaluating it, just with a slight modification
        """
        
        L = np.zeros_like(energy)
        
        integral = self.radial_action((energy, L), rpow=2)
        
        return (integral*np.pi) * (4.*np.pi)**2
    
    @deprecated
    def vdispr2_via_jeans_integration(self, logr=None, anisotropy=0., density=None):
        """Obtain the radial velocity dispersion squared through integration of the 1st Jeans equation
        
        logr : can provide integration points optionally, should be decreasing and log(radius)
        anisotropy: anisotropy parameter beta = 1 - (sigma_phi**2 + sigma_t**2) / 2sigma_r**2 
                    can be a function of radius
        density: if provided use a different density distribution, than the one which generates the potential
        
        returns: r, sigmar2   radii and radial velocity dispersion
        """
        print("This function is outdated, better use 'compute_velocity_dispersions' instead")
        
        if callable(anisotropy):
            faniso = anisotropy
        elif np.isscalar(anisotropy):
            def faniso(r):
                return anisotropy
        else:
            raise ValueError("Did not understand type for anisotropy (should be scalar or function)")
            
        if density is None:
            def density(r):
                return self.density(r)

        def drhosigr2_dlogr(logr, sigr2=0.):
            r = np.exp(logr)
            return (density(r) * self.accr(r) - 2.*density(r) / r * sigr2 * faniso(r)) * r

        if logr is None:
            logr = np.linspace(np.log(self.rmax()*1e3),np.log(self.rmin()), 10000)
        else:
            assert np.all(logr[1:] <= logr[:-1]), "logr has to be descending"
        rhosigr2 = np.zeros_like(logr)

        for i in range(1,len(logr)):
            dlogr = logr[i] - logr[i-1]

            sigr2=rhosigr2[i-1]/density(np.exp(logr[i]-1))

            rhosigr2[i] = rhosigr2[i-1] +  drhosigr2_dlogr(logr[i-1], sigr2)*dlogr
            
        return np.exp(logr[::-1]), (rhosigr2/density(np.exp(logr)))[::-1]
    
    #----------- Sampling Methods --------------#

    def sample_particles(self, ntot=10000, mode="r_e_l_vr_m", rmax=None, rpmin=None, rpmax=None, ninterp=None, nintegrate=None, nsteps_metropolis=None):
        """ Samples particles radii, energies, angular momenta, radial velocities and masses
        using a metropolis algorithm for the (E,L | r) sampling. This is not the fastest
        possibility, but it is very robust and works for every profile, including anisotropic
        ones

        --- important parameters ---
        ntot : number of particles
        rmax : maximal radius to sample
        rpmin : If given, all particles have a peri-center rp > rpmin
        rpmax : If given, all particles have a peri-center rp < rpmax

        get_rho: If true, the density profile is returned as well

        --- numerical parameters ---
        ninterp: number of interpolation points for the denisty profile
        nintegrate : number of integration points for the energy integral (200 is usually already very precise)
        nsteps_chain : number of steps in the metropolis chain (to be safe use 32 or higher)
                       sampling time scales linear with this parameter
        """
        rmax = rmax or self.rmax()
        rpmin = rpmin or self.rmin()
        rpmax = rpmax or rmax

        nintegrate = nintegrate or self.cfg.sampling.nintegrate
        ninterp = ninterp or self.cfg.sampling.ninterp
        nsteps_metropolis = nsteps_metropolis or self.cfg.sampling.nsteps_metropolis

        ri = np.logspace(np.log10(rpmin), np.log10(rmax), ninterp)

        p = {}

        rho = numerics.integrate.integrate_f_paspace(self.f_of_rperi_rapo, self.potential, self.accr, ri, N=nintegrate, rperirange=(rpmin, rpmax))
        p["r"],p["m"] = numerics.sample.sample_rimi_from_density(ri, rho, ntot)

        p["rp"], p["ra"] = numerics.sample.sample_rp_ra_given_r_metropolis_perisplit(self.f_of_rperi_rapo, self.potential, self.accr, p["r"], rperirange=(rpmin, rpmax), nsteps_chain=nsteps_metropolis)
        p["e"],p["l"],p["vr"] = numerics.sample.E_L_vr_from_rp_r_ra(self.potential, p["rp"], p["r"], p["ra"])

        p["rrho"] = ri
        p["rho"] = rho

        if mode == "dict":
            return p
        else:
            res = []
            for key in mode.split("_"):
                assert key in p, "Unknown key %s" % key
                res.append(p[key])
            return res

    def sample_particles_perisplits(self, size_per_split=10000, rpsplits=(None, None), mode="r_e_l_vr_m", rmax=None, flat=True, **kwargs):
        """See sample_r_E_L_vr_m_metropolis for a detailed description of optional keyword parameters

        size_per_split : number of particles in each split
        rpsplits : a list of splitting points
        flat : whether to return particles in form (nsplits, nper_split) or as a flat array
        """
        
        res = []

        for i in range(len(rpsplits)-1):
            res.append(self.sample_particles(size_per_split, mode=mode, rpmin=rpsplits[i], rpmax=rpsplits[i+1], rmax=rmax, **kwargs))

        outputs = []
        ncol = len(res[0])
        for j in range(ncol):
            outputs.append(np.stack([r[j] for r in res], axis=0))
        if flat:
            return [o.flatten() for o in outputs]
        else:
            return outputs

    # ----------- Utility Methods --------------#
    # these help with estimating some important scales etc.

    def rcirc_eta_of_rperi_rapo(self, rperi, rapo):
        """Given a peri and apo-center, finds the radius where a circular orbit
        is possible and the angular-momentum in units of the circular angular momentum
        
        useful for translating results to DASH simulations"""
        e, l = self.E_L_of_rperi_rapo(rperi, rapo)

        rcirc = self.r_of_ecirc(np.atleast_1d(e))
        lcirc = self.vcirc(rcirc)*rcirc

        return rcirc.reshape(np.shape(rperi)), (l/lcirc).reshape(np.shape(rperi))
    
    def rperiapo_of_eta_rcirc(self, rcirc, eta):
        """Given a circular radius and the angular momentum in units of
        the circular angularmomentum, calculates the peri and apo center
        of a corresponding orbit
        
        useful for translating results to DASH simulations"""
        E = self.potential(rcirc) + 0.5*self.vcirc(rcirc)**2
        l = rcirc*self.vcirc(rcirc)*eta

        rperi, rapo = self.rperi((rcirc,E,l)), self.rapo((rcirc,E,l))
        return rperi, rapo

    def effective_pericenter_tidal_eigval(self, r, vcirc_fac=1.):
        """Eigenvalues of the effective tidal tensor at peri-center, when the
        effect of the centrifugal force is included"""
        
        assert np.min(vcirc_fac) >= 1., "vcirc_fac is the ratio between pericenter velocity and circular velocity, has to be >= 1."
        
        lam = self.tidal_eigval(r)
        omega = 2.*np.pi / self.tcirc(r)
        lam[0] += omega**2/vcirc_fac**2
        
        return lam

    def two_body_relaxation_time(self, r, N, modeN="Ntot", lam=None, rsoft=None, rmax=None, rnorm=None):
        """An estimate of the two-body relaxation at a given radius
        
        r : radius
        N : number of particles used in the simulation
        modeN : use 'Ntot' if N is the number of particles inside the normalization radius (e.g. rvir)
                or use 'Nr' if N is the number of particles inside r
        lam : ratio between maximal and minimal radius (can be None if rsoft and rmax are given)
        rsoft : softening (can be None if lam is given)
        rmax : maximal radius (can be None if lam is given)
        rnorm : radius where the particle number is normalized
        """
        if rnorm is None:
            rnorm = self.r0()
        if rmax is None:
            rmax = rnorm
        if lam is None:
            lam = rmax/rsoft
        
        if modeN == "Ntot":
            Nr = self.m_of_r(r) / self.m_of_r(rnorm) * N
        elif modeN == "Nr":
            Nr = N
        else:
            raise ValueError("Unknown modeN = ", modeN)
        
        tdyn = self.tdyn(r)
        
        return 0.1 * Nr / np.log(lam) * tdyn
    
    def radius_phase_space_core(self, dmtype="WDM", h=0.68, omega_dm=0.26, verbose=True, **kwargs):
        """Estimates the size of the core given by the phase space density constraint
        as explained in arxiv:2207.05082 (Delos & White 2022)
        
        dmtype :   can be "WDM" or "WIMP"
        h :        reduced hubble parameter
        omega_dm : dark matter (not full matter) density parameter
        verbose : set to False to suppress warning messages
        
        other kwargs vary depending in "WDM" or "WIMP":
        "WDM":
          mx : mass in keV (1 default)
          gx : degeneracy (1.5 default), see Bode (2001), arXiv:astro-ph/0010389
        "WIMP":
          mx : mass in GeV (100 default)
          Td : decoupling temperature in MeV (30 default)
          ad : scale factor of decoupling. Put to None to approximate from Td
        Note: For the WIMP case I could only reproduce the numbers in arxiv:2207.05082
              up to a few percent accuracy. Therefore, I print a warning here.
        """
        
        if dmtype == "WDM":
            fmax = numerics.utility.fmax_wdm(h=h, omega_dm=omega_dm, G=self.G, **kwargs)
        elif dmtype == "WIMP":
            fmax = numerics.utility.fmax_wimp(h=h, omega_dm=omega_dm, G=self.G, **kwargs)
        else:
            raise ValueError("Unknown dmtype=%s, so far can only handle WDM or WIMP" % dmtype)
            
        return self.radius_of_f(np.atleast_1d(fmax))[0]
    
    def to_string(self):
        raise NotImplementedError("to_string not implemented for this profile, need this for caching etc...")
    
    def __str__(self):
        return f"RadialProfile(anisotropy={self.anisotropy})"
    
    def __repr__(self):
        return self.__str__() + "\n" + self.cfg.__repr__()