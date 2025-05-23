import numpy as np
from ..phasespace import PhaseSpace, EddingtonPhaseSpace, AnalyticPhaseSpace, ActionMap, InterpolatorActionMap, ActionMapThroughLLines
from ..config import Config, time_in_years
from .. import numerics
import functools
from functools import partial
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
        
        self.action_map = ActionMapThroughLLines(self)
    
    #----------- Abstract methods  --------------#
    # These methods have to be implemented by any subclass

    def density(self, r, component="total"):
        """Abstract: The density profile"""
        raise NotImplementedError("This is an abstract class, please implement a subclass")

    def m_of_r(self, r, component="total"):
        """Abstract: The mass contained inside radius r"""
        raise NotImplementedError("This is an abstract class, please implement a subclass")

    def potential(self, r, zero_at_zero=False, component="total"):
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
    def rmin(self, component="total"):
        return self.cfg.general.rmin
    
    def rmax(self, component="total"):
        return self.cfg.general.rmax
    
    # Scales that depend on the potential structure
    def rtid(self):
        """Tidal radius corresponding to the maximum of the potential"""
        opt = numerics.search.maximize_scalar(lambda r: self.potential(r, component="total"), (self.rmin(), self.rmax()))
        return opt.x
    
    def rlmax(self):
        """Radius with the maximum possible angular momentum"""
        opt = numerics.search.maximize_scalar(lambda r: self.m_of_r(r, component="total")*r, (self.rmin(), self.rapo_max()))
        return opt.x
    
    def rmax_vmax(self, component="total"):
        """Radius and velocity where the circular velocity is maximal"""
        rmax = numerics.search.maximize_scalar(lambda r: self.m_of_r(r, component=component)/r, (self.rmin(), self.rmax())).x
        if(np.isfinite(rmax)):
            return rmax, self.vcirc(rmax, component=component)
        else:
            return np.inf, np.inf
        
    def rperi_max(self):
        """The maximal radius at which orbital peri-centers can lie
        Either corresponds to rmax or to the rlmax
        """
        rlmax, rmax = self.rlmax(), self.rmax()
        return rlmax if rlmax < rmax else rmax
    
    def rapo_max(self):
        """The maximal radius at which orbital apo-centers can lie
        Either corresponds to rmax or to the tidal radius, depending on the profile
        """
        rtid, rmax = self.rtid(), self.rmax()
        return rtid if rtid < rmax else rmax
    
    #----------- Potential related functions ----------# 
    # These functions follow directly from the ones above and do not
    # need to be implemented by subclasses

    def accr(self, r, component="total"):
        """Radial Acceleration (negative means pull towards center)"""
        return  -self.G * self.m_of_r(r, component=component) / r**2

    def daccdr(self, r, component="total"):
        """ accr = -G m(r) / r^2
        daccr/dr = 2 G m(r) / r^3 - G m'(r) / r^2 = -2 G accr(r) / r - G rho(r) 4 pi
        """
        return -2 * self.accr(r, component=component) / r - 4.*np.pi * self.density(r, component=component) * self.G
    
    def tdyn(self, r, component="total"):
        """Dynamical Time-scale r / vcirc(r)"""
        return r / self.vcirc(r, component=component)
    
    def tcirc(self, r, inyears=False, component="total"):
        """Time needed for a circular orbit at radius r in code-units
        
        inyears : transform to years
        """
        if inyears:
            return time_in_years(2.*np.pi*r / self.vcirc(r, component=component), self.cfg.units)
        else:
            return 2.*np.pi*r / self.vcirc(r, component=component)
    
    def vcirc(self, r, component="total"):
        """Circular velocity at radius r"""
        return np.sqrt(np.clip(-self.accr(r, component=component) * r, 0., None))

    def tidal_tensor(self, x, x0=(0.,0.,0.), component="total"):
        """The Tidal Tensor Tij = - d2phi/(dxi dxy)"""
        dx = x-np.array(x0)
        r = np.sqrt(np.sum(dx**2, axis=-1))
        
        accr = self.accr(r, component=component)
        daccr_dr = self.daccdr(r, component=component)
        
        tid = np.zeros(x.shape[:-1] + (3,3))

        for i in range(0,3):
            for j in range(0,3):
                if i == j:
                    # this is  d(xi/r)/dxj
                    der_xr = 1. / r - x[...,i]**2/r**3
                else:
                    der_xr = -x[...,i]*x[...,j]/r**3

                tid[...,i,j] = accr * der_xr + daccr_dr * (x[...,i]/r * x[...,j]/r)

        return tid

    def tidal_eigval(self, r, component="total"):
        """Eigenvalues of the Tidal tensor"""
        accr = self.accr(r, component=component)
        daccr_drr = self.daccdr(r, component=component)
        
        lam_r = daccr_drr
        lam_phi = accr / r

        return np.array((lam_r, lam_phi, lam_phi))
    
    def e_l_of_rperi_rapo(self, rperi, rapo):
        "Maps peri- and apo-center radii to energy and angular-momentum"
        return numerics.utility.e_l_of_rp_ra(lambda r: self.potential(r, component="total"), rperi, rapo, 
                                             accr=lambda r: self.accr(r, component="total"), eps_circ=self.cfg.actions.eps_circ)
    
    def posvel_to_rEL(self, pos, vel):
        """Calculates the radius, energy and angular momentum of particles
        
        pos : positions of the particles
        vel : velocities of the particles
        
        returns : (r, E, L)  with  the radius, energy and angular momentum
        """
        r = np.sqrt(np.sum(pos**2, axis=-1))
        E = self.potential(r, component="total") + 0.5*np.sum(vel**2, axis=-1)
        L = np.sqrt(np.sum(np.cross(pos, vel)**2, axis=-1))
        
        return r, E, L
    
    #----------- Search related functions --------------#
    # These functions help to find e.g. a radius where a given condition is true
    
    def _search_radius(self, f, rlow=None, rup=None, niter=None, logspace=True, invalid_val=np.nan):
        rlow = rlow or self.rmin()
        rup = rup or self.rmax()
        niter = niter or self.cfg.actions.niter_pa

        return numerics.search.ridders_method(f, rlow, rup, mode="positive", niter=niter, logspace=logspace, invalid_val=invalid_val)
    
    def r_of_potential(self, phi, component="total"):
        "Find radius where the potential is phi (if non-monotoneous considering only ascending part)"
        return self._search_radius(lambda r: self.potential(r, component=component) - phi, rup=self.rapo_max())

    def r_of_ecirc(self, ecirc, region="asc"):
        "Find radius where the circular energy is ecirc (if non-monotoneous region can be 'asc' or 'desc')"
        def f(r): return self.potential(r, component="total") + 0.5*self.vcirc(r, component="total")**2 - ecirc
        rlmax = self.rlmax() # radius where circular energy is maximal

        if region == "asc":
            rup = min(self.rmax(), rlmax)
            return self._search_radius(f, rup=rup)
        elif region == "desc":
            if not np.isfinite(rlmax):
                raise ValueError("Cannot search for descending part, as there is no maximum")
            return self._search_radius(f, rlow=rlmax)
        else:
            raise ValueError("Unknown mode %s" % region)
        
    def r_of_lcirc(self, lcirc, region="asc"):
        def f(r): return self.vcirc(r, component="total")*r - lcirc
        rlmax = self.rlmax() # radius where circular angular momentum is maximal

        if region == "asc":
            rup = min(self.rmax(), rlmax)
            return self._search_radius(f, rup=rup)
        elif region == "desc":
            if not np.isfinite(rlmax):
                raise ValueError("Cannot search for descending part, as there is no maximum")
            return self._search_radius(f, rlow=rlmax, rup=self.rtid())
        else:
            raise ValueError("Unknown mode %s" % region)

    def radius_of_f(self, f, l=1., rlow=None, rup=None, component="total"):
        "Radius where the phase space density f(phi(r), l) = f"
        def func(r): return np.log(self.f_of_el(self.potential(r, component="total"), l, component=component)/f)
        return self._search_radius(func, rlow=rlow or self.rmin()*2, rup=rup or self.rmax()/2)
    
    def radius_of_pot(self, phi, rlow=None, rup=None, component="total"):
        "Radius where potential(r) = phi"
        def func(r): return self.potential(r, component=component) - phi
        return self._search_radius(func, rlow=rlow or self.rmin()*2, rup=rup or self.rmax()/2)
    
    def rperi_rapo_of_r_e_l(self, r, e, l, search_method=None, rlow=None, rup=None, niter=None, return_err=False, exceptions=True, invalid_val=None):
        def energy_permitted(r):
            return e - 0.5*l**2/r**2 - self.potential(r, component="total")

        if niter is None: niter = self.cfg.actions.niter_pa
        if rlow is None: rlow = self.rmin()
        if rup is None: rup = self.rapo_max()
        if search_method is None: search_method = self.cfg.actions.search_method

        if search_method == "binary":
            rp = numerics.search.vectorized_binary_search(energy_permitted, rlow*np.ones_like(r), r, niter=niter, return_err=return_err, exceptions=exceptions, xfallback=r)
            ra = numerics.search.vectorized_binary_search(energy_permitted, r, rup*np.ones_like(r), niter=niter, return_err=return_err, exceptions=exceptions, xfallback=r)
        elif search_method == "ridders":
            rp = numerics.search.ridders_method(energy_permitted, rlow*np.ones_like(r), r, mode="positive", niter=niter, logspace=True, invalid_val=invalid_val)
            ra = numerics.search.ridders_method(energy_permitted, r, rup*np.ones_like(r), mode="positive", niter=niter, logspace=True, invalid_val=invalid_val)
        else:
            raise ValueError("Unknown mode %s" % search_method)
        
        return rp, ra
    
    #----------- Phase Space Distribution --------------#

    def set_phase_space(self, phase_space="eddington", anisotropy=0.):
        if phase_space == "eddington":
            if anisotropy is None:
                self.phase_space = None
            else:
                self.phase_space = EddingtonPhaseSpace(lambda r: self.density(r, component="self"), lambda r: self.potential(r, component="total"), self.cfg.general, self.cfg.eddington, anisotropy=anisotropy)
        else:
            self.phase_space = phase_space
        
        if self.phase_space is not None:
            self.anisotropy = self.phase_space.anisotropy
        else:
            self.anisotropy = anisotropy

    def f_of_e(self, e, component="self"):
        assert self.phase_space is not None, "No phase space defined"
        return self.phase_space.f_of_e(e)
    
    def f_of_el(self, e, l, r=None, component="self"):
        """Phase space distribution function of energy and angular momentum f(e,l).
        
        Providing r may be necessary if the function is defined implicity through f(rp,ra)
            (as may be the case for subclasses based on adiabatic remnants)
        """
        assert self.phase_space is not None, "No phase space defined"
        return self.phase_space.f_of_el(e, l)
    
    def f_of_rperi_rapo(self, rp, ra, component="self"):
        e, l = self.e_l_of_rperi_rapo(rp, ra)
        return self.f_of_el(e, l, component=component)
    
    def f_of_jl(self, j, l, component="self"):
        rp,ra = self.action_map.rp_ra_of_jl(j, l)
        return self.f_of_rperi_rapo(rp, ra, component=component)
    
    def f(self, e=None, l=None, j=None, r=None, rp=None, ra=None, component="self"):
        """A general wrapper for the phase space distribution function
        
        Orbits can be characterized uniquely by (e,l), (j,l) or (rp,ra)
        So in principle any of these combinations should be sufficient to get the phase space density
        However, in some situations phase space densities may be defined implicity for some variables
        but explicity for others. For this scenario it can be benficial to provide additional variables
        and the implementation can make the optimal choice

        if you know peri and apocenters, the (rp,ra) version is always a good choice
        if you know energy and angular momentum (e,l) in some cases also passing r may be required (for finding rp,ra)
        For isotropic profiles passing just e is enough
        """
        if rp is not None and ra is not None:
            return self.f_of_rperi_rapo(rp, ra, component=component)
        elif e is not None and l is not None:
            return self.f_of_el(e, l, r=r, component=component)
        elif j is not None and l is not None:
            return self.f_of_jl(j,l, component=component)
        elif e is not None:
            return self.f_of_e(e, component=component)
        else:
            raise ValueError("Unknown combination of variables")
    
    def g_of_e(self, e, nintegrate=100):
        """Density of states g(E) associated with some energy. dm/de = g(E) * f(E)"""
        def integrand(r):
            return r**2 * np.sqrt(2.*(e[...,np.newaxis] - self.potential(r, component="total")))
        
        rmax = self.radius_of_pot(e, component="total")

        return (4.*np.pi)**2 * numerics.integrate.integrate_exp_tanh_a_b(integrand, self.rmin(), rmax, N=nintegrate)

    #----------- Action Calculation Methods --------------#

    def orbit_valid(self, e, l):
        if np.isfinite(self.rtid()):
            raise NotImplementedError("Not implemented for profiles with boundary")
        else:
            return e - l**2/(2.*self.rmax()**2) - self.potential(self.rmax(), component="total") < 0. # may not have apo-center beyond rmax

    def radial_action_of_r_e_l(self, r, e, l):
        """Numerically infer the radial action Jr as in Binney and Tremaine (2008) eq 3.224"""
        valid = self.orbit_valid(e, l)
        j = np.ones_like(r)*np.nan

        rp, ra = self.rperi_rapo_of_r_e_l(r[valid], e[valid], l[valid])
        j[valid] = self.radial_action_of_rp_ra(rp, ra)
        
        return j


    def radial_action_of_rp_ra(self, rp, ra, nintegrate=None, eps_circ=None):
        """Numerically infer the radial action Jr as in Binney and Tremaine (2008) eq 3.224
        
        orbits are approximated as close to circular if ra <= rp*(1+eps_circ)
        """
        nintegrate = nintegrate or self.cfg.actions.nintegrate
        eps_circ = eps_circ or self.cfg.actions.eps_circ
        pot = lambda r: self.potential(r, component="total")
        accr = lambda r: self.accr(r, component="total")
        daccdr = lambda r: self.daccdr(r, component="total")
        return numerics.integrate.calculate_radial_action_tanh_peri_apo(pot, rp, ra, nintegrate=nintegrate, accr=accr, daccdr=daccdr, eps_circ=eps_circ)

    def radial_period_of_rp_ra(self, rperi, rapo, nintegrate=None, eps_circ=None):
        """Numerically infer the radial orbital period time"""
        nintegrate = nintegrate or self.cfg.actions.nintegrate
        eps_circ = eps_circ or self.cfg.actions.eps_circ
        pot = lambda r: self.potential(r, component="total")
        accr = lambda r: self.accr(r, component="total")
        daccdr = lambda r: self.daccdr(r, component="total")
        djde = numerics.integrate.calculate_dj_de_tanh_peri_apo(pot, rperi, rapo, nintegrate=nintegrate, accr=accr, daccdr=daccdr, eps_circ=eps_circ)
        return djde*2.*np.pi
    
    #----------- Integrals and Moments --------------#

    def compute_pa_space_integral(self, r, f_of_rp_ra=None, vrmoment=0, vtmoment=0, vmoment=0, nintegrate=40, ramax=None, component="self"):
        """Integrates a function over velocity-space through peri-apo-space discretization
        f_of_rp_ra : the phase space density (dM/d3x/d3v) with peri and apo centers as arguments
        """
        if ramax is None: 
            ramax = self.rapo_max()
        else:
            ramax = min(self.rapo_max(), ramax)

        if f_of_rp_ra is None:
            f_of_rp_ra = functools.partial(self.f_of_rperi_rapo, component=component)
        
        pot = lambda r: self.potential(r, component="total")
        accr = lambda r: self.accr(r, component="total")
        daccdr = lambda r: self.daccdr(r, component="total")

        is_limited = numerics.search.profile_is_limited(accr, rpmin=self.rmin())
        if is_limited:
            rperi, rapo, rlmax, rtid, ramax_of_rp = numerics.interpolate.define_paspace_boundaries(pot, accr, daccdr, rpmin=self.rmin(), rmax=self.rmax())
            rperirange, raporange = (self.rmin(), rlmax), (self.rmin(), ramax_of_rp)
        else:
            rperirange, raporange = (self.rmin(), ramax), (self.rmin(), ramax)

        return numerics.integrate.integrate_f_paspace(f_of_rp_ra, pot, accr, r, N=nintegrate, N2=nintegrate,
                                                      rperirange=rperirange, raporange=raporange, 
                                                      vrmoment=vrmoment, vtmoment=vtmoment, vmoment=vmoment)
    
    def compute_vr2_vt2(self, r, nintegrate=40, component="self"):
        """Returns the velocity dispersions vr2 and vt2 as a function of radius"""
        rho_x_vr2 = self.compute_pa_space_integral(r, vrmoment=2, nintegrate=nintegrate, component=component)
        rho_x_vt2 = self.compute_pa_space_integral(r, vtmoment=2, nintegrate=nintegrate, component=component)
        rho = self.compute_pa_space_integral(r, nintegrate=nintegrate, component=component)

        return rho_x_vr2/rho, rho_x_vt2/rho
    
    def compute_line_of_sight_vdisp2_and_dens(self, R, nintegrate=40, ninterp=100, component="self"):
        """computes the line of sight velocity dispersion and the column density at projected radius R"""

        rip = np.geomspace(self.rmin(), self.rmax(), ninterp+2)[1:-1]
        rho_x_vr2 = self.compute_pa_space_integral(rip, vrmoment=2, nintegrate=nintegrate, component=component)
        rho_x_vt2 = self.compute_pa_space_integral(rip, vtmoment=2, nintegrate=nintegrate, component=component)
        # We also integrate the density numerically to inherit the same discreteness error
        rho = self.compute_pa_space_integral(rip, nintegrate=nintegrate, component=component)

        # Zero-densities can cause some errors with log-interpolation, let's remopve them and set the right boundary to zero
        rip, rho, rho_x_vr2, rho_x_vt2 = rip[rho > 0], rho[rho > 0], rho_x_vr2[rho > 0], rho_x_vt2[rho > 0]

        def ip_rho(r): return np.exp(np.interp(np.log(r), np.log(rip), np.log(rho), right=-np.inf))
        def ip_rho_x_vr2(r): return np.exp(np.interp(np.log(r), np.log(rip), np.log(rho_x_vr2), right=-np.inf))
        def ip_rho_x_vt2(r): return np.exp(np.interp(np.log(r), np.log(rip), np.log(rho_x_vt2), right=-np.inf))
        
        return numerics.integrate.integrate_line_of_sight_vdisp2_and_dens(ip_rho, ip_rho_x_vr2, ip_rho_x_vt2, R, nintegrate=nintegrate)
    
    def integral_density_squared(self, rmin=None, rmax=None, nintegrate=100, component="self"):
        if rmin is None: rmin = self.rmin()
        if rmax is None: rmax = self.rmax()

        def integrand(r):  return self.density(r, component=component)**2 * 4*np.pi*r**2
        
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
                return self.density(r, component="self")

        def drhosigr2_dlogr(logr, sigr2=0.):
            r = np.exp(logr)
            return (density(r) * self.accr(r, component="total") - 2.*density(r) / r * sigr2 * faniso(r)) * r

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

    def sample_particles(self, ntot=10000, result="r_e_l_vr_m", rmax=None, rpmin=None, rpmax=None, ninterp=None, nintegrate=None, nsteps_metropolis=None, weight_rp_ra=None, component="self"):
        """ Samples particles radii, energies, angular momenta, radial velocities and masses
        using a metropolis algorithm for the (E,L | r) sampling. This is not the fastest
        possibility, but it is very robust and works for every profile, including anisotropic
        ones

        --- important parameters ---
        ntot : number of particles
        rmax : maximal radius to sample
        rpmin : If given, all particles have a peri-center rp > rpmin
        rpmax : If given, all particles have a peri-center rp < rpmax
        weight_rp_ra : Can be a function f(rp,ra) to adapt sampling rate for differt orbits
                       Can also be "equal_log" to use a sampling rate that is generally good
                       as it has roughly equal varaince in each logarithmic bin in radius

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

        if weight_rp_ra is None:
            def weight_rp_ra(rp, ra):
                return 1.
        elif (type(weight_rp_ra) == str) and (weight_rp_ra == "equal_logr"):
            def weight_rp_ra(rp, ra): # choose so that we have equal uncertainty in log-r bins
                rm = np.sqrt(rp*ra)
                return 1./(4.*np.pi*rm**3* self.density(rm, component="self"))
        else:
            assert callable(weight_rp_ra)
        
        def fpa(rp, ra):
            return self.f_of_rperi_rapo(rp, ra, component=component) * weight_rp_ra(rp, ra)
        
        pot = lambda r: self.potential(r, component="total")
        accr = lambda r: self.accr(r, component="total")

        rho = numerics.integrate.integrate_f_paspace(fpa, pot, accr, ri, N=nintegrate, rperirange=(rpmin, rpmax))
        p["r"],p["m"] = numerics.sample.sample_rimi_from_density(ri, rho, ntot)

        p["rp"], p["ra"] = numerics.sample.sample_rp_ra_given_r_metropolis_perisplit(fpa, pot, accr, p["r"], rperirange=(rpmin, rpmax), nsteps_chain=nsteps_metropolis)
        p["e"],p["l"],p["vr"] = numerics.sample.E_L_vr_from_rp_r_ra(pot, p["rp"], p["r"], p["ra"])

        p["m"] /= weight_rp_ra(p["rp"], p["ra"] )

        p["rrho"] = ri
        p["rho"] = rho

        if ("pos" in result) or ("vel" in result):
            p["pos"] = numerics.sample.random_direction(ntot, ndim=3) * p["r"][...,np.newaxis]
            if "vel" in result:
                vr = p["pos"] * (p["vr"] / p["r"])[...,np.newaxis]
                vt_xy = numerics.sample.random_direction(ntot, ndim=2) * (p["l"]/p["r"])[...,np.newaxis]
                e1, e2 = numerics.sample.orthogonal_vectors(vr)
                p["vel"] = vr + e1 * vt_xy[...,0,np.newaxis] + e2 * vt_xy[...,1,np.newaxis]

        if "dict" in result:
            return p
        else:
            res = []
            for key in result.split("_"):
                assert key in p, "Unknown key %s" % key
                res.append(p[key])
            return res
        
    def sample_particles_new(self, ntot=10000, result="r_e_l_vr_m", rpmin=None, rpmax=None, ramin=None, ramax=None, ninterp=None, nintegrate=None, nsteps_metropolis=None, weighted=None, f=None, component="self"):
        """ 
        result : a string with the keys to be returned, separated by "_". May contain 
               "rp", "ra", "r", "e", "l", "j", "vr", "pos", "vel", "m"
               Returns a list of the requested keys in the order they are given.
               If result contains "dict", a dictionary with all keys is returned.
               Examples: "pos_vel_m", "r_e_l_vr_m", "j_l_m", "dict", "dict_pos_vel"
        rpmin, rpmax : minimal and maximal peri-center radius
        ramin, ramax : minimal and maximal apo-center radius

        weighted : Can be a function that determines particle number density for unequal mass sampling.
                Weight function should only depend on orbits (but not phases) and may have signatures
                w(rp, ra, **kwargs), w(e, l, **kwargs), w(j, l, **kwargs)
                The larger w, the more particles (of lower mass) on those orbits. The normalization is irrelevant
                Also supported is weighted="nice" which uses weights so that the density profile is optimally resolved
        f :     Provide to use a different phase space distribution function (defaults to self.f)
                May have same signatures as weighted
        """
        rpmin = rpmin or self.rmin()
        ramin = max(ramin or self.rmin(), rpmin)
        ramax = ramax or self.rapo_max()
        rpmax = min(rpmax or self.rlmax(), ramax)

        nintegrate = nintegrate or self.cfg.sampling.nintegrate
        ninterp = ninterp or self.cfg.sampling.ninterp
        nsteps_metropolis = nsteps_metropolis or self.cfg.sampling.nsteps_metropolis

        f = f or functools.partial(self.f, component=component)

        result_vars = result.split("_")

        def orbit_valid(rp, ra):
            return (rp > rpmin) & (ra > ramin) & (ra < ramax) & (rp < rpmax)

        if weighted is None:
            def weighted(rp, ra, **kwargs): return 1.
        else:
            if (type(weighted) == str) and (weighted == "nice"):
                # Choose weights so that we have roughly equal uncertainty in log-r bins
                # To avoid problems for profiles that approach 0 density, we
                # limit the density to be above the mean density at the maximal radius
                rhomean_min = self.m_of_r(ramax, component=component) / (4.*np.pi/3.*ramax**3)

                def weighted(rp, ra, **kwargs):
                    rgeom = np.sqrt(rp*ra)
                    return 1./(4.*np.pi*rgeom**3* np.clip(self.density(rgeom, component=component),rhomean_min,None))
            
            assert callable(weighted)

        def fweighted(j=None, l=None, rp=None, ra=None):
            e,_ = self.e_l_of_rperi_rapo(rp, ra)
            return f(rp=rp, ra=ra, j=j, l=l, e=e) * weighted(rp=rp, ra=ra, j=j, l=l, e=e) * orbit_valid(rp, ra)
        
        p = {}
        msamp, p["j"], p["l"], p["rp"], p["ra"] = self.action_map.sample_jl(nsamp=ntot, get_rp_ra=True, f=fweighted, nf=self.cfg.sampling.nf)

        p["e"] = self.e_l_of_rperi_rapo(p["rp"], p["ra"])[0]

        p["m"] = msamp / weighted(rp=p["rp"], ra=p["ra"], j=p["j"], l=p["l"], e=p["e"])

        assert np.min(p["m"]) > 0, "Something went wrong with the sampling, negative masses (Maybe your weights are negative?)"

        if set(result_vars).intersection(("r", "vr", "pos", "vel", "dict")): # Sampling radii is expensive, so avoid it if not asked for
            assert np.all((p["j"] > 0) & (p["l"] > 0)), "Something went wrong here!"

            pot = lambda r: self.potential(r, component="total")
            p["r"] = numerics.sample.sample_r_given_rp_ra_metropolis(pot, p["rp"], p["ra"], nsteps=nsteps_metropolis)
            
            p["vr"] = numerics.sample.E_L_vr_from_rp_r_ra(pot, p["rp"], p["r"], p["ra"])[2]

            if ("pos" in result) or ("vel" in result):
                p["pos"] = numerics.sample.random_direction(ntot, ndim=3) * p["r"][...,np.newaxis]
                if "vel" in result:
                    vr = p["pos"] * (p["vr"] / p["r"])[...,np.newaxis]
                    vt_xy = numerics.sample.random_direction(ntot, ndim=2) * (p["l"]/p["r"])[...,np.newaxis]
                    e1, e2 = numerics.sample.orthogonal_vectors(vr)
                    p["vel"] = vr + e1 * vt_xy[...,0,np.newaxis] + e2 * vt_xy[...,1,np.newaxis]

        if "dict" in result:
            return p
        else:
            res = []
            for key in result_vars:
                assert key in p, "Unknown key %s" % key
                res.append(p[key])
            return res

    def sample_particles_perisplits(self, size_per_split=10000, rpsplits=(None, None), result="r_e_l_vr_m", rmax=None, flat=True, **kwargs):
        """See sample_r_E_L_vr_m_metropolis for a detailed description of optional keyword parameters

        size_per_split : number of particles in each split
        rpsplits : a list of splitting points
        flat : whether to return particles in form (nsplits, nper_split) or as a flat array
        """
        
        res = []

        for i in range(len(rpsplits)-1):
            res.append(self.sample_particles(size_per_split, result=result, rpmin=rpsplits[i], rpmax=rpsplits[i+1], rmax=rmax, **kwargs))

        outputs = []
        if "dict" in result:
            out = {}
            for key in res[0]:
                out[key] = np.stack([r[key] for r in res], axis=0)
                if flat:
                    out[key] = out[key].reshape((-1,) + out[key].shape[2:])
            return out
        else:
            ncol = len(res[0])
            for j in range(ncol):
                outputs.append(np.stack([r[j] for r in res], axis=0))
            if flat:
                return [o.reshape((-1,) + o.shape[2:]) for o in outputs]
            else:
                return outputs

    # ----------- Utility Methods --------------#
    # these help with estimating some important scales etc.

    def rcirc_eta_of_rperi_rapo(self, rperi, rapo):
        """Given a peri and apo-center, finds the radius where a circular orbit
        is possible and the angular-momentum in units of the circular angular momentum
        
        useful for translating results to DASH simulations"""
        e, l = self.e_l_of_rperi_rapo(rperi, rapo)

        rcirc = self.r_of_ecirc(np.atleast_1d(e))
        lcirc = self.vcirc(rcirc, component="total")*rcirc

        return rcirc.reshape(np.shape(rperi)), (l/lcirc).reshape(np.shape(rperi))
    
    def rperiapo_of_eta_rcirc(self, rcirc, eta):
        """Given a circular radius and the angular momentum in units of
        the circular angularmomentum, calculates the peri and apo center
        of a corresponding orbit
        
        useful for translating results to DASH simulations"""
        E = self.potential(rcirc, component="total") + 0.5*self.vcirc(rcirc, component="total")**2
        l = rcirc*self.vcirc(rcirc, component="total")*eta

        rperi, rapo = self.rperi((rcirc,E,l)), self.rapo((rcirc,E,l))
        return rperi, rapo

    def effective_pericenter_tidal_eigval(self, r, vcirc_fac=1.):
        """Eigenvalues of the effective tidal tensor at peri-center, when the
        effect of the centrifugal force is included"""
        
        assert np.min(vcirc_fac) >= 1., "vcirc_fac is the ratio between pericenter velocity and circular velocity, has to be >= 1."
        
        lam = self.tidal_eigval(r, component="total")
        omega = 2.*np.pi / self.tcirc(r, component="total")
        lam[0] += omega**2/vcirc_fac**2
        
        return lam

    def two_body_relaxation_time(self, r, N, modeN="Ntot", lam=None, rsoft=None, rmax=None, rnorm=None, component="self"):
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
            Nr = self.m_of_r(r, component=component) / self.m_of_r(rnorm, component=component) * N
        elif modeN == "Nr":
            Nr = N
        else:
            raise ValueError("Unknown modeN = ", modeN)
        
        tdyn = self.tdyn(r, component="total")
        
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