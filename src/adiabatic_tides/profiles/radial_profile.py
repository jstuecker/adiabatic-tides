import numpy as np
from ..phasespace import PhaseSpace, EddingtonPhaseSpace, AnalyticPhaseSpace, ActionMap, InterpolatorActionMap
from ..config import Configureable, only_on_change, GeneralConfig, EddingtonConfig, ActionsConfig, SamplingConfig
import time
from .. import numerics
import functools

def deprecated(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        raise DeprecationWarning("This function is deprecated, please use a different one")
        return func(*args, **kwargs)
    return new_func


class RadialProfile(Configureable):
    DEFAULT_CONFIG = {
        "general": GeneralConfig(),
        "eddington": EddingtonConfig(),
        "actions": ActionsConfig(),
        "sampling": SamplingConfig()
    }

    def __init__(self, rmin=None, rmax=None, phase_space="eddington", anisotropy=0., **configs):
        """This is an abstract class defining the interface of RadialProfiles,
        don't initialize!"""

        self.G = 43.0071057317063e-10 # This is the gravitational constant in units of Mpc (km/s)^2 / Msol 
        self.is_disrupted = False
        self.potential_zero_at_infty = True # should replace this by a function that returns the potential zero-point
        self._f_initialized = False

        super().__init__(**configs)
        if rmin is not None:
            self.cfg["general"].rmin = rmin
        if rmax is not None:
            self.cfg["general"].rmax = rmax

        self.q = {}
        self.ip = {}
        
        self.reset_interpolators()

        self._sc = None

        self.set_phase_space(phase_space, anisotropy=anisotropy)
        
        self.action_map = InterpolatorActionMap(self)

    def set_phase_space(self, phase_space="eddington", anisotropy=0.):
        if phase_space == "eddington":
            self.phase_space = EddingtonPhaseSpace(self.density, self.potential, self.cfg, anisotropy=anisotropy)
        else:
            self.phase_space = phase_space
        
        if self.phase_space is not None:
            self.anisotropy = self.phase_space.anisotropy
        else:
            self.anisotropy = anisotropy

    def rmin(self):
        return self.cfg["general"].rmin / self.cfg["general"].scale_geometry
    def rmax(self):
        return self.cfg["general"].rmax * self.cfg["general"].scale_geometry
    def rapo_max(self):
        """The maximal radius at which orbital apo-centers can lie
        Either corresponds to rmax or to the tidal radius, depending on the profile
        """
        rtid, rmax = self.rtid(), self.rmax()
        return rtid if rtid < rmax else rmax

    def reset_interpolators(self):
        """Resets the interpolators, like j_of_el, e_of_kl etc...
        
        Calling this is only necessary if the profile has changed for some reason
        """
        self._e_of_jl_initialized = False
        self._j_of_el_initialized = False
        self._rel_circ_initialized = False
        self._tidal_radius_initialized = False
        
    def _initialize_numerical_scales(self):
        """Sets some default values for numerical scales"""
        
        self._sc = {}
        self._sc["rmin"] = self.r0() * 1e-12
        self._sc["rmax"] = self.r0() * 1e5
        self._sc["nbins_circ"] = 500
        self._sc["drfac_finitediff"] = 1e-5
        
        self._sc["rperimin"] = self.r0() * 1e-12
        self._sc["rapomax"] = self.r0() * 1e10

        self._sc["pss_rbins"] = 2000
        self._sc["pss_ebins"] = 200
        self._sc["pss_e_analytic_low"] = -0.999
        self._sc["pss_e_analytic_up"] = -0.1
        
        self._sc["ip_j_of_el_nbinsE"] = 250
        self._sc["ip_j_of_el_nbinsL"] = 100
        
        self._sc["ip_e_of_jl_nbinsE"] = 1000
        self._sc["ip_e_of_jl_nbinsL"] = 100
        
        self._sc["log_lmin"] = -7
        self._sc["log_emin"] = -10
        self._sc["log_rmin"] = -5
        
        self._sc["nbins_jr"] = 50
        
        self._sc["niter_apoperi"] = 30
        
        self._sc["log_emin_up"] = -4
        self._sc["fintegration_nstepsL"] = 101
        self._sc["fintegration_nstepsE"] = 201
        
        self._sc["rel_interpolation_kind"] = "cubic"
    
    def set_numerical_scales(self, **kwargs):
        """Sets numerical scales
        
        The default numerical scales are usually good enough. Only change the
        scales when you really need extra precision.
        
        Use for example like this:
        .set_numerical_scales(nbins_jr=100, fintegration_nstepsE=200)
        
        The list of possible keywords can be seen in the code of 
        ._initialize_numerical_scales()
        """
        # assert 0
        if self._sc is None:
            self._initialize_numerical_scales()
        for kw in kwargs:
            assert kw in self._sc, "scale with name '%s' unknown" % kw
            self._sc[kw] = kwargs[kw]
            
    def scaledict(self):
        """A dictionary containg the value of all numerical scales"""
        # assert 0
        if self._sc is None:
            self._initialize_numerical_scales()
        return self._sc
            
    def scale(self, name):
        """Query the value of a numerical scale"""
        # assert 0
        if self._sc is None:
            self._initialize_numerical_scales()
        return self._sc[name]

    #----------- Core functions that every profile should implement --------------#
    def density(self, r): # The density
        """Abstract: The density profile"""
        raise NotImplementedError("This is an abstract class, please implement a subclass")

    def m_of_r(self, r):
        """Abstract: The mass contained inside radius r"""
        raise NotImplementedError("This is an abstract class, please implement a subclass")

    def potential(self, r, zero_at_zero=False):
        """Abstract: The gravitational potential. By default normed to 0 at infinity"""
        raise NotImplementedError("This is an abstract class, please implement a subclass")

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
        cfg : SamplingConfig = self.cfg["sampling"]
        cfg_gen : GeneralConfig = self.cfg["general"]

        if rmax is None: rmax = self.rmax()
        if rpmin is None: rpmin = self.rmin()
        if rpmax is None: rpmax = rmax

        if nintegrate is None: nintegrate = int(cfg.nintegrate * cfg_gen.scale_accuracy)
        if ninterp is None: ninterp = int(cfg.ninterp * cfg_gen.scale_accuracy)
        if nsteps_metropolis is None: nsteps_metropolis = int(cfg.nsteps_metropolis * cfg_gen.scale_accuracy)

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

    #----------- Functions that can be implemented on the abstract level already ----------# 
    def accr(self, r):
        """Radial Acceleration (negative means pull towards center)"""
        return  -self.G * self.m_of_r(r) / r**2

    def daccdr(self, r):
        """ accr = -G m(r) / r^2
        daccr/dr = 2 G m(r) / r^3 - G m'(r) / r^2 = -2 G accr(r) / r - G rho(r) 4 pi
        """
        return -2 * self.accr(r) / r - 4.*np.pi * self.density(r) * self.G
    
    def m0(self):
        """The mass contained inside r0"""
        return self.m_of_r(self.r0())
    
    def tdyn(self, r):
        """Dynamical Time-scale r / vcirc(r)"""
        return r / self.vcirc(r)
    
    def tcirc(self, r, inyears=False):
        """Time needed for a circular orbit at radius r. Default unit is (mpc/km) s
        
        inyears : transform to years
        """
        if inyears:
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
    
    def _search_radius(self, f, rlow=None, rup=None, niter=None, logspace=True, invalid_val=np.nan):
        rlow = self.rmin() if rlow is None else rlow
        rup = self.rmax() if rup is None else rup
        niter = int(self.cfg["actions"].niter_pa * self.cfg["general"].scale_accuracy) if niter is None else niter

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
    
    def rperi_rapo_of_r_e_l(self, r, e, l, search_method=None, rlow=None, rup=None, niter=None, return_err=False, exceptions=True):
        def energy_permitted(r):
            return e - 0.5*l**2/r**2 - self.potential(r)

        if niter is None: niter = self.cfg["actions"].niter_pa
        if rlow is None: rlow = self.rmin()
        if rup is None: rup = self.rmax()
        if search_method is None: search_method = self.cfg["actions"].search_method

        if search_method == "binary":
            rp = numerics.search.vectorized_binary_search(energy_permitted, rlow*np.ones_like(r), r, niter=niter, return_err=return_err, exceptions=exceptions, xfallback=r)
            ra = numerics.search.vectorized_binary_search(energy_permitted, r, rup*np.ones_like(r), niter=niter, return_err=return_err, exceptions=exceptions, xfallback=r)
        elif search_method == "ridders":
            rp = numerics.search.ridders_method(energy_permitted, rlow*np.ones_like(r), r, mode="positive", niter=niter, logspace=True)
            ra = numerics.search.ridders_method(energy_permitted, r, rup*np.ones_like(r), mode="positive", niter=niter, logspace=True)
        else:
            raise ValueError("Unknown mode %s" % search_method)
        
        return rp, ra
    
    def radial_action_of_r_e_l(self, r, e, l):
        """Numerically infer the radial action Jr as in Binney and Tremaine (2008) eq 3.224"""
        rp, ra = self.rperi_rapo_of_r_e_l(r, e, l)
        return self.radial_action_of_rp_ra(rp, ra)

    def radial_action_of_rp_ra(self, rp, ra, nintegrate=None, invalid_vr_to_zero=True):
        """Numerically infer the radial action Jr as in Binney and Tremaine (2008) eq 3.224"""
        if nintegrate is None:
            nintegrate = int(self.cfg["actions"].nintegrate * self.cfg["general"].scale_accuracy)
        return numerics.integrate.calculate_radial_action_tanh_peri_apo(self.potential, rp, ra, nintegrate=nintegrate, invalid_vr_to_zero=invalid_vr_to_zero)

    def radial_period_of_rp_ra(self, rperi, rapo, nintegrate=None):
        """Numerically infer the radial orbital period time"""
        if nintegrate is None:
            nintegrate = int(self.cfg["actions"].nintegrate * self.cfg["general"].scale_accuracy)
        djde = numerics.integrate.calculate_dj_de_tanh_peri_apo(self.potential, rperi, rapo, nintegrate=nintegrate)
        return djde*2.*np.pi

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

    @deprecated
    def _initialize_rel_circ_interpolators(self, reinit=False):
        """"""
        
        if (not self._rel_circ_initialized) | reinit:
            self._initialize_tidal_radius(reinit=reinit)
            
            def _rel_circ_interpolator(ri, log=True, kind=self.scale("rel_interpolation_kind")):
                rcirc = ri
                vcirc = self.vcirc(rcirc)
                Lcirc = vcirc*rcirc
                Ecirc = self.potential(rcirc) + 0.5*vcirc**2
                
                if log:
                    ip_l_of_e = numerics.interpolate.flexible_interpolator(Ecirc, Lcirc, logy=True, eps_for_logy=1e-20*self._lscale, kind=kind)
                    ip_r_of_e = numerics.interpolate.flexible_interpolator(Ecirc, rcirc, logy=True, eps_for_logy=self.rmin(), kind=kind)
                    ip_r_of_l = numerics.interpolate.flexible_interpolator(Lcirc, rcirc, logy=True, eps_for_logy=self.rmax(), logx=True, eps_for_logx=1e-20*self._lscale, kind=kind)
                else:
                    ip_l_of_e = numerics.interpolate.flexible_interpolator(Ecirc, Lcirc, logy=False, fill_value=(0., Lcirc[-1]), kind=kind)
                    ip_r_of_e = numerics.interpolate.flexible_interpolator(Ecirc, rcirc, logy=False, kind=kind)
                    ip_r_of_l = numerics.interpolate.flexible_interpolator(Lcirc, rcirc, logy=False, kind=kind)

                return ip_l_of_e, ip_r_of_e, ip_r_of_l, (rcirc, Ecirc, Lcirc)

            if self._has_tidal_radius:
                # Ecirc(r) and Lcirc(r) are not monothonic, we have to make 
                # separate functions for the increasing and decreasing part
                # rlmax is the radius of the maximum
                self.ri_desc = np.linspace(self._rlmax, self._rtid, self.scale("nbins_circ"))
                self.ip_lcirc_of_e_desc, self.ip_rcirc_of_e_desc, self.ip_rcirc_of_l_desc, _ = _rel_circ_interpolator(self.ri_desc, log=False)

                rmax_asc = self._rlmax
            else:
                rmax_asc = self.scale("rmax")

            self.ri_asc = np.logspace(np.log10(self.rmin()), np.log10(rmax_asc), self.scale("nbins_circ"))
            self.ip_lcirc_of_e_asc, self.ip_rcirc_of_e_asc, self.ip_rcirc_of_l_asc, _ = _rel_circ_interpolator(self.ri_asc, log=True)

            self._rel_circ_initialized = True
    
    @deprecated
    def rcirc_rmax_of_e(self, e, reinit=False):
        """The radii where a circular orbit with energy e is possible
        
        For monothonic profiles rmax is undefined and will be set to infty.
        For non-monothonic profiles (e.g. with a tidal field) it can be
        possible to have circular orbits with energy e at two different radii.
        However, the larger one, rmax, is unstable, corresponding to a maximum
        of the effective potential.
        On the first call interpolators for this function are calculated
        through ._initialize_rel_circ_interpolators().
        
        e : energy
        reinit : if given, reinitializes the interpolator

        returns : rcirc, rmax -- radii where e is the minimum and maximum 
                  of the effective potential. For monothonic profiles it is 
                  rmax=infty
        
        -- relevant numerical scales:
        rmin, rmax, nbins_circ
        """
        self._initialize_rel_circ_interpolators(reinit=reinit)

        lmin = self.ip_rcirc_of_e_asc(e)
        if self._has_tidal_radius:
            lmax =  self.ip_rcirc_of_e_desc(e)
        else:
            lmax = np.ones_like(e) * np.infty

        return lmin, lmax
    
    @deprecated
    def lminmax_of_e(self, e, reinit=False):
        """The minimum and maximum angular momentum possible for energy e

        For monothonic profiles lmin is always zero and lmax will correspond
        to the angular momentum of a circular orbit with energy e.
        For non-monothonic profiles (e.g. with a tidal field) it can be
        possible to have circular orbits with energy e at two different angular momenta.
        However, the lower angular momentum, lmin, corresponds to a maximum of the
        effective potential. Therefore orbits with l<lmin are unbound and the orbit with
        l=lmin is instable. lmin, lmax are therefore boundaries of possible angular momenta
        On the first call interpolators for this function are calculated
        through ._initialize_rel_circ_interpolators().
        
        e : energy
        reinit : if given, reinitializes the interpolator

        returns : lmin, lmax: the minimum and maximum possible energy momentum at energy e
        
        -- relevant numerical scales:
        rmin, rmax, nbins_circ
        """
        self._initialize_rel_circ_interpolators(reinit=reinit)

        lmax = self.ip_lcirc_of_e_asc(e)
        if self._has_tidal_radius:
            lmin =  self.ip_lcirc_of_e_desc(e)
        else:
            lmin = np.zeros_like(e)

        return lmin, lmax
    
    @deprecated
    def rcirc_rmax_of_l(self, l, reinit=False):
        """The radius of the minimum and maximum of the effective potential
        phieff(r) = phi(r) + 0.5 L**2/r**2
        
        For monothonic profiles rmax is undefined and will be infinty.
        For non-monothonic profiles (e.g. with a tidal field). The effective
        potential can have a maximum and therefore a circular orbit with angular
        momentum l can exist at rcirc and rmax. However, rmax is instable, since
        it is a maximum. All orbits with r > rmax are unbound. Therefore bound
        orbits are confined to r < rmax(l).
        
        l : angular momentum
        reinit : if given, reinitializes the interpolator

        returns : rcirc, rmax: the minimum and maximum of the effective potential
        
        -- relevant numerical scales:
        rmin, rmax, nbins_circ
        """
        if (not self._rel_circ_initialized) | (reinit):
            self._initialize_rel_circ_interpolators()
            
        rmin = self.ip_rcirc_of_l_asc(l)
        if self._has_tidal_radius:
            rmax =  self.ip_rcirc_of_l_desc(l)
        else:
            rmax = np.infty * np.ones_like(l)
        
        rmin = np.clip(rmin, 0., None)
        
        #assert np.all(rmin <= rmax)

        return rmin, rmax

    def E_L_of_rperi_rapo(self, rperi, rapo):
        """Given a peri and apo-center, finds the energy and angular-momentum of the corresponding orbit"""
        return numerics.utility.e_l_of_rp_ra(self.potential, rperi, rapo)
        
    def rcirc_eta_of_rperi_rapo(self, rperi, rapo):
        """Given a peri and apo-center, finds the radius where a circular orbit
        is possible and the angular-momentum in units of the circular angular momentum
        
        useful for translating results to DASH simulations"""
        phiperi = self.potential(rperi)
        phiapo = self.potential(rapo)

        l = np.sqrt(2.*(phiapo-phiperi)/(1./rperi**2 - 1./rapo**2 ) )
        e = phiperi + 0.5 * l**2 / rperi**2

        rcirc = self.rcirc_rmax_of_e(e)[0]
        lcirc = self.vcirc(rcirc)*rcirc

        return rcirc, l/lcirc
    
    def rperiapo_of_eta_rcirc(self, rcirc, eta):
        """Given a circular radius and the angular momentum in units of
        the circular angularmomentum, calculates the peri and apo center
        of a corresponding orbit
        
        useful for translating results to DASH simulations"""
        E = self.potential(rcirc) + 0.5*self.vcirc(rcirc)**2
        l = rcirc*self.vcirc(rcirc)*eta

        rperi, rapo = self.rperi((rcirc,E,l)), self.rapo((rcirc,E,l))
        return rperi, rapo

    def integral_density_squared(self, rmin=None, rmax=None, nintegrate=100):
        if rmin is None: rmin = self.rmin()
        if rmax is None: rmax = self.rmax()

        def integrand(r):  return self.density(r)**2 * 4*np.pi*r**2
        
        if np.min(rmin) == 0.:
            return numerics.integrate.integrate_double_exponential_a_b(integrand, rmin, rmax, N=nintegrate)
        else:
            return numerics.integrate.integrate_exp_double_exp_a_b(integrand, rmin, rmax, N=nintegrate)
        
    def radius_of_f(self, f, rmin=None, rmax=None):
        """Approximates the radius where the phase space density reaches a given value
        
        This function is useful to find a radius where a primordial phase space 
        density constrained starts getting violated by the profile.
        At the given radius it holds f_of_e(potential(r)) = f. Typically this is the
        highest phase space density that is reached at that radius and states with
        non-zero angular momentum will have lower phase space densities. Therefore
        r will be the largest radius where the phase space density f can be reached
        by any particles.
               
        f : phase space density in Msol / (km/s)**3 / Mpc**3
        rmin : Minimum radius for the binary search
        rmax : Maximum radius for the binary search
        
        returns : radius in Mpc
        """
        def func(r):
            return self.f_of_e(self.potential(r)) - np.array(f)
        if rmin is None:
            rmin = self.rmin()
        if rmax is None:
            rmax = self.rmax()

        rres = numerics.search.vectorized_binary_search(func, rmin, rmax, niter=100, mode="sqrt")
        
        return rres
    
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
        
        c = 299792458.0
        
        def fmax_wdm(h=0.68, omega_dm=0.26, gx=1.5, mx=1.):
            """Following https://arxiv.org/pdf/2207.05082.pdf

            the phase space density of a thermal relic WDM
            """
            def v0_wdm(omega_dm=0.28, h=0.678, gx=1.5, mx=1., a=1.):
                """omgega_dm: dark matter (not full matter) density paramater, mx in kev
                result : velocity in km/s
                """
                # Bode (2001), arXiv:astro-ph/0010389
                v0 = 0.012 * a**(-1) * (omega_dm / 0.3)**(1./3.) * (h/0.65)**(2./3.) * (1.5/gx)**(1./3.) * (1./mx)**(4./3.)
                return v0

            rho_dm = 3. * (h * 100.)**2 / (8.*np.pi*self.G) * omega_dm

            v0 = v0_wdm(mx=mx, omega_dm=omega_dm, gx=gx, h=h)

            return 0.0221 * v0**-3 * rho_dm
        
        def fmax_wimp(h=0.68, omega_dm=0.26,  mx=100, Td=30., ad=5.332e-12):
            """Following https://arxiv.org/pdf/2207.05082.pdf

            mx : WIMP mass in GeV
            Td : Decoupling Temperature in MeV
            omgega_dm: dark matter (not full matter) density paramater, mx in kev
            ad : scale factor of decoupling (where the temperature of the universe is Td)
                 This can be put to None to use an approximation by the neutrino temperature
                 which may have errors of order 10% if the wimp decoupled a bit before the
                 neutrinos
            
            the phase space density of WIMP's in Msol (km/s)**3 Mpc**3
            """
            mev, Tdev = mx*1e9, Td*1e6

            if ad is None:
                print("Approximating ad by assuming evaluating T(a_d)=Td while using the temperature T(a) of the Neutrino background.\n"
                      "This may give inaccurate results by 10-20%. For full accuracy use a full thermal history and determine ad")
                Tcmb = 2.725 #K
                kb = 8.617333262e-5 # eV/kelvin
                Tnu = Tcmb*(4./11.)**(1./3.) * kb   # in eV
                ad = (Tnu/Tdev)
            
            v0 =  np.sqrt(Tdev * mev)*ad / mev * c / 1e3  # velocity today in km/s

            rho_dm = 3. * (h * 100.)**2 / (8.*np.pi*self.G) * omega_dm

            return (2.*np.pi)**(-3./2.) * v0**-3 * rho_dm
        
        
        if dmtype == "WDM":
            fmax = fmax_wdm(h=h, omega_dm=omega_dm, **kwargs)
        elif dmtype == "WIMP":
            fmax = fmax_wimp(h=h, omega_dm=omega_dm, **kwargs)
        else:
            raise ValueError("Unknown dmtype=%s, so far can only handle WDM or WIMP" % dmtype)
            
        return self.radius_of_f(fmax)
    
    def to_string(self):
        raise NotImplementedError("to_string not implemented for this profile, need this for caching etc...")