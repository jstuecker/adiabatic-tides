import numpy as np
import os
from scipy.integrate import simps
from scipy.interpolate import  RectBivariateSpline, NearestNDInterpolator, LinearNDInterpolator
from . import mathtools
from .phasespace import PhaseSpace, EddingtonPhaseSpace, AnalyticPhaseSpace, ActionMap, InterpolatorActionMap
from .config import Configureable, only_on_change, GeneralConfig, EddingtonConfig, ActionsConfig, SamplingConfig
import time


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
        # assert 0
        
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
        assert 0
        if self._sc is None:
            self._initialize_numerical_scales()
        return self._sc
            
    def scale(self, name):
        """Query the value of a numerical scale"""
        assert 0
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

        rho = mathtools.integrate_f_paspace(self.f_of_el, self.potential, self.accr, ri, N=nintegrate, rperirange=(rpmin, rpmax))
        p["r"],p["m"] = mathtools.sample_rimi_from_density(ri, rho, ntot)

        p["rp"], p["ra"] = mathtools.sample_rp_ra_given_r_metropolis_perisplit(self.f_of_el, self.potential, self.accr, p["r"], rperirange=(rpmin, rpmax), nsteps_chain=nsteps_metropolis)
        p["e"],p["l"],p["vr"] = mathtools.E_L_vr_from_rp_r_ra(self.potential, p["rp"], p["r"], p["ra"])

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
    
    def rperi_rapo_of_r_e_l(self, r, e, l, search_method=None, rlow=None, rup=None, niter=None, return_err=False, exceptions=True):
        def energy_permitted(r):
            return e - 0.5*l**2/r**2 - self.potential(r)

        if niter is None: niter = self.cfg["actions"].niter_pa
        if rlow is None: rlow = self.rmin()
        if rup is None: rup = self.rmax()
        if search_method is None: search_method = self.cfg["actions"].search_method

        if search_method == "binary":
            rp = mathtools.vectorized_binary_search(energy_permitted, rlow*np.ones_like(r), r, niter=niter, return_err=return_err, exceptions=exceptions, xfallback=r)
            ra = mathtools.vectorized_binary_search(energy_permitted, r, rup*np.ones_like(r), niter=niter, return_err=return_err, exceptions=exceptions, xfallback=r)
        elif search_method == "ridders":
            rp = mathtools.ridders_method(energy_permitted, rlow*np.ones_like(r), r, mode="positive", niter=niter, logspace=True)
            ra = mathtools.ridders_method(energy_permitted, r, rup*np.ones_like(r), mode="positive", niter=niter, logspace=True)
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
        return mathtools.calculate_radial_action_tanh_peri_apo(self.potential, rp, ra, nintegrate=nintegrate, invalid_vr_to_zero=invalid_vr_to_zero)

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
        
    
    def radial_period(self, particle, exceptions="warning"):
        """Infers the time needed for a radial period of the orbit of a single particle
        
        cannot be broadcasted to more than one particle
        
        particles : Tuple defining a particle either given by 
                    (pos, vel) -- positions and velocities or by
                    (E, L) -- Energy, angular momentum or by
                    (r, E, L) -- radius, Energy, angular momentum 
                    Each can be vector-like. Not providing r decreases speed
        exceptions : If true exceptions may be thrown if there is a problem with determining
               rapo or rperi
        
        returns : the time needed for one full radial orbit
        """
        if len(particle) == 2:
            if np.shape(np.atleast_1d(particle[0]))[-1] == 3:
                pos, vel = particle
                r, E, L = self.posvel_to_rEL(pos, vel)
            else:
                E, L = particle
                r, _rmax = self.rcirc_rmax_of_l(L)
        else:
            r, E, L = particle
        
        def Tr_integrand(r, E, L):
            SQ = 2*E - 2*self.potential(r) - L**2 / r**2
            
            if SQ <= 0.:
                res = 0.
            else:
                res = 2./np.sqrt(np.clip(SQ, 0., None))

            return res

        rperi = self.rperi((r, E, L), exceptions=exceptions)
        rapo = self.rapo((r, E, L), exceptions=exceptions)
        
        assert rperi < rapo, "radial period undefined for circular orbits"
        
        assert np.max(np.isnan(rapo)) == False
        assert np.max(np.isnan(rperi)) == False

        from scipy.integrate import quad
        
        Tr, err = quad(Tr_integrand, rperi, rapo, args=(E,L))

        return Tr
    
    def vdispr2_via_jeans_integration(self, logr=None, anisotropy=0., density=None):
        """Obtain the radial velocity dispersion squared through integration of the 1st Jeans equation
        
        logr : can provide integration points optionally, should be decreasing and log(radius)
        anisotropy: anisotropy parameter beta = 1 - (sigma_phi**2 + sigma_t**2) / 2sigma_r**2 
                    can be a function of radius
        density: if provided use a different density distribution, than the one which generates the potential
        
        returns: r, sigmar2   radii and radial velocity dispersion
        """
        
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
            logr = np.linspace(np.log(self.scale("rmax")*1e3),np.log(self.scale("rmin")), 10000)
        else:
            assert np.all(logr[1:] <= logr[:-1]), "logr has to be descending"
        rhosigr2 = np.zeros_like(logr)

        for i in range(1,len(logr)):
            dlogr = logr[i] - logr[i-1]

            sigr2=rhosigr2[i-1]/density(np.exp(logr[i]-1))

            rhosigr2[i] = rhosigr2[i-1] +  drhosigr2_dlogr(logr[i-1], sigr2)*dlogr
            
        return np.exp(logr[::-1]), (rhosigr2/density(np.exp(logr)))[::-1]
    
    def _initialize_tidal_radius(self, reinit=False):
        """Calcualtes the maximum of the potential and of the angular momentum"""
        if (not self._tidal_radius_initialized) | reinit:
            if self.is_disrupted:
                self._rlmax, self._philmax, self._lmax = 0.,0.,0.
                self._rtid, self._phitid, self._elmax, self._philmax, self._lscale  = 0., 0., 0., 0., 0.
                self._tidal_radius_initialized = True
                return

            self._rlmax, self._philmax, self._lmax = self.tidal_lmax_radius(warning=False, getphi=True, getl=True)
            self._rtid, self._phitid = self.tidal_boundary(warning=False, getphi=True)
            self._has_tidal_radius = self._rtid < self.scale("rmax")
            if self._has_tidal_radius:
                self._lscale = self._lmax
                self._elmax = self._philmax + 0.5*self._lmax**2/self._rlmax**2
            else:
                self._lscale = self.vcirc(self.r0()) * self.r0()
                self._elmax = 0.
                
            self._tidal_radius_initialized = True
    
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
                    ip_l_of_e = mathtools.flexible_interpolator(Ecirc, Lcirc, logy=True, eps_for_logy=1e-20*self._lscale, kind=kind)
                    ip_r_of_e = mathtools.flexible_interpolator(Ecirc, rcirc, logy=True, eps_for_logy=self.scale("rmin"), kind=kind)
                    ip_r_of_l = mathtools.flexible_interpolator(Lcirc, rcirc, logy=True, eps_for_logy=self.scale("rmin"), logx=True, eps_for_logx=1e-20*self._lscale, kind=kind)
                else:
                    ip_l_of_e = mathtools.flexible_interpolator(Ecirc, Lcirc, logy=False, fill_value=(0., Lcirc[-1]), kind=kind)
                    ip_r_of_e = mathtools.flexible_interpolator(Ecirc, rcirc, logy=False, kind=kind)
                    ip_r_of_l = mathtools.flexible_interpolator(Lcirc, rcirc, logy=False, kind=kind)

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

            self.ri_asc = np.logspace(np.log10(self.scale("rmin")), np.log10(rmax_asc), self.scale("nbins_circ"))
            self.ip_lcirc_of_e_asc, self.ip_rcirc_of_e_asc, self.ip_rcirc_of_l_asc, _ = _rel_circ_interpolator(self.ri_asc, log=True)

            self._rel_circ_initialized = True
    
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
    
    def tidal_boundary(self, getphi=False, warning=True, eps=1e-12, maxiter=200):
        """The tidal radius -- corresponding to a maximum in the potential
        
        getphi : if True, appends the potential at rtid to the result
        warning : if True, prints warnings if rtid-> infty. I.e. if the
                  potential has no tidal radius
        eps : desired relative accuracy
        maxiter : when to stop iterating, if the desired rel. accuracy is never
                  reached
                  
        returns : rtid, the tidal radius and possibly its potential value
        """
        return find_boundary(self, getphi=getphi, warning=warning, eps=eps, maxiter=maxiter)
    
    def tidal_lmax_radius(self, getphi=False, getl=False, warning=True, eps=1e-12, maxiter=200):
        """The radius of maximum circular angular momentum
        
        This is the maximum in vcirc(r)*r. This is the radius where the bound
        orbit with highest energy and highest angular momentum is possible.
        Will be infty for monothonic profiles
        
        getphi : if True, appends the potential at rlmax to the result
        getl : if True, appends the angular momentum at rlmax to the result
        warning : if True, prints warnings if rtid-> infty. I.e. if the
                  potential has no tidal radius
        eps : desired relative accuracy
        maxiter : when to stop iterating, if the desired rel. accuracy is never
                  reached
                  
        returns : rlmax,  and possibly phimax and lmax
        """
        res = find_boundary(self, getphi=getphi, warning=warning, eps=eps, mode="lmax", maxiter=maxiter)
        if getl:
            rlmax = np.atleast_1d(res)[0]
            if rlmax < self.scale("rmax"):
                lmax = self.vcirc(rlmax) * rlmax
            else:
                lmax = np.infty
            return list(np.atleast_1d(res)) + [lmax]
        else:
            return res
        
    def rmax_vmax(self, mode="self", warning=True, eps=1e-12, maxiter=200):
        """The radius and the circular velocity where vcirc is maximual
        
        This is the maximum in vcirc(r)*r. This is the radius where the bound
        orbit with highest energy and highest angular momentum is possible.
        Will be infty for monothonic profiles
        
        mode : if "self" will only consider self-gravity,
               if "full" will also consider the tidal field (if exists)
        warning : if True, prints warnings if rtid-> infty. I.e. if the
                  potential has no tidal radius
        eps : desired relative accuracy
        maxiter : when to stop iterating, if the desired rel. accuracy is never
                  reached
                  
        returns : rlmax,  and possibly phimax and lmax
        """
        
        if mode == "self":
            rmax = find_boundary(self, warning=warning, eps=eps, mode="vmaxself", maxiter=maxiter)
            vmax = self.self_vcirc(rmax)
        elif mode == "full":
            rmax = find_boundary(self, warning=warning, eps=eps, mode="vmax", maxiter=maxiter)
            vmax = self.vcirc(rmax)
        else:
            raise ValueError("Unknown mode %s" % mode)
        return rmax,vmax
    
    def E_L_of_rperi_rapo(self, rperi, rapo):
        """Given a peri and apo-center, finds the energy and angular-momentum of the corresponding orbit"""
        phip = self.potential(rperi)
        phia = self.potential(rapo)

        e = phip + (phia - phip)*(rapo**2) / (rapo**2 - rperi**2)
        l = np.sqrt(2. * (phia - phip) / (rperi**-2 - rapo**-2))
        
        return e, l
        
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
    
    def integral_density_squared(self, cumulative=False, rmin=None, rmax=None, nbins=None):
        def dA_dlogr(logri):
            ri = np.exp(logri)
            try :
                return self.self_density(ri)**2 * 4*np.pi*ri**3
            except :
                return self.density(ri)**2 * 4*np.pi*ri**3

        if rmin is None:
            rmin = self.scale("rmin")
        if rmax is None:
            rmax = self.scale("rmax")
        if nbins is None:
            nbins = self.scale("nbins_circ")
            
        ri = np.logspace(np.log10(rmin), np.log10(rmax), nbins)
        res = mathtools.cum_simpson(dA_dlogr, np.log(ri))
        
        if cumulative:
            return ri, res
        else:
            return res[-1]
        
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
            rmin = self.scale("rmin") #* 1e-8
        if rmax is None:
            rmax = self.scale("rmax") #* 1e8
        emin, emax = self.potential(np.array((rmin, rmax)))
        
        rres = mathtools.vectorized_binary_search(func, rmin, rmax, niter=100, mode="sqrt")
        
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
        
    
    def self_density(self, r):
        """Self-Density in Msol/Mpc**3, does not include tidal field contributions"""
        return self.density(r)
    def self_m_of_r(self, r):
        """The mass contained inside radius r, does not include tidal field contributions"""
        return self.m_of_r(r)
    def self_accr(self, r):
        """Radial Acceleration (negative means pull towards center)"""
        return -self.G * self.self_m_of_r(r) / r**2
    def self_potential(self, r, zero_at_zero=False):
        """Self-Potential, does not include tidal field contributions"""
        return self.potential(r, zero_at_zero=zero_at_zero)
    def self_vcirc(self, r):
        """Circular velocity at radius r, does not include tidal field contributions"""
        return np.sqrt(np.clip(-self.self_accr(r) * r, 0., None))
    
    def to_string(self):
        raise NotImplementedError("to_string not implemented for this profile, need this for caching etc...")
        

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
            self.r200c = mathtools.RvirOfMvir(m200c, h=h)
        elif r200c is not None:
            assert m200c is None, "You provided both m200c and r200c, please only provide one"
            self.r200c = r200c
            self.m200c = mathtools.MvirOfRvir(r200c, h=h)
        else:
            raise ValueError("You have to provide either m200c or r200c")

        self.rs = self.r200c / self.conc
        super().__init__(anisotropy=anisotropy, rmin=rminrs*self.rs, rmax=rmaxrs*self.rs, **config)

        self.rhoc = self.m200c/(4.*np.pi*self.rs**3 * (np.log(1.+self.conc) - self.conc/(1.+self.conc)))
        self.phi0 = - 4.*np.pi*self.G*self.rhoc*self.rs**2
        
        self.phasespace_initialized =  False

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
    def __init__(self, slope=-1., rhoc=None, rscale=1., m0=None):
        super().__init__(phase_space=None)
        
        self.slope = slope
        self.rscale = rscale
        if rhoc is not None:
            self.rhoc = rhoc
        elif m0 is not None:
            assert 0
        else:
            raise ValueError("Pleas provide either rhoc or m0")
            
        self.phic = 4.*np.pi * self.G * self.rhoc * self.rscale**2 / ( (3. + self.slope) * (2. + self.slope)  )
            
        # Calculate the normalization of the phasespace distribution
        from scipy.special import gamma

        beta = -(6+self.slope)/(4.+2.*self.slope)
        rhostar = 4.*np.pi*np.sqrt(2.) * 2. * np.sqrt(np.pi) * gamma(-beta-1.5) / (4. * gamma(-beta)) * self.phic**(beta+1.5)
        self.f0 = self.rhoc / rhostar
        
    def _initialize_numerical_scales(self):
        """Sets some default values for numerical scales"""
        
        super()._initialize_numerical_scales()
        
        self._sc["fintegration_nstepsE"] = 501
        self._sc["fintegration_nstepsL"] = 201
        self._sc["ip_e_of_jl_nbinsE"] = 2000
        self._sc["ip_e_of_jl_nbinsL"] = 200

        self._sc["log_emin"] = -18
        self._sc["rmin"] = self.r0() * 1e-12
        self._sc["rperimin"] = self.r0() * 1e-12
        if self.slope >= -0.75:
            self._sc["log_emin"] = -23
            self._sc["rmin"] = self.r0() * 1e-15
            self._sc["rperimin"] = self.r0() * 1e-15
        if self.slope >= -0.5:
            self._sc["log_emin"] = -34
            self._sc["rmin"] = self.r0() * 1e-20
            self._sc["rperimin"] = self.r0() * 1e-20

        self._sc["niter_apoperi"] = 35
        
            
        #self._sc["fintegration_nstepsE"] = 1001
        
    def density(self, r):
        return self.rhoc * (r/self.rscale)**self.slope
    
    def drhodr(self, r):
        return self.rhoc * (r/self.rscale)**(self.slope-1.) * self.slope / self.rscale
    
    def rho_of_phi(self, phi, deriv=0):
        # phi = self.phic * (r/self.rscale)**(2.+self.slope)
        # rho = self.rhoc * (r/self.rscale)**self.slope

        # (r/self.rscale) = (phi/self.phic)**(1.(2.+self.slope))
        assert deriv <= 2

        alpha = self.slope/(2.+self.slope)
        if deriv == 0:
            return self.rhoc * (phi/self.phic)**alpha
        elif deriv == 1:
            return self.rhoc * (phi/self.phic)**alpha * alpha / phi
        elif deriv == 2:
            return self.rhoc * (phi/self.phic)**alpha * alpha * (alpha-1.) / phi**2
    
    def m_of_r(self, r):
        return 4.*np.pi * self.rhoc / self.rscale**self.slope / (3. + self.slope) * r**(3.+self.slope)
    
    def potential(self, r, zero_at_zero=True):
        """The gravitational  potential.
        zero_at_zero: if True, norm to phi(r->0)=0. This can be useful
        to avoid problems caused by roundoff errors as r->0"""
        
        assert self.slope > -2., "Have to check normalization for this case"
        
        return self.phic * (r/self.rscale)**(2.+self.slope)
        
    def r0(self):
        return self.rscale
    
    def f_of_e(self, energy):
        """The phase space distribution function 
        
        energy : (vector-like) energies to evaluate the distribution at
        
        returns : phase space density f(E) = dN/d3x/d3v
        """
        beta = -(6+self.slope)/(4.+2.*self.slope)
        
        val = self.f0 * energy**beta
        assert(np.all(~np.isnan(val)))

        return self.f0 * energy**beta
    
    def f_of_el(self, e, l):
        return self.f_of_e(e)
    
    def to_string(self):
        return "powerlaw_slope=%.3f_rscale=%.5e_rhoc=%.5e" % (self.slope, self.rscale, self.rhoc)
    
    def to_dict(self):
        d = {}
        
        d["slope"] = self.slope
        d["rscale"] = self.rscale
        d["rhoc"] = self.rhoc
        
        return d
    
class AnisotropicPowerlawProfile(RadialProfile):
    def __init__(self, alpha=None, beta=0., gamma=None, rhoc=1.):
        """
        Initialize a powerlaw profile with the given parameters.

        density profile: rho = rhoc * r**(-alpha)
        phase space profile: f(E,L) ~ E**-gamma L**-beta

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
            alpha = alpha_of_gamma_beta(gamma, beta)
        elif gamma is None:
            assert beta < alpha/2.
            gamma = gamma_of_alpha_beta(alpha, beta)
        else:
            raise ValueError("Please provide either alpha or gamma, not both")
        
        #print(f"alpha={alpha}, beta={beta}, gamma={gamma}")

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Normalization constants:
        self.rhoc = rhoc
        self.phic = 4.*np.pi * self.G * self.rhoc / ( (3. - self.alpha) * (2. - self.alpha)  )

        from scipy.special import gamma as GammaF

        Cby = 2**(1.5 - beta) * np.pi**1.5 * GammaF(1. - beta) * GammaF(gamma + beta - 1.5) / GammaF(gamma)
        
        self.fc = self.rhoc / Cby / self.phic**(-gamma-beta+1.5)

        def f_of_el(e, l):
            return self.fc * e**-self.gamma * l**(-2.*self.beta)
        
        self.set_phase_space(AnalyticPhaseSpace(f_of_el=f_of_el, anisotropy=self.beta))

    def density(self, r):
        return self.rhoc*r**(-self.alpha)
    
    def m_of_r(self, r):
        return 4.*np.pi * self.rhoc / (3. - self.alpha) * r**(3.-self.alpha)
    
    def potential(self, r, zero_at_zero=True):
        assert self.alpha < 2., "Have to check normalization for this case"
        
        return self.phic * r**(2.-self.alpha)

    def r0(self):
        return 1.0
    
    def _initialize_numerical_scales(self):
        super()._initialize_numerical_scales()

        self._sc["rmin"] = 1e-12
        self._sc["rperimin"] = 1e-12

    def to_string(self):
        return "alpha=%.3f_beta=%.5e_rhoc=%.5e" % (self.alpha, self.beta, self.rhoc)
    
    def to_dict(self):
        d = {}
        
        d["alpha"] = self.alpha
        d["beta"] = self.beta
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


class NumericalProfile(RadialProfile):
    def __init__(self, ri=None, rho=None, mass=None, r0=None, ancorphi="rmin", from_dict=None, potential_profile=None, boundary="powerlaw", anisotropy=0., **configs):
        """A radial profile of which only the density form is known
        
        ri : radius sampling points
        rho : density -- can be an array like ri or a function
        r0 : base radius, will be maximum radius of the profile if not provided
        ancorphi : where to set the potential to zero? Can be 'rmax', 'rmin' or "infty"
        potential_profile : can be passed to use the potential from another profile
                            (might e.g. be relevant for Eddington inversion)
        boundary : How to handle radii r < min(ri). Can be "constant" or "powerlaw"
                   For the powerlaw case a powerlaw profile is fitted based on the
                   two smallest radii. This is the recommended mode if applicable.
        from_dict : load a previous profile from a dict created by .to_dict()
        """
        super().__init__(anisotropy=anisotropy, rmin=ri[0], rmax=ri[-1], **configs)
        
        self.potential_profile = potential_profile 
        
        self.q = {}

        assert ancorphi == "rmin", "Only rmin is support from now on"
        
        if from_dict:
            self.from_dict(from_dict)
            return
        
        assert (ri is not None) & (rho is not None)

        if r0 is None:
            r0 = np.max(ri)
        self.base_radius = r0

        self.boundary = boundary
        
        self.set_density_profile(ri, rho)

    def _discrete_radii(self):
        return self.ri
            
    def set_density_profile(self, ri, rho, update=True, integration_mode="trapez"):
        """Change the bins that are used to bin the mass and solve the forces
        
        ri : radius sampling points
        rho : density -- can be an array like ri or a function
        update : whether to update the mass, potential and force-profiles. Should always 
                 be "True" unless you know what you are doing
        """
        self.ri = ri

        if callable(rho):
            rhoi = rho(ri)
        else:
            rhoi = rho

        self.ip_rho, self.ip_m, self.ip_phi = mathtools.solve_poisson_via_spline_with_smart_boundaries(ri, rhoi, lower_boundary=self.boundary, G=self.G)
        self.q["rho"], self.q["mofr"], self.q["phi"] = rho, self.ip_m(self.ri), self.ip_phi(self.ri)

        if callable(rho):
            self.ip_rho = rho

        self.phasespace_initialized = False
        self.potential_zero_at_infty = False

    def self_density(self, r):
        """Density in Msol/Mpc**3"""
        return self.ip_rho(r)
        
    def density(self, r):
        """Density in Msol/Mpc**3"""
        return self.self_density(r)
    
    def self_m_of_r(self, r):
        """The mass contained inside radius r"""
        return self.ip_m(r)
    
    def m_of_r(self, r):
        """The mass contained inside radius r"""
        if self.potential_profile is not None:
            return self.potential_profile.m_of_r(r)
        else:
            return self.self_m_of_r(r)

    def self_potential(self, r, zero_at_zero=False):
        """The gravitational  potential"""
        # assert not zero_at_zero, "mode not implemented"
        return self.ip_phi(r)

    def potential(self, r, zero_at_zero=False):
        """The gravitational  potential"""
        if self.potential_profile is not None:
            return self.potential_profile.potential(r)
        else:
            return self.self_potential(r, zero_at_zero=zero_at_zero)
        
    def phimax(self):
        return self.q["phi"][-1]

    def r0(self):
        """A scale radius"""
        return self.base_radius

    def to_dict(self):
        """Returns a dictionary with all variables that describe the current state"""
        d = {}
        d["ri"] = self.ri
        d["rhoi"] = self.q["rho"]
        d["base_radius"] = self.base_radius
        return d

    def from_dict(self, d):
        """Load a state  extracted from a previos '.to_dict()' call"""
        self.base_radius = d["base_radius"]
        self.set_density_profile(d["ri"], d["rhoi"], update=True)
        
    def to_string(self):
        # We just create a hash here which allows comparison
        # whether two MCProfiles are identical
        import zlib
        mystr = "baseradius%.5e" % self.base_radius
        mystr += "_rihash" + str(zlib.adler32(self.ri.data.tobytes()))
        mystr +=  "_rhoihash" + str(zlib.adler32(self.q["rho"].data.tobytes()))
        
        return mystr
    
    def g_of_e(self, energy):
        """Density of states"""
        if not self.phasespace_initialized:
            self._initialize_phasespace()
        
        return np.interp(energy, self.q["phi"], self.q["g"])
    
    def n_of_e(self, energy):
        return self.g_of_e(energy) * self.f_of_e(energy)
    
class MonteCarloProfile(RadialProfile):
    def __init__(self, ri=None, mi=None, base_profile=None, rmax=None, rmin=None, nbins=1000, rbins=None, ancorphi="rmax", from_dict=None):
        """A radial profile which is given by a histogram of particles
        
        ri : the radii of the particles (Mpc), can be provided later via set_particles
        mi : the masses (Msol), can be provided later via set_particles
        
        base_profile : An analytic base profile. Optional and will only used for setting scales
        rmax : the largest radius that is considered to have mass (in Mpc)
        rmin : the smallest radius that is consider to have mass (in Mpc)
        nbins : the number of bins
        rbins : explicitly set the bins -- if given rmin, rmax and nbins will be ignored
        ancorphi : where to set the potential to zero? Can be 'rmax', 'rmin' or "infty"
        from_dict : load a previous profile from a dict created by .to_dict()
        """
        super().__init__()
        
        self.q = {}
        
        assert ancorphi in ("rmax", "rmin", "infty"), "Invalid value for ancorphi=%s" % ancorphi
        self.ancorphi = ancorphi
        
        if from_dict:
            self.from_dict(from_dict)
            return
    
        if ri is not None:
            self.set_particles(ri,mi, update=False)
        
        #self.base_profile = base_profile
        if base_profile is not None:
            self.base_radius = base_profile.r0()
        else:
            self.base_radius = np.max(ri)
        
        self.set_bins(rmax=rmax, rmin=rmin, nbins=nbins, rbins=rbins, update=False)
        
        if ri is not None:
            self._update()
            
    def set_bins(self, rmax=None, rmin=None, nbins=1000, rbins=None, update=True):
        """Change the bins that are used to bin the mass and solve the forces
        
        rmax : the largest radius that is considered to have mass (in Mpc)
        rmin : the smallest radius that is consider to have mass (in Mpc)
        nbins : the number of bins
        rbins : explicitly set the bins -- if given rmin, rmax and nbins will be ignored
        update : whether to update the mass, potential and force-profiles. Should always 
                 be "True" unless you know what you are doing
        """
        if rbins is None:
            if rmin is None:
                rmin = self.base_radius * 1e-6
            if rmax is None:
                rmax = self.base_radius * 1e1
            self.rbins = np.logspace(np.log10(rmin), np.log10(rmax), nbins)
        else:
            self.rbins = rbins
            
        self.rbinscent = np.sqrt(self.rbins[1:]*self.rbins[:-1])
        self.Vbins = 4./3.*np.pi*(self.rbins[1:]**3 - self.rbins[:-1]**3)
        
        if update:
            self._update()

    def set_particles(self, ri, mi=1., update=True):
        """Set the particles positions and masses of this profile
        
        ri : radii or positions of the particles
        mi : masses of the particles
        update : if True, the density/mass/gravity profiles will be recalculated
                 should always be True, unless you know what you are doing
        """
        if ri.shape[-1] == 3:
            ri = np.sqrt(np.sum(ri**2, axis=-1))
        self.ri = ri
        self.mi = np.ones_like(self.ri) * mi
        
        if update:
            self._update()
            
    def _set_mass_profile(self, rho, m):
        assert (len(rho) == len(self.rbins)-1) & (len(m) == len(self.rbins))
        self.q["rho"], self.q["mofr"] = rho, m
        
        accr = - self.G * self.q["mofr"] / self.rbins**2
        self.q["phi"] = - mathtools.trapez_integral_cumulative(self.rbins, accr)

    def _update(self):
        """Recalculate the density/mass/gravity profiles"""
        #self.q["rho"], self.q["mofr"] = mathtools.get_mass_profile(self.ri, self.mi, self.rbins)
        rho, m = mathtools.get_mass_profile(self.ri, self.mi, self.rbins)
        self._set_mass_profile(rho, m)

    def density(self, r):
        """Density in Msol/Mpc**3"""
        return np.interp(np.log10(r), np.log10(self.rbinscent), self.q["rho"])
    
    def m_of_r(self, r):
        """The mass contained inside radius r"""
        return np.interp(np.log10(r), np.log10(self.rbins), self.q["mofr"])
    
    def potential(self, r, zero_at_zero=False):
        """The gravitational  potential"""
        assert not zero_at_zero, "mode not implemented"
        dphi = np.interp(np.log10(r), np.log10(self.rbins), self.q["phi"])
        if self.ancorphi == "rmax":
            return dphi - self.q["phi"][-1]
        else: # ancored at 0
            return dphi
        
    def r0(self):
        """A scale radius"""
        return self.base_radius

    def to_dict(self):
        """Returns a dictionary with all variables that describe the current state"""
        d = {}
        d["ri"] = self.ri
        d["mi"] = self.mi
        d["rbins"] = self.rbins
        d["base_radius"] = self.base_radius
        return d

    def from_dict(self, d):
        """Load a state  extracted from a previos '.to_dict()' call"""
        self.base_radius = d["base_radius"]
        self.set_particles(d["ri"], d["mi"], update=False)
        self.set_bins(rbins=d["rbins"], update=True)
        
    def to_string(self):
        # We just create a hash here which allows comparison
        # whether two MCProfiles are identical
        import zlib
        mystr = "baseradius%.5e" % self.r0()
        mystr += "_rbinshash" + str(zlib.adler32(self.rbins.data.tobytes()))
        mystr +=  "_rihash" + str(zlib.adler32(self.ri.data.tobytes()))
        mystr +=  "_mihash" + str(zlib.adler32(self.mi.data.tobytes()))
        
        return mystr
    
class ParticleProfile(NumericalProfile):
    def __init__(self, particles, rbins):
        """ This class is going to replace MonteCarloProfile and will ahve additional options
        particles -- can either be (r,m) or (r,m,vr,L) or (pos,vel,m)
        """
        self.rbins = rbins
        self.ri = np.sqrt(rbins[1:]*rbins[:-1])

        self.set_particles(particles, update=False)

        rho, mprof = mathtools.get_mass_profile(self.p["r"], self.p["m"], self.rbins)

        super().__init__(self.ri, rho, mprof, boundary="constant")

    def _update_mass_profile(self):
        rho, mprof = mathtools.get_mass_profile(self.p["r"], self.p["m"], self.rbins)
        super().set_density_profile(self.ri, rho)

    def set_particles(self, particles, update=True):
        """
        particles -- can either be (r,m) or (r,m,vr,l) or (pos,vel,m)
        """
        self.p = {}
        if len(particles) == 2:
            self.p["r"], self.p["m"] = particles
            self.p["vr"], self.p["l"] = None, None
        elif len(particles) == 4:
            self.p["r"], self.p["m"], self.p["vr"], self.p["l"] = particles
        elif len(particles) == 3:
            assert 0, "not tested"
            pos, vel, self.p["m"] = particles
            self.p["r"] = np.linalg.norm(pos, axis=-1)
            self.p["vr"] = np.sum(vel*pos, axis=-1)/self.p["r"]
            self.p["l"] = np.linalg.norm(np.cross(pos, vel), axis=-1)
        
        if update:
            self._update_mass_profile()

    def integrate_orbits_in_other_potential(self, accr, tmax, nsteps=1000, update=True):
        self.p["r"], self.p["vr"] = mathtools.integrate_radial_orbits(accr, self.p["r"], self.p["vr"], self.p["l"], tmax, nsteps=nsteps)

        self._update_mass_profile()
    
    def to_dict(self):
        """Returns a dictionary with all variables that describe the current state"""
        d = {}
        d["rbins"] = self.rbins
        d["p"] = self.p
        return d

    def from_dict(self, d):
        """Load a state  extracted from a previos '.to_dict()' call"""
        raise NotImplementedError("Not implemented yet")

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
    
class CompositeProfile(RadialProfile):
    def __init__(self, *profiles, idmain=0):
        """Create a profile by combining several profiles.
        
        All functions where it makes sense (e.g. density, potential) 
        will return the sum of all profile components.
        
        idmain : the index of the main profile which is used for setting
                 the r0 scale
        """
        super().__init__()
        self.profiles = profiles
        self.idmain = idmain
    def density(self, r):
        """Density in Msol/Mpc**3"""
        return np.sum([prof.density(r) for prof in self.profiles], axis=0)
    def drhodr(self, r):
        """Radial derivative of the density"""
        return np.sum([prof.drhodr(r) for prof in self.profiles], axis=0)
    def m_of_r(self, r):
        """The mass contained inside radius r"""
        return np.sum([prof.m_of_r(r) for prof in self.profiles], axis=0)
    def potential(self, r, zero_at_zero=False):
        """The gravitational potential"""
        return np.sum([prof.potential(r, zero_at_zero=zero_at_zero) for prof in self.profiles], axis=0)
    def r0(self):
        """A scale radius"""
        return self.profiles[self.idmain].r0()
    def daccdr(self, r):
        """The radial derivative of the acceleration"""
        return np.sum([prof.daccdr(r) for prof in self.profiles], axis=0)

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
    elif mode == "vmaxself":
        def acc(r):
            return -4.*np.pi*profile.self_density(r)*r**3 + profile.self_m_of_r(r)
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