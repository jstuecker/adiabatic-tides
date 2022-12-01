import numpy as np
import os
from scipy.integrate import simps
from scipy.interpolate import  RectBivariateSpline, NearestNDInterpolator, LinearNDInterpolator
from . import mathtools
from . import phasespace

import time

class RadialProfile():
    def __init__(self):
        """This is an abstract class defining the interface of RadialProfiles,
        don't initialize!"""
        
        # This is the gravitational constant in units of Mpc (km/s)^2 / Msol 
        self.G = 43.0071057317063e-10
        self.is_disrupted = False
        self.potential_zero_at_infty = True
        
        self.reset_interpolators()

        self._sc = None
        
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
        if self._sc is None:
            self._initialize_numerical_scales()
        for kw in kwargs:
            assert kw in self._sc, "scale with name '%s' unknown" % kw
            self._sc[kw] = kwargs[kw]
            
    def scaledict(self):
        """A dictionary containg the value of all numerical scales"""
        if self._sc is None:
            self._initialize_numerical_scales()
        return self._sc
            
    def scale(self, name):
        """Query the value of a numerical scale"""
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

    def r0(self):
        """A radius that is used for normalization of the profile"""
        raise NotImplementedError("This is an abstract class, please implement a subclass")
        
    #----------- Optional features, that can be helpful in some situations ----------#
    def sample_particles(self, ntot=10000, rmax=None, seed=42):
        """Abstract: Sample particles' positions, velocities and masses"""
        raise NotImplementedError("This optional function has not been implemented")
    
    def f_of_e(self, E):
        """Abstract: A phase space distribution function that only depends on energy"""
        raise NotImplementedError("This optional function has not been implemented")

    #----------- Functions that can be implemented on the abstract level already ----------# 
    def accr(self, r):
        """Radial Acceleration (negative means pull towards center)"""
        return  -self.G * self.m_of_r(r) / r**2
    
    
    def daccdr(self, r, h=None):
        """Radial derivative of the acceleration. Infered numerically"""
        if h is None:
            h = r*self.scale("drfac_finitediff")
        return (self.accr(r+h) - self.accr(r-h))/(2.*h)
    
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
    
    
    def rperi(self, particles, rlow=None, niter=None, return_err=False, exceptions=True):
        """Calculates the peri-center radii of particles
        
        particles : Tuple defining the particles either given by 
                    (pos, vel) -- positions and velocities or by
                    (E, L) -- Energies, angular momenta or by
                    (r, E, L) -- radii, Energies, angular momenta 
                    Each can be vector-like. Not providing r decreases speed
        rlow : An estimate of a radius that is definetly smaller than the 
               peri-center. A reasonable guess is made if not provided. Don't
               put this to zero!
        niter : number of iterations in the binary search. Increasing this
               improves accuracy. niter=30 already gives 1e-8 relative Error
               in most cases
        return_err : If true an error estimate is appended to the return value
        exceptions : If true exceptions may be thrown if not rlow < rperi < r
        
        returns : an arrray with the peri-center radius for each particle
        """
        if niter is None:
            niter = self.scale("niter_apoperi")
        
        if len(particles) == 2:
            if np.shape(np.atleast_1d(particles[0]))[-1] == 3:
                pos, vel = particles
                r, E, L = self.posvel_to_rEL(pos, vel)
            else:
                E, L = particles
                r, _rmax = self.rcirc_rmax_of_l(L)
        else:
            r, E, L = particles
        
        if rlow is None:
            rlow = self.scale("rperimin")

        def energy_permitted(r, E, L): # This function changes sign at peri/apo center
            return E - 0.5*L**2/r**2 - self.potential(r)

        return mathtools.vectorized_binary_search(energy_permitted, rlow*np.ones_like(r), r, E=E, L=L, niter=niter, return_err=return_err, exceptions=exceptions, xfallback=r)
    
    def rapo(self, particles, rup=None, niter=None, return_err=False, exceptions=True):
        """Calculates the peri-center radii of particles
        
        particles : Tuple defining the particles either given by 
                    (pos, vel) -- positions and velocities or by
                    (E, L) -- Energies, angular momenta or by
                    (r, E, L) -- radii, Energies, angular momenta 
                    Each can be vector-like. Not providing r decreases speed
        rup :  An estimate of a radius that is definetly larger than the 
               apo-center. A reasonable guess is made if not provided. Don't
               put this to zero!
        niter : number of iterations in the binary search. Increasing this
               improves accuracy. niter=30 already gives 1e-8 relative Error
               in most cases
        return_err : If true an error estimate is appended to the return value
        exceptions : If true exceptions may be thrown if not rlow < rperi < r
                    
        returns : an arrray with the apo-center radius for each particle
        """
        if niter is None:
            niter = self.scale("niter_apoperi")
        
        if len(particles) == 2:
            if np.shape(np.atleast_1d(particles[0]))[-1] == 3:
                pos, vel = particles
                r, E, L = self.posvel_to_rEL(pos, vel)
            else:
                E, L = particles
                r, _rmax = self.rcirc_rmax_of_l(L)
                if rup is None:
                    rup = np.clip(_rmax, 0., self.scale("rapomax"))
        else:
            r, E, L = particles
        
        if rup is None:
            rup = self.scale("rapomax")
            
        def energy_permitted(r, E, L): # This function changes sign at peri/apo center
            return E - 0.5*L**2/r**2 - self.potential(r)

        return mathtools.vectorized_binary_search(energy_permitted, r, rup*np.ones_like(r), E=E, L=L, niter=niter, return_err=return_err, exceptions=exceptions, xfallback=r)
    
    def radial_action(self, particles, nbins=None, rlow=None, rup=None, exceptions=False, dlog_rmin=None, rpow=0):
        """Numerically infer the radial action Jr as in Binney and Tremaine (2008) eq 3.224
        
        particles : Tuple defining the particles either given by 
                    (pos, vel) -- positions and velocities or by
                    (E, L) -- Energies, angular momenta or by
                    (r, E, L) -- radii, Energies, angular momenta 
                    Each can be vector-like. Not providing r decreases speed
        nbins : The number of function evaluations used for integration.
                Note that required memory scales as npart*nbins. This parameter
                is most vital for the desired accuracy, but also the performance. 
                Here some examples:
                20   : relative error ~1e-2
                50   : relative error ~ 8e-4
                100  : relative error ~ 3e-5
                200  : relative error ~ 1e-6
                (Tested on an NFW profile)
                defaults to self.scale("nbins_jr")
        rup : If the potential is not monotonic, specifiy the tidal radius here.
               (Otherwise rup defaults to something that should be fine)
        exceptions : If true exceptions may be thrown if there is a problem with determining
               rapo or rperi
        dlog_rmin : a parameter used for deciding placement of the integration points. Defaults to
               self.scale("log_rmin"). 
        rpow : if given, weight the integrand by r**pow. The action integral has pow=0.
               However, since other quantities (like the denisty of states) use an almost
               equivalent integral but with pow=2, we can evaluate them with this routine
               as well, by modifying pow.
        
        returns : the radial action jr
        """
        if nbins is None:
            nbins = self.scale("nbins_jr")

        if len(particles) == 2:
            if particles[0].shape[-1] == 3:
                pos, vel = particles
                r, E, L = self.posvel_to_rEL(pos, vel)
            else:
                E, L = particles
                r, _rmax = self.rcirc_rmax_of_l(L)
                if rup is None:
                    rup = np.clip(_rmax, 0., self.scale("rapomax"))
        else:
            r, E, L = particles
        
        if rpow == 0.:
            def Jr_integrand(r, E, L):
                SQ = 2*E - 2*self.potential(r) - L**2 / r**2

                return 1./np.pi * np.sqrt(np.clip(SQ, 0., None))
        else:
            def Jr_integrand(r, E, L):
                SQ = 2*E - 2*self.potential(r) - L**2 / r**2

                return 1./np.pi * np.sqrt(np.clip(SQ, 0., None)) * r**rpow

        rperi = self.rperi((r, E, L), exceptions=exceptions, rlow=rlow)
        rapo = self.rapo((r, E, L), exceptions=exceptions, rup=rup)
        
        assert np.max(np.isnan(rapo)) == False
        assert np.max(np.isnan(rperi)) == False
        
        circular = rperi >= rapo * (1.-1e-10)  # this can only be violated due to numerical roundoff errors where rapo=rperi
        
        # define different integration bins for each particle
        # shape will be (npart, nbins)
        if dlog_rmin is None:
            dlog_rmin = self.scale("log_rmin")
        n1,n2,n3 = nbins//2, nbins//4, nbins//2
        if (n1+n2+n3) % 2 == 0: # avoid even number of samples for simpson rule
            n2 += 1
        rbins_fac = mathtools.bins_log_lin_log(0., 0.1, 0.9, 1.0, dlogmin=dlog_rmin, n1=n1, n2=n2, n3=n3, include_xminxmax=True)
        #rbins = np.linspace(rperi, rapo, nbins).T
        rbins = (rperi[...,np.newaxis] + (rapo-rperi)[...,np.newaxis]*rbins_fac)
                
        # evaluate the integrand
        Ig = Jr_integrand(rbins, E[...,np.newaxis], L[...,np.newaxis])
        
        # sum up result using simpson rule
        Jr = simps(Ig, x=rbins, axis=-1)
        #Jr = mathtools.trapez_integral_lastax(rbins, Ig)
        Jr[circular] = 0.
        # negative Jr should usually not happen. It can be caused by the 3rd order interpolation assumption
        # of the simpson rule. In this cases just use the trapez rule
        Jr[Jr < 0.] = mathtools.trapez_integral_lastax(rbins[Jr < 0.], Ig[Jr < 0.])
        Jr[np.isnan(Jr)] = 0. # Those are usually almost circular orbits where the simpson rule divide 0 by 0
        #assert np.max(np.isnan(Jr)) == False
        
        return Jr
    
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
    
    def _initialize_j_of_el(self, nbinsE=None, nbinsL=None, reinit=False):
        """Initializes an interpolator of the action as a function of energy and angular momentum
        
        see j_of_el for details"""
        if self._j_of_el_initialized & (not reinit):
            return
        
        if nbinsE is None:
            nbinsE = self.scale("ip_j_of_el_nbinsE")
        if nbinsL is None:
            nbinsL = self.scale("ip_j_of_el_nbinsL")
        
        self._initialize_tidal_radius(reinit)
        
        try:
            emin = self.phi0
            emax = 0.
        except:
            emin = self.potential(self.scale("rmin"))
            emax = self.potential(self.scale("rmax"))

        if not self._has_tidal_radius:
            de = emax-emin
            E = mathtools.bins_log_lin_log(emin, emin+0.1*de, emax-0.1*de, emax, n1=nbinsE//3, n2=nbinsE//3, n3=nbinsE//3, dlogmin=self.scale("log_emin"), dlogmin_up=self.scale("log_emin_up"))
        else:
            assert self._elmax > emin
            E = mathtools.bins_log_both_ends(emin, self._elmax, nbinsE//2, nbinsE//2, dlogmin=self.scale("log_emin"))
        E = np.unique(E)
        assert np.min(E[1:] > E[:-1]) == True, E

        lfacs = np.logspace(self.scale("log_lmin"),0.,nbinsL)
        
        lmin, lmax = self.lminmax_of_e(E)
        Em, Lfacm = np.meshgrid(E, lfacs, indexing="ij")
        Lm = Lfacm * (lmax-lmin)[...,np.newaxis] + lmin[...,np.newaxis]
        
        Jrm = self.radial_action((Em.reshape(-1), Lm.reshape(-1)), exceptions="silent").reshape(Em.shape)
        
        assert np.min(Jrm) >= 0.
        
        spline_o3 = RectBivariateSpline(E, lfacs, Jrm, kx=3, ky=3)
        spline_o1 = RectBivariateSpline(E, lfacs, Jrm, kx=1, ky=1)

        elim = np.min(E), np.max(E)
        lflim = np.min(lfacs), np.max(lfacs)
        def j_of_el_func(E, lfacs):
            valid = (E >= elim[0]) & (E <= elim[1])
            valid &= (lfacs >= lflim[0]) & (lfacs <= lflim[1])
            
            res = spline_o3(E, lfacs, grid=False)
            
            # 3rd order interpolation can lead to (unphysical) slightly negative values
            # in very few instances. We just replace by linear interpolation for these cases
            # which is always >= 0.
            res[res < 0.] = spline_o1(E[res < 0.], lfacs[res < 0.], grid=False)
            res[~valid] = np.nan

            return res
        
        
        self.interp_j_of_el = j_of_el_func
        self._j_of_el_initialized = True
        
    def j_of_el(self, E, L, reinit=False):
        """radial action as a function of energy and angular momentum.
        
        On the first call this calculates an interpolation table through
        ._initialize_j_of_el(). This interpolation table is a grid in energy E 
        and Lfac where Lfac goes from 0 to 1 between Lmin(E) and Lmax(E). These
        boundaries are the minimal and maximum angular momenta possible for
        a given energy. Note that usually Lmin(E) = 0., but in cases of
        potentials that are non-monothoneos (like those with a tidal field) it
        can be Lmin(E) > 0
        
        E : energy
        L : angular momentum
        reinit : if given, reinitializes the interpolation table.

        returns : the radial action
        
        -- very relevant numerical scales:
        ip_j_of_el_nbinsE, ip_j_of_el_nbinsL, log_lmin, log_emin, log_emin_up
        
        -- weakly relevant numerical scales:
        rmin, rmax, nbins_circ, rperimin, rapomax
        """
        
        self._initialize_j_of_el(reinit=reinit)
            
        E = E*np.ones_like(L)
        lmin, lmax = self.lminmax_of_e(E)
        lfac = (L-lmin)/(lmax-lmin)
        valid = (lfac >= 0.) & (lfac < 1.)  & (lmax > lmin)
        
        jres = np.ones_like(L) * np.nan
        jres[valid] = self.interp_j_of_el(E[valid], lfac[valid])
        
        assert np.nanmin(jres) >= 0.
            
        return jres
    
    def _initialize_e_of_jl(self, nbinsE=None, nbinsL=None, reinit=False):
        """Initializes an interpolator of the action as a function of energy and angular momentum
        
        see e_of_jl for details"""
        if self._e_of_jl_initialized & (not reinit):
            return
        
        if nbinsE is None:
            nbinsE = self.scale("ip_e_of_jl_nbinsE")
        if nbinsL is None:
            nbinsL = self.scale("ip_e_of_jl_nbinsL")

        try:
            emin = self.phi0
            emax = 0.
        except:
            emin = self.potential(self.scale("rmin"))
            emax = self.potential(self.scale("rmax"))
            
            #assert np.abs(emin / emax) >= 1e-14, "Warning, I can run into roundoff errors here %g" % (emin/emax)

        de = emax-emin
        E = mathtools.bins_log_lin_log(emin, emin+0.1*de, emax-0.1*de, emax, n1=nbinsE//3, n2=nbinsE//3, n3=nbinsE//3, dlogmin=self.scale("log_emin"), dlogmin_up=self.scale("log_emin_up"))

        lmin, lmax = self.lminmax_of_e(E)
        Em, Lfacm = np.meshgrid(E, np.logspace(self.scale("log_lmin"),0.,nbinsL), indexing="ij")
        Lm = Lfacm * (lmax-lmin)[...,np.newaxis] + lmin[...,np.newaxis]

        Jrm = self.radial_action((Em.reshape(-1), Lm.reshape(-1)), exceptions="silent").reshape(Em.shape)
        
        xym = np.stack([np.log10(Jrm+Lm), (Jrm-Lm)/(Jrm+Lm)], axis=-1)
        Eint = LinearNDInterpolator(xym.reshape(-1,2), Em.reshape(-1))
        Eint_NN = NearestNDInterpolator(xym.reshape(-1,2), Em.reshape(-1))
        
        def e_of_jl(Jr, L):
            """Energy as function of the radial action and angular momentum"""
            assert np.min(Jr+L) >= 0
            
            xy = np.stack([np.log10(Jr+L), (Jr-L)/(Jr+L)], axis=-1)
            Ei = Eint(xy)
            #Ei[Jr+L == 0.] = np.min(E)
            # For locations outside the sample range the  LinearNDInterpolator becomes np.nan
            # for these cases we just use the nearest neighbor interpolation
            # note that this happens almost never
            invalid = np.isnan(Ei)

            Ei[invalid] = Eint_NN(xy[invalid])
            return Ei

        self._e_of_jl = e_of_jl
        self._e_of_jl_initialized = True

    def e_of_jl(self, J, L, reinit=False):
        """Energy as a function of radial action and angular momentum.
        
        On the first call this calculates a mesh-free linear interpolator through
        ._initialize_j_of_el(). The distorted grid J,L is obtained by setting up 
        a regular grid in (E,Lmax(E)) and mapping it to J,L. Further the interpolation 
        is done in (log(J+L),(J-L)/(J+L)) space to simplify the bounadries of the 
        interpolation space. 
        
        J : radial action
        L : angular momentum
        reinit : if given, reinitializes the interpolator

        returns : energy
        
        -- very relevant numerical scales:
        ip_e_of_jl_nbinsE, ip_e_of_jl_nbinsL, log_lmin, log_emin, log_emin_up
        
        -- weakly relevant numerical scales:
        rmin, rmax, nbins_circ, rperimin, rapomax
        """
        self._initialize_e_of_jl(reinit=reinit)
        
        return self._e_of_jl(J, L)
    
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
    def __init__(self, conc, m200c=None, r200c=None, h=0.679):
        """Set up an NFW profile with a given mass and concentration
        
        conc : concentration -- so that the scale radius is rs = r200c / c
        m200c : virial mass of the halo in units of Msol, r200c can be 
                provided instead
        r200c : virial radius of the halo in units of Mpc, m200c can be
                provided instead
        h : reduced hubble parameter. Set to 1 to use units where masses
            are measured in Msol/h and lengths in units of Mpc/h
        """
        super().__init__()
        
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
        x = r / self.rs
        M0 = 4.*np.pi*self.rs**3*self.rhoc
        
        m = np.zeros_like(r)
        sel = x > 1e-5
        m[sel] = M0 * (np.log(1 + x[sel]) + 1. / (1. + x[sel]) - 1.)
        m[~sel] = 0.5 * M0 * x[~sel]**2
        
        return m
    
    def potential(self, r, zero_at_zero=False):
        """The gravitational  potential. 0 at r -> infty.
        zero_at_zero: if True, norm to phi(r->0)=0. This can be useful
        to avoid problems caused by roundoff errors as r->0"""
        phi = np.zeros_like(r)
        x = r / self.rs
        sel = x > 1e-4
        if zero_at_zero:
            phi[sel] = self.phi0 * (np.log(1. + x[sel]) / x[sel] - 1.)
            phi[~sel] = self.phi0 * (- x[~sel]/2. + x[~sel]**2/3.)
        else:
            phi[sel] = self.phi0 * np.log(1. + x[sel]) / x[sel]
            phi[~sel] = self.phi0 * (1. - x[~sel]/2. + x[~sel]**2/3.)
        return phi
    
    def r0(self):
        """The virial radius"""
        return self.r200c
    
    def _initialize_phasespace(self):
        """private function. Initializes phase space calculation"""
        self.phasespace_initialized = True
        self.pss = phasespace.IsotropicPhaseSpaceSolver(self, rmin=self.scale("rmin"), rmax=self.scale("rmax"), rnorm=self.r200c, rbins=self.scale("pss_rbins"), nbinsE=self.scale("pss_ebins"), dlog_emin=np.log10(1+self.scale("pss_e_analytic_low")),  sample_profile_f=True)

    def f_of_e(self, energy):
        """The phase space distribution function of the NFW without anisotropy
        
        energy : (vector-like) energies to evaluate the distribution at
        
        returns : phase space density f(E) = dN/d3x/d3v
        """
        if not self.phasespace_initialized:
            self._initialize_phasespace()
        
        # For very low energy the DF integral diverges. That's why we just use the
        # asymptotic limit here. See also Widrow (2000)
        E0 = np.abs(self.phi0)
        #def extrap_lowenergy(E):
        #    return (1. + E/E0)**(-2.5)
        def f_widrow(e):
            estar = -e/E0

            f = (estar**1.5*(1-estar)**-2.5 *(-np.log(estar)/(1.-estar))**-2.7419
                 *np.exp(0.3620*estar-0.5639*estar**2. -0.0859*estar**3.-0.4912*estar**4.) )

            return f
        
        too_low = energy / E0 < self.scale("pss_e_analytic_low")
        too_high = energy / E0 > self._sc["pss_e_analytic_up"]
        
        #int_ok = (energy / E0 > self.scale("pss_e_analytic_low")) & (energy / E0 < self._sc["pss_e_analytic_up"])
        res = np.array(self.pss.f_of_e(energy, interpolate=True))
        
        norm_low = self.pss.f_of_e(self.scale("pss_e_analytic_low")*E0, interpolate=True) / f_widrow(self.scale("pss_e_analytic_low")*E0)
        norm_up = self.pss.f_of_e(self.scale("pss_e_analytic_up")*E0, interpolate=True) / f_widrow(self.scale("pss_e_analytic_up")*E0)
        res[too_low] = f_widrow(energy[too_low]) * norm_low
        res[too_high] = f_widrow(energy[too_high]) * norm_up
        
        return res * self.m_of_r(self.r0())

    def sample_particles(self, ntot=10000, rmax=None, seed=None):
        """Sample particles' positions, velocities and masses consistent with the NFW profile"""
        if not self.phasespace_initialized:
            self._initialize_phasespace()
        
        return self.pss.sample_particles(ntot=ntot, rmax=rmax, seed=seed)

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
        super().__init__()
        
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
    
    def to_string(self):
        return "powerlaw_slope=%.3f_rscale=%.5e_rhoc=%.5e" % (self.slope, self.rscale, self.rhoc)
    
    def to_dict(self):
        d = {}
        
        d["slope"] = self.slope
        d["rscale"] = self.rscale
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


class NumericalProfile(RadialProfile):
    def __init__(self, ri=None, rhoi=None, mass=None, r0=None, ancorphi="rmax", from_dict=None, potential_profile=None):
        """A radial profile of which only the density form is known
        
        ri : radius sampling points
        rhoi : densities
        r0 : base radius, will be maximum radius of the profile if not provided
        ancorphi : where to set the potential to zero? Can be 'rmax', 'rmin' or "infty"
        potential_profile : can be passed to use the potential from another profile
                            (might e.g. be relevant for Eddington inversion)
        
        from_dict : load a previous profile from a dict created by .to_dict()
        """
        super().__init__()
        
        self.potential_profile = potential_profile
        
        self.q = {}
        
        assert ancorphi in ("rmax", "rmin", "infty"), "Invalid value for ancorphi=%s" % ancorphi
        self.ancorphi = ancorphi
        
        if from_dict:
            self.from_dict(from_dict)
            return
        
        assert (ri is not None) & (rhoi is not None)

        if r0 is None:
            r0 = np.max(ri)
        self.base_radius = r0
        
        self.set_density_profile(ri, rhoi)
            
    def set_density_profile(self, ri, rhoi, update=True):
        """Change the bins that are used to bin the mass and solve the forces
        
        ri : radius sampling points
        rhoi : densities
        update : whether to update the mass, potential and force-profiles. Should always 
                 be "True" unless you know what you are doing
        """
        self.ri = ri
        self.q["rho"] = rhoi
        
        m0 = 4.*np.pi/3. * rhoi[0] * ri[0]**3
        self.q["mofr"] = m0 + mathtools.trapez_integral_cumulative(self.ri, 4.*np.pi*self.q["rho"]*ri**2)
        accr = - self.G * self.q["mofr"] / self.ri**2
        self.q["phi"] = - mathtools.trapez_integral_cumulative(self.ri, accr)
        
        self.phasespace_initialized = False
        self.potential_zero_at_infty = False

    def self_density(self, r):
        """Density in Msol/Mpc**3"""
        return np.interp(np.log10(r), np.log10(self.ri), self.q["rho"])
        
    def density(self, r):
        """Density in Msol/Mpc**3"""
        return self.self_density(r)
    
    def self_m_of_r(self, r):
        """The mass contained inside radius r"""
        return np.interp(np.log10(r), np.log10(self.ri), self.q["mofr"])
    
    def m_of_r(self, r):
        """The mass contained inside radius r"""
        if self.potential_profile is not None:
            return self.potential_profile.m_of_r(r)
        else:
            return self.self_m_of_r(r)

    def self_potential(self, r, zero_at_zero=False):
        """The gravitational  potential"""
        assert not zero_at_zero, "mode not implemented"
        dphi = np.interp(np.log10(r), np.log10(self.ri), self.q["phi"])
        if self.ancorphi == "rmax":
            return dphi - self.q["phi"][-1]
        else: # ancored at 0
            return dphi

    def potential(self, r, zero_at_zero=False):
        """The gravitational  potential"""
        if self.potential_profile is not None:
            return self.potential_profile.potential(r)
        else:
            return self.self_potential(r, zero_at_zero=zero_at_zero)
        
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
    
    def _initialize_phasespace(self):
        """private function. Initializes phase space calculation"""
        self.phasespace_initialized = True
        self.pss = phasespace.IsotropicPhaseSpaceSolver(self, rmin=self.scale("rmin"), rmax=self.scale("rmax"), rnorm=self.r0(), rbins=self.scale("pss_rbins"), nbinsE=self.scale("pss_ebins"), dlog_emin=np.log10(1+self.scale("pss_e_analytic_low")),  sample_profile_f=False)

    def f_of_e(self, energy):
        if not self.phasespace_initialized:
            self._initialize_phasespace()
            
        f = self.pss.f_of_e(energy)*self.m_of_r(self.r0())
        
        return f
    
    def sample_particles(self, ntot=10000, rmax=None, seed=None, res_of_r=None):
        """Sample particles' positions, velocities and masses consistent with the Numerical profile
        for more info see PhaseSpaceSolver.sample_particles"""
        if not self.phasespace_initialized:
            self._initialize_phasespace()
        
        return self.pss.sample_particles(ntot=ntot, rmax=rmax, seed=seed, res_of_r=res_of_r)
    
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
        
class RadialTidalProfile(RadialProfile):
    def __init__(self, alpha=0.):
        """A repulsive potential of form phi = -0.5*alpha*r**2
        
        alpha : eigenvalue of the tidal tensor. alpha>0 corresponds to a field
                stretching the mass distribution and leading to disruption.
                alpha < 0 does not make much sense in this context"""

        super().__init__()
        
        self.alpha = alpha
        self.rhoalpha = - 3.* self.alpha / (4.*np.pi*self.G)
        
    def density(self, r):
        """Density in Msol/Mpc**3"""
        return self.rhoalpha
    def drhodr(self, r):
        """Radial derivative of the density"""
        return 0.
    def m_of_r(self, r):
        """The mass contained inside radius r"""
        return - self.alpha/self.G * r**3
    def potential(self, r, zero_at_zero=False):
        """The gravitational potential"""
        return - 0.5 * self.alpha* r**2
    def daccdr(self, r):
        """The radial derivative of the acceleration"""
        return 3.*self.alpha/self.G * r**2
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
    
    
class AdiabaticProfile(RadialProfile):
    def __init__(self, prof_initial=None, prof_pert=None, tide=None, tidfac_rvir=None, tidfac_rs=None, norm_at_zero=False, rmin=None, rmax=None, nbins=150, ri=None, niter=0, h5cache=None, resetcache=False, verbose=False, store_every_step=False):
        """Adiabatically transformed profile
        
        Calculates the adiabatic transformation of an initial profile when
        it is slowly exposed to a given perturbation (e.g. a tidal field). In
        the adiabatic limit the phase space distribution function as a function
        of the actions f(J,L) is conserved in this process. Therefore, it is
        possible to approximate the final profile through an iterative procedure.
        See Sellwood & McGaugh (2005):
        (1) rho_{n+1} = integral_over_E_and_L(... * f_n(E,L) dL dE)
         -- f(E,L) = f(J(E,L),L) = f_ini(J(E,L), L) = f_ini(E(J(E,L), L))
            (all these depend implicitly on rho_{n} and phi_{n})
        (2) phi_{n+1} = ... solution to Poisson equation given rho_{n+1}
        Different than in Sellwood & McGaugh (2005), if the perturbation
        is a tidal field, some part of the J,L space does not exist in the
        perturbed profile, since the space of bound orbits is reduced through
        the tidal field. Therefore the tidal field leads to mass loss through
        changing the boundaries of the integration, and by modifying the mapping 
        J(E,L) (and vcirc(r))
        
        prof_initial : initial RadialProfile, for example and NFWProfile
        prof_pert : perturbation that should be added adiabatically
        tide : if prof_pert is None, add a tidal profile with this amplitude
               of the tidal field as the adiabatic perturbation
        tidfac_rvir : if prof_pert is None, add a tidal profile as the 
                      adiabatic perturbation. The tidal field will be
                      tidfac_rvir*Tvir where Tvir=acc(rvir)/rvir is the
                      tidal field necessary to have the (pre-revirialization)
                      tidal radius at the virial radius.
        tidfac_rs : Alternative way of specifying the tidal field in units
                    of the tidal field units at the scale radius Ts=acc(rs)/rs.
                    This parameter makes only sense for NFW profiles.
                    NFW profiles will behave equivalent at same tidfac_rs
        norm_at_zero : Wehther to normalize the potential at zero instead of infinity.
                    For powerlaw profiles it is needed to use True, for NFWs False
        rmin : Minimal radius that should be parameterized. (Default values are
               good usually)
        rmax : Maximum radius that should be parameterized. Will be overwritten by
               tidal radius during iterations (Default values are good usually) 
        nbins : Number of bins for parameterizing the profile. Usually nbins=100-200
                is good enough.
        ri : depreceated
        niter : Number of iterations to perform initially. Typically 10-30 iterations
                are sufficient.
        h5cache : If given, caches results into an hdf5file. If an entry with the same
                  intial profile, final profile and number of iterations is found, it
                  will load it.
        resetcache : if True, will delete the h5cachefile before doing any calculations
        verbose : degree of verbosity
        store_every_step : whether to store every step on initial iteration. This is
                  only useful if you plan to investigate the behavior versus the number
                  of iterations
        """
        super().__init__()
        
        self.prof_initial = prof_initial
        self.base_radius = self.prof_initial.r0()
        self.h5cache = h5cache
        self.verbose = verbose
        self.norm_at_zero = norm_at_zero
        self.is_disrupted = False
        self.iter_disrupted = 0
        
        if prof_pert is None:
            if tide is not None:
                alpha = tide
            elif tidfac_rvir is not None:
                assert tidfac_rs is None, "Can't proved tidfac_rs and tidfac_rvir"
                tid_rvir = np.abs(prof_initial.accr(prof_initial.r0()) / prof_initial.r0())
                alpha = tid_rvir * tidfac_rvir
            else:
                tid_rs = np.abs(prof_initial.accr(prof_initial.rs) / prof_initial.rs)
                alpha = tid_rs * tidfac_rs
            self.prof_pert = RadialTidalProfile(alpha=alpha)
        else:
            self.prof_pert = prof_pert

        self.nbins = nbins
        
        if rmin is None:
            rmin = 1e-12 * self.r0()
        if rmax is None:
            rmax = 1e5 * self.r0()
        self.rmin, self.rmax = rmin, rmax
        
        self._initialize_numerical_scales()
        
        if self.h5cache is not None:
            if resetcache & os.path.exists(self.h5cache):
                if self.verbose:
                    print("Resetting previous cache %s" % self.h5cache)
                os.remove(h5cache)
            
            loaded = self._try_load_from_h5cache(self.h5cache, verbose=self.verbose, iteration=niter)
            if loaded:
                return
        
        if ri is None:
            ri = np.logspace(np.log10(self.rmin), np.log10(self.rmax), self.nbins)
        else:
            print("Warning setting ri is depreceated and will be removed soon")

        self._update_profile(ri, self.prof_initial.density(ri))
        
        self._iteration = 0
        if niter > 0:
            self.iterate(niter, verbose=verbose, store_every_step=store_every_step)
            
    def _initialize_numerical_scales(self):
        """Sets some default values for numerical scales"""
        if self.prof_initial._sc is None:
            self.prof_initial._initialize_numerical_scales()
        self._sc = {}
        for kw in self.prof_initial._sc:
            self._sc[kw] = self.prof_initial.scale(kw)
        self._sc["rmin"] = self.rmin
        self._sc["rmax"] = self.rmax
            
    def set_numerical_scales(self, modify_initial_prof=True, **kwargs):
        """Sets numerical scales
        
        The default numerical scales are usually good enough. Only change the
        scales when you really need extra precision.
        
        Use for example like this:
        .set_numerical_scales(nbins_jr=100, fintegration_nstepsE=200)
        
        The list of possible keywords can be seen in the code of 
        ._initialize_numerical_scales()
        
        modify_initial_prof : whether to also set the scales of the initial profile
        """
        super().set_numerical_scales(**kwargs)
        
        if modify_initial_prof:
            print("I am also adapting the numerical scales of the initial profile")
            self.prof_initial.set_numerical_scales(**kwargs)
            self.prof_initial._initialize_e_of_jl(reinit=True)
            
    def _try_load_from_h5cache(self, h5cachefile, verbose=0, iteration=None):
        from . import h5methods
        import h5py
        if os.path.exists(h5cachefile):
            if verbose >= 2:
                print("Current keys in %s:" % h5cachefile)
                h5methods.h5py_print_file_structure(h5cachefile)
            
            h5path = "/" + self.to_string(iteration=iteration)
            with h5py.File(h5cachefile, "r") as h5file:
                if h5path in h5file:
                    if verbose:
                        print("Reading result of previous calculation from %s (i=%d)" % (h5cachefile, iteration))
                    if verbose >= 2:
                        print("\nkey=%s" % (h5path,))
                    mydict = h5methods.h5py_read_dict(h5file, path=h5path)
                    self.from_dict(mydict)
                    return True
                else:
                    if verbose:
                        print("No previous results found under %s" % (h5cachefile,))
                    if verbose >= 2:
                        print("\nkey=%s" % (h5path,))
        return False
    
    def _store_in_h5cache(self, h5cachefile, verbose=0):
        """Stores an entry in an hdf5 file that describes the current state of this profile."""
        from . import h5methods
        import h5py
        with h5py.File(h5cachefile, "a") as h5file:
            h5path = "/" + self.to_string()
            if verbose:
                print("Caching result of calculation to %s (i=%d)" % (h5cachefile, self._iteration))
            if verbose >= 2:
                print("\nkey=%s" % (h5path,))
            h5methods.h5py_write_dict(h5file, self.to_dict(), path=h5path, overwrite=True, verbose=verbose-1)
                
            
    def iterate(self, niter=1, verbose=True, store_every_step=False):
        """Perform adiabatic iterations
        
        See documentation of __init__ for a detailled explanation
        
        niter : number of iterations to perform
        verbose : if True, prints some information at each iteration
        store_every_step : if True and a h5cache was set, stores every step
                           (otherwise just the last step)
        """
        for i in range(0, niter):
            t0 = time.time()
            self._iteration += 1
            if not self.is_disrupted:
                self._initialize_tidal_radius()
                # Parameterize the new profile only up to the tidal radius, since rho(r > tid) = 0:
                rinew = np.logspace(np.log10(self.rmin), np.log10(np.min([self._rtid*(1+1e-8), self.rmax*(1.-1e-8)])), self.nbins)
                # calculate the new density given the current distribution function f(J,L):
                if self._rtid < self.rmin: # We have been disrupted
                    rhonew = np.zeros_like(rinew)
                else:
                    rhonew = self.rho_from_f(rinew)

                if np.max(rhonew) == 0.:
                    self.is_disrupted = True
                    self.iter_disrupted = self._iteration
                    if verbose:
                        print("This profile has been disrupted after %d iterations!" % self._iteration)
            # update the profile and solve poisson's equation etc...
                self._update_profile(rinew, rhonew)
            if verbose:
                print("i=%d, mfrac=%g, dt=%.1fs" % (self._iteration, self.self_m_of_r(self.r0()) / self.prof_initial.m_of_r(self.r0()), time.time()-t0))
                
            if store_every_step:
                self._store_in_h5cache(self.h5cache, verbose=self.verbose)
                
        if (self.h5cache is not None) & (not store_every_step):
            self._store_in_h5cache(self.h5cache, verbose=self.verbose)
            

    def _update_profile(self, ri, rhoi):
        """Changes the density profile and updates the potential, mass and acceleration"""
        if self.norm_at_zero:
            self.phioffset = self.prof_initial.potential(self.rmin, zero_at_zero=self.norm_at_zero)
        
        self.reset_interpolators()
        
        self.q = {}
        
        self.ri = ri
        self.q["rho"] = rhoi
        
        if self.is_disrupted: # This probably means we are disrupted
            def zero(x):
                return 0.
            
            self._ip_rhofr = zero
            self._ip_mofr = zero
            self._ip_mphi= zero
            
            self._initialize_tidal_radius()
            
            return
            
        self._ip_rhofr = mathtools.flexible_interpolator(self.ri, self.q["rho"], logx=True, logy=True, 
                                                         eps_for_logx=0., 
                                                         eps_for_logy=np.min(rhoi[rhoi>0.]))

        def dmdr(r):
            return 4.*np.pi*self.self_density(r)*r**2
        
        self.q["mofr"] = self.prof_initial.m_of_r(self.ri[0]) + mathtools.cum_simpson(dmdr, self.ri)
        self._ip_mofr = mathtools.flexible_interpolator(self.ri, self.q["mofr"], logx=True, logy=True, 
                                                        eps_for_logx=0.,
                                                        eps_for_logy=np.min(self.q["mofr"][self.q["mofr"]>0.]),
                                                        )

        def dphidr(r):
            return - self.self_accr(r)
        self.q["phi"] = mathtools.cum_simpson(dphidr, self.ri)

        if self.norm_at_zero:
            self.q["phi"] = self.q["phi"] + self.phioffset
            self._ip_mphi = mathtools.flexible_interpolator(self.ri, self.q["phi"], logx=True, logy=True, 
                                                            eps_for_logx=0.,
                                                            eps_for_logy=0.,
                                                            )
        else: # normalize potential to 0 at infinity -- assume there is no mass outside of rmax:
            dphi_inf_rmax = -self.G * self.q["mofr"][-1] / self.ri[-1]
            self.q["phi"] = self.q["phi"] - self.q["phi"][-1] + dphi_inf_rmax
            self._ip_mphi = mathtools.flexible_interpolator(self.ri, -self.q["phi"], logx=True, logy=True, 
                                                            eps_for_logx=0.,
                                                            eps_for_logy=0.,
                                                            )
        self._initialize_tidal_radius()

    def f_of_el(self, e, l, r=None, get_eini=False):
        """distribution function as a function of energy and angular momentum.
        
        calculated through adiabatic matching to the initial profile:
        f(e,l) = f(j(e,l), l) = f_ini(j(e,l), l) = f_ini(eini(j(e,l), l))
        
        j(e,l) and eini(jini, lini) are parametereized through interpolators.
        See the functions .j_of_el and .e_of_jl.
        
        For unbound orbits: f(e,l) = 0
        
        e : energy
        l : angular momentum
        r : radius (only needed if profile has a tidal boundary)
        get_eini : if True, also returns energy of corresponding particles in 
                   the initial profile
        
        returns : the phase space distribution function
        """
        
        self._initialize_tidal_radius()

        rtid,phitid = self._rtid, self._phitid
        rlmax, philmax, lmax = self._rlmax, self._philmax, self._lmax
        
        lmin_of_e, lmax_of_e = self.lminmax_of_e(e)
        e = e * np.ones_like(l)
            
        valid = np.ones(l.shape, dtype=np.bool)
        if self._has_tidal_radius:
            valid &= (l >= lmin_of_e) & (l < lmax_of_e) 
            if r is not None:
                # this check is necessary since low, but unbound
                #energy values can also be reached beyond rtid
                valid &= r <= self._rtid 
        else:
            if self.norm_at_zero:
                valid &= e < np.max(self.q["phi"])
            else:
                valid &= e < 0.
        
        j = self.j_of_el(e[valid], l[valid])
        jvalid = (j >= 0.) & (~np.isnan(j))
        valid[valid] &= jvalid
        
        eini = self.prof_initial.e_of_jl(j[jvalid],l[valid])
        fini = self.prof_initial.f_of_e(eini)

        f = np.zeros_like(l)
        f[valid] = fini
        
        assert np.sum(np.isnan(f)) == 0

        if get_eini:
            eini_out = np.ones_like(l) * np.nan
            eini_out[valid] = eini
            return f, eini_out
        else:
            return f

    def rho_from_f(self, r, nstepsE=None, nstepsL=None, use_initial=False):
        """calculate the density from the phase space distribution function
        
        nstepsE : number of integration steps in energy direction
        nstepsL : number of integration steps in angular momentum direction
        use_initial : if set to True, will directly assume f(E,L) = fini(E,L). This
                 option is only useful to see how well the initial profile can be
                 reconstructed from this integration -- as a benchmark of the method
        
        If is constructed through adiabatic matching to the initial profile. See
        the documentation of __init__() and also consider Sellwood & McGaugh (2005).
        Especially their equation (3)
        (https://iopscience.iop.org/article/10.1086/491731/pdf)
        
        returns : the density profile evaluated at radii r
        """
        if nstepsE is None:
            nstepsE = self.scale("fintegration_nstepsE")
        if nstepsL is None:
            nstepsL = self.scale("fintegration_nstepsL")
            
        assert (np.min(r) >= self.rmin * 0.999) & (np.max(r) <= self.rmax * 1.001)
   
        self._initialize_tidal_radius()
        if self._has_tidal_radius: # Test for disruption
            rtest = np.logspace(np.log10(self.rmin), np.log10(self.rmax), 100)
            if np.max(self.density(rtest)) <= 0:
                # we have nowhere an attractive force!
                # this means that we are disrupted!
                # Better to stop here, to avoid all kinds of errors based
                # on unfullfilled assumptions
                return np.zeros_like(r)
            
    
        rho = np.zeros_like(r)
        valid_r = r < self._rtid * (1.-1e-8)
        r = r[valid_r]
        
        if use_initial:
            phir = self.prof_initial.potential(r)
            #phir = self.potential(r)
            def integrand(r, e, l):
                u = np.sqrt(2.*(e - phir[:,np.newaxis,np.newaxis]) - l**2/r**2)
                #assert np.max(np.isnan(u)) == False
                f = self.prof_initial.f_of_e(e)
                assert np.max(np.isnan(f)) == False
                assert np.max(np.isnan(r)) == False
                assert np.max(np.isnan(l)) == False
                integ =  l*f / (r*r*u)
                
                integ[u==0.] = 0.
                integ[np.isnan(u)] = 0.
                assert np.max(np.isnan(integ)) == False
                return integ
        else:
            phir = self.potential(r)
            def integrand(r, e, l):
                u = np.sqrt(2.*(e - phir[:,np.newaxis,np.newaxis]) - l**2/r**2)
                integ = l*self.f_of_el(e, l, r=r) / (r*r*u)
                integ[np.isnan(u) | (u==0.)] = 0.
                assert np.max(np.isnan(integ)) == False
                return integ

        rtid,phitid = self._rtid, self._phitid
        rlmax, philmax, lmax = self._rlmax, self._philmax, self._lmax
        
        emin = phir

        if self._has_tidal_radius: # maximum possible energy is given by maximum angular momentum radius
            emax = np.zeros_like(r)
            lcirc = r*self.vcirc(r)
            sel = r <= self._rlmax
            emax[sel] = self._philmax + 0.5 * self._lmax**2 / self._rlmax**2
            emax[~sel] = phir[~sel] + 0.5*lcirc[~sel]**2/r[~sel]**2
        else:
            emax = self.potential(self.rmax)
            
        assert np.sum(emax <= emin) == 0

        efac = mathtools.bins_log_lin_log(0., 0.1, 0.9, 1.0, n1=nstepsE//3, n2=nstepsE//3, n3=nstepsE//3, dlogmin=self.scale("log_emin"), dlogmin_up=self.scale("log_emin_up"))
        lfac = mathtools.bins_log_lin_log(0., 0.1, 0.9, 1.0, n1=nstepsL//3, n2=nstepsL//3, n3=nstepsL//3, dlogmin=self.scale("log_lmin"))
        
        grid3_r, grid3_efac, grid3_lfac = np.meshgrid(r, efac, lfac, indexing="ij")
        grid2_e = emin[:,np.newaxis] + grid3_efac[:,:,0] * (emax - emin)[:,np.newaxis]
        grid2_de = grid3_efac[:,:,0] * (emax - emin)[:,np.newaxis]

        grid2_lmin, _ = self.lminmax_of_e(grid2_e)
        grid2_lmax = r[...,np.newaxis]*np.sqrt(2.*(grid2_de))
        grid2_lmin = np.clip(grid2_lmin, 0., grid2_lmax*0.999999) # This is here for numerical reasons
 
        assert np.sum(grid2_lmin > grid2_lmax) == 0

        assert np.sum(np.isnan(grid2_lmax)) == False
        grid3_l = (grid2_lmax-grid2_lmin)[:,:,np.newaxis] * grid3_lfac + grid2_lmin[:,:,np.newaxis]
        
        # There are some by-zero division errors coming from f=0 regions which are hard to get rid off.
        # We simply mute these errors, since the affected values don't contribute to the result
        float_err_handling = np.geterr()
        np.seterr(divide="ignore", invalid="ignore") 
        
        fgrid3 = integrand(r[:,np.newaxis,np.newaxis], grid2_e[:,:,np.newaxis], grid3_l)
        
        assert np.sum(np.isnan(fgrid3)) == 0
        res = mathtools.simpson_2d(fgrid3, grid2_e, grid3_l, axisx=1, axisy=2)
        assert np.sum(np.isnan(res)) == 0
        
        np.seterr(**float_err_handling)
        
        rho[valid_r] = res * 4.*np.pi
        return rho
    
    def density(self, r):
        """Density in Msol/Mpc**3"""
        return self.self_density(r) + self.prof_pert.density(r)
    def potential(self, r, zero_at_zero=False):
        """The gravitational  potential"""
        return self.self_potential(r, zero_at_zero=zero_at_zero) + self.prof_pert.potential(r, zero_at_zero=zero_at_zero)
    def m_of_r(self,r):
        """The mass contained inside radius r"""
        return self.self_m_of_r(r) + self.prof_pert.m_of_r(r)

    def self_density(self, r):
        """Density in Msol/Mpc**3"""
        return self._ip_rhofr(r)
    def self_m_of_r(self, r):
        """The mass contained inside radius r"""
        mofr = self._ip_mofr(r)
        #assert np.min(mofr) >= 0.
        return mofr
    def self_potential(self, r, zero_at_zero=False):
        """The gravitational  potential"""
        r = np.array(r)
        sel = r > self.ri[-1]
        if self.norm_at_zero:
            dphi = np.array(self._ip_mphi(r))
            dphi[sel] = self.q["phi"][-1] - (self.G * self.q["mofr"][-1] / r[sel] - self.G * self.q["mofr"][-1] / self.ri[-1])
        else:
            dphi = np.array(-self._ip_mphi(r))
            dphi[sel] = -self.G * self.q["mofr"][-1] / r[sel]
        return dphi

    def r0(self):
        """A scale radius. Typically rvir"""
        return self.base_radius
    
    def to_string(self, iteration=None):
        #assert isinstance(self.prof_initial, NFWProfile), "Caching only works with NFW so far"
        #assert isinstance(self.prof_pert, RadialTidalProfile), "Caching only works with Tidal profiles so far"
        
        if iteration is None:
            iteration = self._iteration
        
        string = "adiabatic_nbins=%d_rmin=%.5e_rmax=%.5e_iter=%d__%s__%s" % (self.nbins, self.rmin, self.rmax, iteration, self.prof_initial.to_string(), self.prof_pert.to_string())
        
        return string
        
    def to_dict(self):
        """Returns a dictionary with all (local) variables that describe the current state"""
        d = {}
        
        d["iteration"] = self._iteration
        d["nbins"] = self.nbins
        d["rmin"] = self.rmin
        d["rmax"] = self.rmax
        d["base_radius"] = self.base_radius
        
        d["numerical_scales"] = dict(self.scaledict())
        d["profini_numerical_scales"] = dict(self.prof_initial.scaledict())
        
        d["ri"] = self.ri
        d["rhoi"] = self.q["rho"]
        
        d["is_disrupted"] = self.is_disrupted
        d["iter_disrupted"] = self.iter_disrupted

        return d
    
    def from_dict(self, d):
        self._iteration = d["iteration"]
        self.nbins = d["nbins"]
        self.rmin = d["rmin"]
        self.rmax = d["rmax"]
        self.base_radius = d["base_radius"]
        if self.rmin == self.rmax:
            print("Warning, I am using a dirty fix to a broken old cache, better regenerate")
            self.rmax = self.base_radius * 1e5

        if "is_disrupted" in d: # This keyword is new, but old files should still be fine
            self.is_disrupted = d["is_disrupted"]
            self.iter_disrupted  = d["iter_disrupted"]
        else:
            self.is_disrupted = False
            self.iter_disrupted = 0
        

        # Check whether none of the numerical scales have been changed. These are implicit parameters
        for kw in d["numerical_scales"]:
            if d["numerical_scales"][kw] != self.scaledict()[kw]:
                print("Warning: numerical scale '%s' differs (previous=%g, current=%g)\nIf you are unsure whether this is save, reset the cache" % (kw, d["numerical_scales"][kw], self.scaledict()[kw]))
            ##print(kw, d["numerical_scales"][kw] == self.scaledict()[kw])
        for kw in d["profini_numerical_scales"]:
            if d["profini_numerical_scales"][kw] != self.prof_initial.scaledict()[kw]:
                print("Warning: initial profile's numerical scale '%s' differs (previous=%g, current=%g)\nIf you are unsure whether this is save, reset the cache" % (kw, d["profini_numerical_scales"][kw], self.prof_initial.scaledict()[kw]))

        self._update_profile(d["ri"], d["rhoi"])
        
    def sample_particles_uniform(self, ntot=1e4, rmax=None, seed=None, nsteps_chain=300, res_of_r=None):
        """Samples particles so that their masses are uniform
        This is done by using a Metropolis Hastings algorithm and is
        therefore rather time-consuming
        
        ntot : number of particles to sample
        rmax : maximum radius to sample to
        seed : random seed
        nsteps_chain : number of steps to use. Higher is better, but takes longer. 
               I recommend to use n >~ 300 to be sure the chain is well converged
        res_of_r: (optional) a function that returns a resolution weight as a 
               function of r. The number of particles at radius r will be 
               proportional to this weight and the mass will be inversely proportional
        """
        if seed is not None:
            np.random.seed(seed)
        
        if rmax is None:
            rmax = self.rmax
        
        def sample_r(ntot):
            if res_of_r is None:
                mmax = self.m_of_r(rmax)
                fsamp = np.random.uniform(0., 1., ntot)
                rsamp = np.interp(fsamp, self.q["mofr"]/mmax, self.ri)
                mass = np.ones(ntot) * (mmax / ntot)
            else:
                def dmdr(r):
                    return 4.*np.pi*self.self_density(r)*r**2 * res_of_r(r)
                
                meff = self.prof_initial.m_of_r(self.ri[0]) + mathtools.cum_simpson(dmdr, self.ri)
                mmax = np.interp(rmax, self.ri, meff)
                
                fsamp = np.random.uniform(0., 1., ntot)
                rsamp = np.interp(fsamp, meff/mmax, self.ri)
                
                mass = mmax / ntot / res_of_r(rsamp)

            return rsamp, mass
        
        rsamp, mass = sample_r(ntot)
        phi = self.potential(rsamp)
        emax = self.potential(100.*rmax)
        
        def likelihood_of_el_given_r(el):
            r = rsamp
            e,l = el[...,0], el[...,1]
            
            valid = (e >= phi) & (e <= emax) & (l >= 0.) & (l**2 <= r**2*(2.*(e-phi)))

            u = np.sqrt(2.*(e[valid]-phi[valid] - 0.5*l[valid]**2/r[valid]**2))
            
            res = np.zeros_like(e)
            res[valid] = l[valid]*self.f_of_el(e[valid],l[valid]) / (r[valid]**2 * u)
            return res

        #e0 = np.random.uniform(phi, self.potential(rmax*10.), rsamp.shape)
        e0 = phi + np.random.uniform(0., 0.5, rsamp.shape) * self.vcirc(rsamp)**2
        lmax = rsamp*np.sqrt(2.*(e0-phi))
        lcirc = self.vcirc(rsamp) * rsamp
        l0 = np.random.uniform(0., lmax, e0.shape)
        
        el = np.stack([e0,l0], axis=-1)
        #stepsize = np.stack([0.2*(emax-phi), 0.2*lcirc], axis=-1)
        stepsize = np.stack([0.2*self.vcirc(rsamp)**2, 0.2*lcirc], axis=-1)
        
        el = mathtools.sample_metropolis_hastings(likelihood_of_el_given_r, el, stepsize=stepsize, nsteps=nsteps_chain)
        
        e, l = el[...,0], el[...,1]
        
        uvpos = mathtools.random_direction(rsamp.shape, 3)
        pos = uvpos * rsamp[...,np.newaxis]
        
        # radial vector
        uvvr = uvpos 
        # random non-radial vector
        uvvl = mathtools.random_direction(rsamp.shape, 3) 
        uvvl = uvvl - np.sum(uvvr*uvvl, axis=-1)[...,np.newaxis] * uvvr
        uvvl = uvvl / np.linalg.norm(uvvl, axis=-1)[...,np.newaxis]
        
        vr = np.sqrt(2.*(e - phi - 0.5*l**2/rsamp**2))
        
        vel = uvvr * vr[...,np.newaxis] + uvvl*(l/rsamp)[...,np.newaxis]
        
        return pos, vel, mass #, rsamp, e, l
        
    def sample_particles(self, ntot=1e4, rmax=None, seed=None, extra_outputs=False):
        """Samples particles, by sampling the initial profile and reweighting their
        masses so that they match the final profile
        
        ntot : number of particles to sample
        rmax : maximum radius to sample to
        seed : a random seed
        extra_outputs : if True, will output pos,vel,mass, r,efinal,eself,eini,L, otherwise just pos,vel,mass
        """
        
        self._initialize_tidal_radius()
        if rmax is None:
            rmax = np.min([self._rtid, self.r0()])
        
        pos,vel,mass = self.prof_initial.sample_particles(ntot=ntot, rmax=rmax, seed=seed)
        
        r, e0, L = self.prof_initial.posvel_to_rEL(pos, vel)
        efinal = self.potential(r) + 0.5*np.sum(vel**2, axis=-1)
        
        float_err_handling = np.geterr()
        np.seterr(divide="ignore", invalid="ignore") 

        fi, eini = self.f_of_el(efinal, L, r=r, get_eini=True)
        f0 = self.prof_initial.f_of_e(e0)
        
        mass = mass * fi/f0
        
        mass[np.isnan(mass)] = 0.
        
        np.seterr(**float_err_handling)
        
        if not extra_outputs:
            return pos,vel,mass
        else:
            eself = self.self_potential(r) + 0.5*np.sum(vel**2, axis=-1)
            return pos,vel,mass, r,efinal,eself,eini,L