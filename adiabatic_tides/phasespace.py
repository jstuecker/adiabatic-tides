import numpy as np
from scipy.integrate import quad
import scipy.optimize as op
from . import mathtools
from scipy.interpolate import interp1d, NearestNDInterpolator, LinearNDInterpolator
        

class IsotropicPhaseSpaceSolver():
    def __init__(self, profile, rbins=1000, rnorm=None, rmax=None, rmin=None, potential_profile=None, sample_profile_f=False, nbinsE=100, dlog_emin=-5, force_profile_f=False):
        """This is a utility class that enables numerically calculating 
        distribution functions in arbitrary spherical profiles. It has only
        been tested for NFW models. On startup this class precomputes the 
        distribution functions and later evaluates them through interpolation
        
        The phase space distribution is calculated numerically according to
        Eddington inversion
        
        The sampling part is done as explained in Errani & Penarrubia (2020)
        (arxiv:1906.01642)
        
        profile : the profile to calculate the distribution function for
        rbins : the number of radial bins. Can also be array to set bins directly
        rnorm : the radius used for normalizing the distrubtion function. 
               (rather unimportant usually)
        rmax : the maximum radius considered for calculating the distribution function
               (Should be way larger than the radius of your profile)
        rmin : the minimal radius considered for calculating the distribution function
               (Should be very close to zero, necessary for avoiding singularities at 0)
        potential_profile : If provided use a different profile to generate the potential
                            than for evaluating the density.
        sample_profile_f : if True, when sampling particles the phase space distriubtion
               will use the function p.f_of_e(e) of the profile when sampling particles
               instead of the own estimate of f_of_e
        force_profile_f : if True, will always consider the function p.f_of_e(e). This is
               useful if using a profile with analytically known f_of_e
        """
        
        self.profile = profile
        self.sample_profile_f = sample_profile_f
        self.force_profile_f = force_profile_f
        
        if potential_profile is None:
            potential_profile = profile
        self.potential_profile = potential_profile
        
        if rnorm is None:
            self.rnorm = self.profile.r0()
        else:
            self.rnorm = rnorm
        
        if rmax is None:
            self.rmax = self.rnorm*10.
        else:
            self.rmax = rmax
            
        if rmin is None:
            self.rmin = self.rmax / 1e8
        else:
            self.rmin = rmin

        if  np.issubdtype(type(rbins), np.integer):
            self.ri = np.logspace(np.log10(self.rmin), np.log10(self.rmax), rbins)
        else:
            self.ri = rbins
            
        self.psi = -(self.potential_profile.potential(self.ri) )
        self.nu = self.profile.self_density(self.ri) / profile.self_m_of_r(self.rnorm)
        
        def filter_nonequal(*xis):
            valid = np.ones_like(xis[0], dtype=np.bool)
            for xi in xis:
                valid[1:] &= np.abs(xi[1:] - xi[:-1]) >= (np.abs(xi[1:]) + np.abs(xi[:-1])) * 1e-12
            xinew = [xi[valid] for xi in xis]
            return xinew
        
        self.psi, self.nu = filter_nonequal(self.psi, self.nu)

        valid = self.psi[1:] > self.psi[:-1] # some bins might be equal due to roundoff-errors
        self.psi = self.psi
        
        self.dndp = np.gradient(self.nu, self.psi, edge_order=2)
        self.d2nd2p = np.gradient(self.dndp, self.psi, edge_order=2)
        
        self.psi, self.nu, self.dndp, self.d2nd2p = filter_nonequal(self.psi, self.nu, self.dndp, self.d2nd2p)
        
        self.dlog_emin = dlog_emin
        
        self.eps = 1e-6
        self.nbinsE = nbinsE
        
        self._setup_interpolators()
        
    def sample_particles(self, ntot=10000, rmax=None, seed=None, verbose=False, res_of_r=None):
        """Samples particles consistent with the phasespace distribution
        
        ntot : number of particles to sample
        rmax : Maximum radius to sample to. This is important for profiles
               which diverge M(r -> infty) -> infty
        seed : random seed
        res_of_r: (optional) a function that returns a resolution weight as a 
               function of r. The number of particles at radius r will be 
               proportional to this weight and the mass will be inversely proportional
        
        returns : pos, vel, mass  -- the positions, velocities and masses
               masses are normalized so that Sum(mass) = M(<rmax)
        """
        float_err_handling = np.geterr()
        np.seterr(divide="ignore", invalid="ignore") 
        
        if seed is not None:
            np.random.seed(seed)
        if rmax is None:
            rmax = self.rnorm
            
        ri, mass = self._sample_r(ntot, rmax=rmax, res_of_r=res_of_r)
        ei = self._sample_e_given_r(ri, verbose=verbose)
        
        pos = mathtools.random_direction(ri.shape, 3) * ri[...,np.newaxis]
        
        phi = self.potential_profile.potential(ri)
        vi = np.sqrt(2.*(ei-phi))
        vel = mathtools.random_direction(ri.shape, 3) * vi[...,np.newaxis]
        
        
        np.seterr(**float_err_handling)
        
        return pos,vel,mass
    
    # !!! This function is depreceated and can be removed soon:
    def sample_test_particles(self, ntot=10000, rmin=None, rmax=None, seed=None, fpow=1.):
        """Samples test tracers that are not following the distribution function
        
        The idea is to have particles which widely span the space of possible
        actions/angular momenta/energies. This is useful for setting up mesh-free
        interpolators. Particles will be uniform in log(r)
        
        ntot : number of particles to sample
        ri : radii of already given particles
        rmin : minimal radius
        rmax : Maximum radius
        seed : random seed
        fpow : take this power of the f(E | r) distribution function. Use
               0 < fpow < 1 to increase the likelihood to sample unlikely
               energies
        
        returns : pos, vel  -- the positions and velocities
        """
        if seed is not None:
            np.random.seed(seed)
        if rmin is None:
            rmin = self.rmin
        if rmax is None:
            rmax = self.rmax
            
        ri = 10**np.random.uniform(np.log10(rmin),np.log10(rmax), int(ntot))
        
        ei = self._sample_e_given_r(ri, fpow=fpow)
        
        pos = mathtools.random_direction(ri.shape, 3) * ri[...,np.newaxis]
        
        phi = self.potential_profile.potential(ri)
        vi = np.sqrt(2.*(ei-phi))
        vel = mathtools.random_direction(ri.shape, 3) * vi[...,np.newaxis]
        
        #mass = np.ones_like(ri) * self.profile.m_of_r(rmax) / len(ri)
        
        return pos,vel#,mass
        
    def _interpolator(self, xi, yi):
        ip = interp1d(xi, yi, kind="cubic", fill_value="extrapolate")
        
        return ip
    
    def _interpolate(self, xeval, xi, yi, getip=False):
        ip = self._interpolator(xi, yi)
        
        return ip(xeval)
        
    def f_of_e(self, e, interpolate=True, use_profile_f=False):
        """Energy distribution per phase space volume. f(e) = dN/d3x/d3v
        
        e : energy to evaluate the distribution function
        interpolate : should be True, unless you are benchmarking the numerics.
             If True uses the precomputed interpolators to evalute the
             phase space distribution rapidly. If False, calculates the phase
             space distribution by numerically solving the integrals.
        use_profile_f : instead use the profiles f_of_e. Use this if the profile
             has an implemented analytic solution or modifies the behavior of this
             function
        """
        if use_profile_f or self.force_profile_f:
            return self.profile.f_of_e(e)
        
        if interpolate:
            return 10.**self.ip_e_logf(e)
        
        # arxiv:1906.01642 eq 3
        ip = self._interpolator(self.psi[::-1], self.d2nd2p[::-1])
        def integrand(p, ei):
            return ip(p)  / np.sqrt(ei-p)
        
        if self.potential_profile.potential_zero_at_infty:
            emax = 0.
        else:
            emax = self.potential_profile.potential(self.profile.scale("rmax"))
        
        def integral(ei):
            return quad(integrand, -emax, ei,  epsrel=self.eps, args=(ei,), full_output=True)[0]

        return 1./(np.sqrt(8.) * np.pi**2) * np.vectorize(integral)(-e)
        
        #assert np.min(e[1:] >= e[:-1]) == True
        #return 1./(np.sqrt(8.) * np.pi**2) * mathtools.cum_simpson(integrand, ei)
        
        
    
    def ntot_of_e(self, e):
        """Numer of particles per energy.
        
        This is proportional to what you would get from a mass-weighted histogram 
        of particles. Includes the effect of the phase space volume and the distribution function."""
        return (4.*np.pi)**2 * self.f_of_e(e) * self._g_of_e(e)
    
    def _setup_interpolators(self):
        self.phimin = self.potential_profile.potential(self.rmin)
        self.phimax = self.potential_profile.potential(self.rmax)

        #n1,n2,n3 = self.nbinsE//4, self.nbinsE//2, self.nbinsE//4
        #efac = np.concatenate([np.logspace(-6,-1, n1), np.linspace(0.101,0.899, n2), 1.-np.logspace(-1, -6, n3)])
        efac = mathtools.bins_log_lin_log(0., 0.1, 0.9, 1.0, self.nbinsE//4, self.nbinsE//2, self.nbinsE//4, dlogmin=self.dlog_emin)
        self.ei = self.phimin + (self.phimax - self.phimin)*efac
        
        self.fi = self.f_of_e(self.ei, interpolate=False)
        self.gi = None
        
        self.ei = self.ei[self.fi > 0.]
        self.fi = self.fi[self.fi > 0.]
        
        assert np.min(self.fi) > 0.
        
        self.ip_e_logf = self._interpolator(self.ei, np.log10(self.fi))
        
        self.maxLikelihood_of_r = None

    def _rapo_zeroL(self, e):
        """Radius at apocenter for zero angular momentum orbits"""
        def perifunc(rp, ei):
            return self.potential_profile.potential(rp) - ei
        
        def apo_single(ei):
            return op.brentq(perifunc, 1e-2*self.rmin, 1e2*self.rmax, args=(ei,))
        
        return np.vectorize(apo_single)(e)

    def _g_of_e(self, e, interpolate=True):
        """Phase space volume associated with Energy e"""
        if interpolate:
            if self.gi is None:
                self.gi = self._g_of_e(self.ei, interpolate=False)
            return np.interp(e, self.ei, self.gi)

        # arxiv:1906.01642 eq 4
        def rapo(ei):
            return self._rapo_zeroL(ei)
        
        def integrand(r, ei):
            de = ei - self.potential_profile.potential(r)
            
            return r**2 * np.sqrt(2.*de)

        def integral(ex):
            return quad(integrand, self.rmin, rapo(ex), epsrel=self.eps, args=(ex,))[0]
        
        return np.vectorize(integral)(e)
    
    def _likelihood_of_e_given_r(self, e, r, maxnormed=True, use_profile_f=False):
        """The probability distribution of energies, given a radius"""
        
        if maxnormed:
            if self.maxLikelihood_of_r is None:
                self.maxLikelihood_of_r = np.max(self._likelihood_of_e_given_r(self.ei, self.ri[...,np.newaxis], maxnormed=False, 
                                                use_profile_f=use_profile_f), axis=-1)
            
            norm = np.interp(r, self.ri, self.maxLikelihood_of_r)
        else:
            norm = 1.
        
        de = np.clip(e - self.potential_profile.potential(r), 0., None)
        
        return r**2 * np.sqrt(2.*de)  * self.f_of_e(e, use_profile_f=use_profile_f) / norm
    
    def _sample_r(self, ntot=100, rmax=None, res_of_r=None):
        """Sample radii and masses from the density profile, 
        res_of_r can be a function increasing resolution and decreasing mass"""
        if rmax is None:
            rmax = self.rnorm
            
        if res_of_r is None:
            mofr = self.profile.self_m_of_r(self.ri)
            mmax = self.profile.self_m_of_r(rmax)
            fsamp = np.random.uniform(0., 1., ntot)
            rsamp = np.interp(fsamp, mofr/mmax, self.ri)
            mass = np.ones(ntot) * (mmax / ntot)
        else:
            def dmdr(r):
                return 4.*np.pi*self.profile.self_density(r)*r**2 * res_of_r(r)

            meff = self.profile.self_m_of_r(self.ri[0]) + mathtools.cum_simpson(dmdr, self.ri)
            mmax = np.interp(rmax, self.ri, meff)

            fsamp = np.random.uniform(0., 1., ntot)
            rsamp = np.interp(fsamp, meff/mmax, self.ri)

            mass = mmax / ntot / res_of_r(rsamp)
        mass[np.isnan(mass)] = 0.

        return rsamp, mass
    
        def sample_r(ntot):
            if res_of_r is None:
                mmax = self.self_m_of_r(rmax)
                fsamp = np.random.uniform(0., 1., ntot)
                rsamp = np.interp(fsamp, self.q["mofr"]/mmax, self.ri)
                mass = np.ones(ntot) * (mmax / ntot)
            else:
                def dmdr(r):
                    return 4.*np.pi*self.self_density(r)*r**2 * res_of_r(r)
                
                meff = self.prof_initial.self_m_of_r(self.ri[0]) + mathtools.cum_simpson(dmdr, self.ri)
                mmax = np.interp(rmax, self.ri, meff)
                
                fsamp = np.random.uniform(0., 1., ntot)
                rsamp = np.interp(fsamp, meff/mmax, self.ri)
                
                mass = mmax / ntot / res_of_r(rsamp)

            return rsamp, mass
        
        rsamp, mass = sample_r(ntot)
    
    def _sample_e_given_r(self, ri, verbose=False, maxiter=10000, fpow=None):
        """Use rejection sampling like in arxiv:1906.01642 to find energies
        of particles, when already given their radius
        
        fpow : take a power of the likelihood. Should be None or 1, unless
               you know what you are doing
        """
        phimax = self.potential_profile.potential(self.rmax)

        phii = self.potential_profile.potential(ri)
        
        ei = np.zeros_like(phii)
        ileft = np.arange(len(ri))
        for i in range(0,maxiter):
            esamp = np.random.uniform(phii[ileft], phimax, ileft.shape)
            Lsamp = self._likelihood_of_e_given_r(esamp, ri[ileft], use_profile_f=self.sample_profile_f)
            if fpow is not None:
                Lsamp = Lsamp**fpow
                
            ysamp = np.random.uniform(0., 1.1, ileft.shape) # 10 % buffer
            sel_pass =  ysamp <= Lsamp
            
            ei[ileft[sel_pass]] = esamp[sel_pass]
            ileft = ileft[~sel_pass]
            
            if verbose:
                print("iter = %d, left=%d (%.1g %%)" % (i, len(ileft), len(ileft)*100./len(ri)))
                
            if len(ileft) == 0:
                break
                
            if i == maxiter-1:
                print("Warning couldn't make it after maxiter=%d iterations (%d particles left)" % (maxiter, len(ileft)))
            
        return ei