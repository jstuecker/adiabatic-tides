from .radial_profile import RadialProfile
import numpy as np
from .. import numerics
import zlib

class NumericalProfile(RadialProfile):
    def __init__(self, ri=None, rho=None, mass=None, r0=None, ancorphi="rmin", from_dict=None, boundary="powerlaw", anisotropy=0., **configs):
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
            
    def set_density_profile(self, ri, rho, update=True):
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

        self.ip_rho, self.ip_m, self.ip_phi = numerics.integrate.solve_poisson_via_spline_with_smart_boundaries(ri, rhoi, lower_boundary=self.boundary, G=self.G)
        self.q["rho"], self.q["mofr"], self.q["phi"] = rho, self.ip_m(self.ri), self.ip_phi(self.ri)

        if callable(rho):
            self.ip_rho = rho

        self.potential_zero_at_infty = False

    def density(self, r):
        return self.ip_rho(r)
    
    def m_of_r(self, r):
        return self.ip_m(r)

    def potential(self, r, zero_at_zero=True):
        return self.ip_phi(r)
        
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

    def __str__(self):
        return f"NumericalProfile with {len(self.ri)} points in ({self.ri[0]:.5e}, {self.ri[-1]:.5e})"

    def __repr__(self):
        s =  super().__repr__()
        s += "\nHash:"
        s += f"\n  ri={zlib.adler32(self.ri.data.tobytes())}"
        s += f"\n  rho={zlib.adler32(self.q['rho'].data.tobytes())}"
        return s

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
        self.q["phi"] = - numerics.integrate.trapez_integral_cumulative(self.rbins, accr)

    def _update(self):
        """Recalculate the density/mass/gravity profiles"""
        #self.q["rho"], self.q["mofr"] = mathtools.get_mass_profile(self.ri, self.mi, self.rbins)
        rho, m = numerics.sample.get_mass_profile(self.ri, self.mi, self.rbins)
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
            or a dictionary containing variables "r", "m" and optionally "vr" and "l"
        """
        self.rbins = rbins
        self.ri = np.sqrt(rbins[1:]*rbins[:-1])

        self.set_particles(particles, update=False)

        rho, mprof = numerics.sample.get_mass_profile(self.p["r"], self.p["m"], self.rbins)

        super().__init__(self.ri, rho, mprof, boundary="constant")

    def _update_mass_profile(self):
        rho, mprof = numerics.sample.get_mass_profile(self.p["r"], self.p["m"], self.rbins)
        super().set_density_profile(self.ri, rho)

    def set_particles(self, particles, update=True):
        """
        particles -- can either be (r,m) or (r,m,vr,l) or (pos,vel,m)
        """
        self.p = {}
        if isinstance(particles, dict):
            for key in "r", "m", "vr", "l":
                if key in particles:
                    self.p[key] = particles[key]
        elif len(particles) == 2:
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
        else:
            raise ValueError("Invalid input for particles")
        
        if update:
            self._update_mass_profile()

    def integrate_orbits_in_other_potential(self, accr, tmax, nsteps=1000, update=True):
        self.p["r"], self.p["vr"] = numerics.sample.integrate_radial_orbits(accr, self.p["r"], self.p["vr"], self.p["l"], tmax, nsteps=nsteps)

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

    def __str__(self):
        return f"ParticleProfile with {len(self.p['r'])} particles in {len(self.ri)} bins in ({self.ri[0]:.5e}, {self.ri[-1]:.5e})"
    
    def __repr__(self):
        s = super().__repr__()
        for key in self.p:
            s += f"\n  {key}: {zlib.adler32(self.p[key].tobytes())}"
        return s