from .radial_profile import RadialProfile
import numpy as np
from .. import numerics
import zlib

class NumericalProfile(RadialProfile):
    def __init__(self, ri, rho, boundary="powerlaw", anisotropy=0., **configs):
        """A radial profile of which only the density form is known
        
        ri : radius sampling points
        rho : density -- can be an array like ri or a function
        boundary : How to handle radii r < min(ri). Can be "constant" or "powerlaw"
                   For the powerlaw case a powerlaw profile is fitted based on the
                   two smallest radii. This is the recommended mode if applicable.
        anisotropy : anisotropy parameter beta
        """
        super().__init__(anisotropy=anisotropy, rmin=ri[0], rmax=ri[-1], **configs)
        
        self.boundary = boundary

        self.set_density_profile(ri, rho)

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
        self.rhoi = rhoi

        if callable(rho):
            self.ip_rho = rho

        self.potential_zero_at_infty = False

    def density(self, r):
        return self.ip_rho(r)
    
    def m_of_r(self, r):
        return self.ip_m(r)

    def potential(self, r, zero_at_zero=True):
        return self.ip_phi(r)

    def to_dict(self):
        """Returns a dictionary with all variables that describe the current state"""
        d = {}
        d["r"] = self.ri
        d["rho"] = self.rhoi
        d["boundary"] = self.boundary
        d["anisotropy"] = self.anisotropy
        return d

    @classmethod
    def from_dict(cls, d):
        """Load a state extracted from a previous '.to_dict()' call"""
        return cls.__init__(d["ri"], d["rho"], boundary=d["boundary"], anisotropy=d["anisotropy"])

    def __str__(self):
        return f"NumericalProfile(anisotropy={self.anisotropy:.5g}) with {len(self.ri)} points in ({self.ri[0]:.5g}, {self.ri[-1]:.5g})"

    def __repr__(self):
        s =  super().__repr__()
        s += "\nBoundary: " + self.boundary
        s += "\nHash:"
        s += f"\n  ri: {zlib.adler32(self.ri.data.tobytes())}"
        s += f"\n  rho: {zlib.adler32(self.rhoi.data.tobytes())}"
        return s

class ParticleProfile(NumericalProfile):
    def __init__(self, particles, rbins):
        """Numerical profile that is derived from binning a set of particles
        particles: can either be (r,m) or (r,m,vr,L) or (pos,vel,m)
            or a dictionary containing variables "r", "m" and optionally "vr" and "l"
        """
        self.rbins = rbins

        self.set_particles(particles, update=False)

        rho, mprof = numerics.sample.get_mass_profile(self.p["r"], self.p["m"], self.rbins)

        super().__init__(np.sqrt(rbins[1:]*rbins[:-1]), rho, boundary="zero", anisotropy=None)

    def _update_mass_profile(self):
        rho, mprof = numerics.sample.get_mass_profile(self.p["r"], self.p["m"], self.rbins)
        super().set_density_profile(self.ri, rho)

    def set_particles(self, particles, update=True):
        "particles -- can either be (r,m) or (r,m,vr,l) or (pos,vel,m) or dictionary"
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
        d["p"] = self.p
        d["rbins"] = self.rbins
        return d

    @classmethod
    def from_dict(cls, d):
        """Load a state extracted from a previous '.to_dict()' call"""
        return cls.__init__(particles=d["p"], rbins=d["rbins"])

    def __str__(self):
        return f"ParticleProfile with {len(self.p['r'])} particles in {len(self.rbins) - 1} bins in ({self.rbins[0]:.5g}, {self.rbins[-1]:.5g})"
    
    def __repr__(self):
        s = super().__repr__()
        for key in self.p:
            s += f"\n  {key}: {zlib.adler32(self.p[key].tobytes())}"
        return s