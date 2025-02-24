from .profiles import RadialProfile
import numpy as np

class CompositeProfile(RadialProfile):
    def __init__(self, *profiles, labels=None, external=False):
        """Create a profile by combining several profiles.
        
        All functions where it makes sense (e.g. density, potential) 
        will return the sum of all profile components.
        
        labels : can be a list of strings.. defaults to ("0", "1"...)
        """
        super().__init__()
        self.profiles = profiles
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