from .profiles import RadialProfile
from .phasespace import EddingtonPhaseSpace
from .config import only_on_change
import numpy as np

class CompositeProfile(RadialProfile):
    def __init__(self, external=(), configs={}, phase_space_mode="joint_inversion", **profiles):
        """Create a profile by combining several profiles.
        
        All functions where it makes sense (e.g. density, potential) 
        will return the sum of all profile components.
        
        labels : Can be a list of strings.. defaults to ("0", "1"...)
        external : List of keys that correspond to external profiles. Profiles
                   that are marked as external will not contribute to mode="self" 
                   and will have an undefined phase space
        """
        super().__init__(phase_space=None, **configs)
        assert not set(profiles.keys()) & set(("alldict", "all", "total", "self", "external")), "Trying to use prohibited profile name"

        self.profiles = {}
        self.phase_spaces = {}
        self.external = ()

        assert phase_space_mode == "joint_inversion", "Only joint inversion is supported currently"

        self.add_profiles(external=external, **profiles)

    def add_profiles(self, external=(), **profiles):
        self.profiles.update(profiles)
        self.external += tuple(external)

        # Phase space may invalidate when changing the potential... reset it
        self.phase_space_valid = False

        self.internal = tuple(label for label in self.profiles if label not in self.external)
 
    def _combine_profiles(self, d, func_name, mode, *args, **kwargs):
        if mode == "alldict":
            return {label: getattr(d[label], func_name)(*args, **kwargs) for label in d}
        elif mode == "all":
            return (getattr(d[label], func_name)(*args, **kwargs) for label in d)
        if mode == "total":
            return np.sum([getattr(d[label], func_name)(*args, **kwargs) for label in d], axis=0)
        elif mode == "self":
            return np.sum([getattr(d[label], func_name)(*args, **kwargs) for label in self.internal], axis=0)
        elif mode == "external":
            return np.sum([getattr(d[label], func_name)(*args, **kwargs) for label in self.external], axis=0)
        elif mode in d:
            return getattr(d[mode], func_name)(*args, **kwargs)
        else:
            valid_modes = tuple(d.keys()) + ("alldict", "all", "total", "self", "external")
            raise ValueError("Invalid mode. Valid modes are: " + ", ".join(valid_modes))
        
    def density(self, r, mode="total"):
        return self._combine_profiles(self.profiles, 'density', mode, r)

    def drhodr(self, r, mode="total"):
        return self._combine_profiles(self.profiles, 'drhodr', mode, r)

    def m_of_r(self, r, mode="total"):
        return self._combine_profiles(self.profiles, 'm_of_r', mode, r)

    def potential(self, r, zero_at_zero=True, mode="total"):
        return self._combine_profiles(self.profiles, 'potential', mode, r, zero_at_zero=zero_at_zero)

    def daccdr(self, r, mode="total"):
        return self._combine_profiles(self.profiles, 'daccdr', mode, r)
    
    def vcirc(self, r, mode="total"):
        return np.sqrt(np.clip(-self.accr(r, mode=mode) * r, 0., None))
    
    def accr(self, r, mode="total"):
        """Radial Acceleration (negative means pull towards center)"""
        return  -self.G * self.m_of_r(r, mode=mode) / r**2
    
    def _initialize_phasespace(self):
        if self.phase_space_valid:
            return

        for label in self.profiles:
            if not label in self.external:
                assert self.profiles[label].phase_space is not None, "Only external profiles can have undefined phase space"
                self.phase_spaces[label] = EddingtonPhaseSpace(self.profiles[label].density, self.potential, self.cfg, anisotropy=self.profiles[label].anisotropy)
       
        self.phase_space_valid = True
    
    def f_of_e(self, E, mode="self"):
        self._initialize_phasespace()
        return self._combine_profiles(self.phase_spaces, 'f_of_e', mode, E)
    
    def f_of_el(self, E, L, mode="self"):
        self._initialize_phasespace()
        return self._combine_profiles(self.phase_spaces, 'f_of_el', mode, E, L)
    
    def f_of_rperi_rapo(self, rp, ra, mode="self"):
        e,l = self.E_L_of_rperi_rapo(rp,ra)
        return self.f_of_el(e,l, mode=mode)
    
    def f_of_jl(self, j, l, mode="self"):
        rp,ra = self.action_map.rp_ra_of_jl(j,l)
        return self.f_of_rperi_rapo(rp,ra, mode=mode)
