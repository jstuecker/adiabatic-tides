from .profiles import RadialProfile
import numpy as np

class CompositeProfile(RadialProfile):
    def __init__(self, external=(), configs={}, **profiles):
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

        self.profiles = profiles
        self.external = external
        
    def _combine_profiles(self, func_name, mode, *args, **kwargs):
        if mode == "alldict":
            return {label: getattr(self.profiles[label], func_name)(*args, **kwargs) for label in self.profiles}
        elif mode == "all":
            return (getattr(self.profiles[label], func_name)(*args, **kwargs) for label in self.profiles)
        if mode == "total":
            return np.sum([getattr(self.profiles[label], func_name)(*args, **kwargs) for label in self.profiles], axis=0)
        elif mode == "self":
            return np.sum([getattr(self.profiles[label], func_name)(*args, **kwargs) for label in self.profiles if label not in self.external], axis=0)
        elif mode == "external":
            return np.sum([getattr(self.profiles[label], func_name)(*args, **kwargs) for label in self.profiles if label in self.external], axis=0)
        elif mode in self.profiles:
            return getattr(self.profiles[mode], func_name)(*args, **kwargs)
        else:
            raise ValueError("Invalid mode")

    def density(self, r, mode="total"):
        return self._combine_profiles('density', mode, r)

    def drhodr(self, r, mode="total"):
        return self._combine_profiles('drhodr', mode, r)

    def m_of_r(self, r, mode="total"):
        return self._combine_profiles('m_of_r', mode, r)

    def potential(self, r, zero_at_zero=True, mode="total"):
        return self._combine_profiles('potential', mode, r, zero_at_zero=zero_at_zero)

    def daccdr(self, r, mode="total"):
        return self._combine_profiles(r, 'daccdr', mode)