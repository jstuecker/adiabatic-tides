from .radial_profile import RadialProfile
from ..phasespace import EddingtonPhaseSpace
import numpy as np
from functools import partial
from .. import numerics
from ..numerics.search import maximize_scalar_logspace
from ..config import Config

def combine_functions(f, component, internal, external, *args, func_combine=np.sum, **kwargs):
    if component == "alldict":
        return {label: f[label](*args, **kwargs) for label in f}
    elif component == "all":
        return (f[label](*args, **kwargs) for label in f)
    if component == "total":
        return func_combine([f[label](*args, **kwargs) for label in f], axis=0)
    elif component == "self":
        return func_combine([f[label](*args, **kwargs) for label in internal], axis=0)
    elif component == "external":
        return func_combine([f[label](*args, **kwargs) for label in external], axis=0)
    elif component in f:
        return f[component](*args, **kwargs)
    else:
        valid_components = tuple(f.keys()) + ("alldict", "all", "total", "self", "external")
        raise ValueError("Invalid component. Valid component are: " + ", ".join(valid_components))


class CompositeProfile(RadialProfile):
    def __init__(self, external=(), phase_space_mode="joint_inversion", config : Config | None = None, **profiles):
        """Create a profile by combining several profiles.
        
        All functions where it makes sense (e.g. density, potential) 
        will return the sum of all profile components. You can name components anyway you like

        e.g. cprof = CompositeProfile(dm=profile1, star=profile2) 
        or           CompositeProfile(mysecondcomponent=profile2, star=profile3, tide=tidal_prifle, external=("tide",))
        and then you can use it e.g. as cprof.density(r, component="star) or cprof.density(r, component="self")
        
        external : List of keys that correspond to external profiles. Profiles
                   that are marked as external will not contribute to component="self" 
                   and will have an undefined phase space
        phase_space_mode : How to combine phase spaces. Can be either "joint_inversion" or "children"
                    children: use pre-defined phase spaces of the profiles. Note that this may lead to
                             inconsistencies if the child phase spaces did not consider the full potential
                    joint_inversion: infer each profile's phasespace in the joint potential
        """
        super().__init__(phase_space=None, config=config)
        assert not set(profiles.keys()) & set(("alldict", "all", "total", "self", "external")), "Trying to use prohibited profile name"

        self.profiles = {}
        self.phase_spaces = {}
        self.external = ()
        self.phase_space_mode = phase_space_mode

        self.add_profiles(external=external, **profiles)

        self.cfg.general.rmin = self.rmin()
        self.cfg.general.rmax = self.rmax()

    def add_profiles(self, external=(), **profiles):
        self.profiles.update(profiles)
        self.external += tuple(external)

        # Phase space may invalidate when changing the potential... reset it
        self._phase_space_initialized = False

        self.internal = tuple(label for label in self.profiles if label not in self.external)
 
    def _combine_profiles(self, d, func_name, component, *args, **kwargs):
        functions = {label: getattr(d[label], func_name) for label in d}
        return combine_functions(functions, component, self.internal, self.external, *args, **kwargs)
    
    def rmin(self, component="total"):
        return self._combine_profiles(self.profiles, 'rmin', component, func_combine=np.max)
    
    def rmax(self, component="total"):
        return self._combine_profiles(self.profiles, 'rmax', component, func_combine=np.min)
        
    def density(self, r, component="total"):
        return self._combine_profiles(self.profiles, 'density', component, r)

    def m_of_r(self, r, component="total"):
        return self._combine_profiles(self.profiles, 'm_of_r', component, r)

    def potential(self, r, zero_at_zero=True, component="total"):
        return self._combine_profiles(self.profiles, 'potential', component, r, zero_at_zero=zero_at_zero)
    
    def _initialize_phasespace(self):
        if self._phase_space_initialized:
            return
        
        assert self.phase_space_mode == "joint_inversion"

        for label in self.profiles:
            if not label in self.external:
                assert self.profiles[label].phase_space is not None, "Only external profiles can have undefined phase space"
                self.phase_spaces[label] = EddingtonPhaseSpace(self.profiles[label].density, self.potential, self.cfg.general, self.cfg.eddington, anisotropy=self.profiles[label].anisotropy)
       
        self._phase_space_initialized = True
    
    def f_of_e(self, e, component="self"):
        if self.phase_space_mode == "children":
            return self._combine_profiles(self.profiles, 'f_of_e', component, e)
        elif self.phase_space_mode == "joint_inversion":
            self._initialize_phasespace()
            return self._combine_profiles(self.phase_spaces, 'f_of_e', component, e)
    
    def f_of_el(self, e, l, r=None, component="self"):
        if self.phase_space_mode == "children":
            return self._combine_profiles(self.profiles, 'f_of_el', component, e, l, r=r)
        elif self.phase_space_mode == "joint_inversion":
            self._initialize_phasespace()
            return self._combine_profiles(self.phase_spaces, 'f_of_el', component, e, l)
    
    def f_of_rperi_rapo(self, rp, ra, component="self"):
        if self.phase_space_mode == "children":
            return self._combine_profiles(self.profiles, 'f_of_rperi_rapo', component, rp, ra)
        elif self.phase_space_mode == "joint_inversion":
            e,l = self.e_l_of_rperi_rapo(rp,ra)
            return self.f_of_el(e,l, component=component)
    
    def f_of_jl(self, j, l, component="self"):
        if self.phase_space_mode == "children":
            # Since children cannot determine what is a valid orbit, we need to check it
            # on the level of the composite profile
            jlvalid = self.action_map.orbit_valid_jl(j,l)
            return self._combine_profiles(self.profiles, 'f_of_jl', component, j, l) * jlvalid
        else:
            rp,ra = self.action_map.rp_ra_of_jl(j,l)
            return self.f_of_rperi_rapo(rp,ra, component=component)
    
    def f(self, e=None, l=None, j=None, r=None, rp=None, ra=None, component="self"):
        if self.phase_space_mode == "children":
            return self._combine_profiles(self.profiles, 'f', component, e=e, l=l, j=j, r=r, rp=rp, ra=ra)
        else:
            super().f(e=e, l=l, j=j, r=r, rp=rp, ra=ra, component=component)

    def __str__(self):
        s = "CompositeProfile:"
        for k, v in self.profiles.items():
            s += f"\n  {k}: {v}"
        s += f"\n  external: ({', '.join(self.external)})"
        return s