from .radial_profile import RadialProfile
from ..phasespace import EddingtonPhaseSpace
import numpy as np
from functools import partial
from .. import numerics
from ..numerics.search import maximize_scalar
from ..config import Config

def combine_functions(f, mode, internal, external, *args, func_combine=np.sum, **kwargs):
    if mode == "alldict":
        return {label: f[label](*args, **kwargs) for label in f}
    elif mode == "all":
        return (f[label](*args, **kwargs) for label in f)
    if mode == "total":
        return func_combine([f[label](*args, **kwargs) for label in f], axis=0)
    elif mode == "self":
        return func_combine([f[label](*args, **kwargs) for label in internal], axis=0)
    elif mode == "external":
        return func_combine([f[label](*args, **kwargs) for label in external], axis=0)
    elif mode in f:
        return f[mode](*args, **kwargs)
    else:
        valid_modes = tuple(f.keys()) + ("alldict", "all", "total", "self", "external")
        raise ValueError("Invalid mode. Valid modes are: " + ", ".join(valid_modes))


class CompositeProfile(RadialProfile):
    def __init__(self, external=(), phase_space_mode="joint_inversion", config : Config | None = None, **profiles):
        """Create a profile by combining several profiles.
        
        All functions where it makes sense (e.g. density, potential) 
        will return the sum of all profile components.
        
        external : List of keys that correspond to external profiles. Profiles
                   that are marked as external will not contribute to mode="self" 
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
 
    def _combine_profiles(self, d, func_name, mode, *args, **kwargs):
        functions = {label: getattr(d[label], func_name) for label in d}
        return combine_functions(functions, mode, self.internal, self.external, *args, **kwargs)
    
    def rmin(self, mode="total"):
        return self._combine_profiles(self.profiles, 'rmin', mode, func_combine=np.max)
    
    def rmax(self, mode="total"):
        return self._combine_profiles(self.profiles, 'rmax', mode, func_combine=np.min)
        
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

    def rmax_vmax(self, mode="total"):
        """Radius and velocity where the circular velocity is maximal"""
        opt = maximize_scalar(lambda r: self.m_of_r(r, mode=mode)/r, (self.rmin(), self.rmax()))
        return opt.x, self.vcirc(opt.x, mode=mode)
    
    def _initialize_phasespace(self):
        if self._phase_space_initialized:
            return
        
        assert self.phase_space_mode == "joint_inversion"

        for label in self.profiles:
            if not label in self.external:
                assert self.profiles[label].phase_space is not None, "Only external profiles can have undefined phase space"
                self.phase_spaces[label] = EddingtonPhaseSpace(self.profiles[label].density, self.potential, self.cfg.general, self.cfg.eddington, anisotropy=self.profiles[label].anisotropy)
       
        self._phase_space_initialized = True
    
    def f_of_e(self, E, mode="self"):
        if self.phase_space_mode == "children":
            return self._combine_profiles(self.profiles, 'f_of_e', mode, E)
        elif self.phase_space_mode == "joint_inversion":
            self._initialize_phasespace()
            return self._combine_profiles(self.phase_spaces, 'f_of_e', mode, E)
    
    def f_of_el(self, E, L, mode="self"):
        if self.phase_space_mode == "children":
            return self._combine_profiles(self.profiles, 'f_of_el', mode, E, L)
        elif self.phase_space_mode == "joint_inversion":
            self._initialize_phasespace()
            return self._combine_profiles(self.phase_spaces, 'f_of_el', mode, E, L)
    
    def f_of_rperi_rapo(self, rp, ra, mode="self"):
        if self.phase_space_mode == "children":
            return self._combine_profiles(self.profiles, 'f_of_rperi_rapo', mode, rp, ra)
        elif self.phase_space_mode == "joint_inversion":
            e,l = self.E_L_of_rperi_rapo(rp,ra)
            return self.f_of_el(e,l, mode=mode)
    
    def f_of_jl(self, j, l, mode="self"):
        rp,ra = self.action_map.rp_ra_of_jl(j,l)
        return self.f_of_rperi_rapo(rp,ra, mode=mode)
    
    def compute_pa_space_integral(self, r, f_of_rp_ra=None, vrmoment=0, vtmoment=0, vmoment=0, nintegrate=40, mode="self"):
        if f_of_rp_ra is not None: # In this case it doesn't make sense to speak of separate components
            return super().compute_pa_space_integral(r, f_of_rp_ra=f_of_rp_ra, vrmoment=vrmoment, vtmoment=vtmoment, vmoment=vmoment, nintegrate=nintegrate)
        
        # Create a dictionary that includes all phase spaces
        fs = {}
        for label in self.profiles:
            fs[label] = partial(self.compute_pa_space_integral, f_of_rp_ra=partial(self.f_of_rperi_rapo, mode=label))
        
        return combine_functions(fs, mode, self.internal, self.external, r, vrmoment=vrmoment, vtmoment=vtmoment, vmoment=vmoment, nintegrate=nintegrate)
    
    def compute_vr2_vt2(self, r, mode="self", nintegrate=40, nint2=40):
        """Returns the velocity dispersions vr2 and vt2 as a function of radius"""
        # For velocity dispersions we have to combine the phase space integral
        # and then divide by the density afterwards!
        # Therefore this function requires a separate implementation

        rho_x_vr2 = self.compute_pa_space_integral(r, vrmoment=2, nintegrate=nintegrate, mode=mode)
        rho_x_vt2 = self.compute_pa_space_integral(r, vtmoment=2, nintegrate=nintegrate, mode=mode)
        rho = self.compute_pa_space_integral(r, mode=mode, nintegrate=nintegrate)

        if isinstance(rho, dict):
            return {key: rho_x_vr2[key] / rho[key] for key in rho}, {rho_x_vt2[key] / rho[key] for key in rho}
        else:
            return rho_x_vr2 / rho, rho_x_vt2 / rho
        
    def compute_line_of_sight_vdisp2_and_dens(self, R, mode="self", nintegrate=40, ninterp=100):
        """computes the line of sight velocity dispersion and the column density at projected radius R"""
        assert not "dict" in mode, "This function does not support dict mode"

        rip = np.geomspace(self.rmin(), self.rmax(), ninterp+2)[1:-1]
        vr2, vt2 = self.compute_vr2_vt2(rip, mode=mode)
        assert np.all((vr2 >= 0) & (vt2 >= 0))
        def ip_vr2_vt2(r):
            return np.exp(np.interp(np.log(r), np.log(rip), np.log(vr2))), np.exp(np.interp(np.log(r), np.log(rip), np.log(vt2)))
        
        return numerics.integrate.integrate_line_of_sight_vdisp2_and_dens(lambda r: self.density(r, mode=mode), ip_vr2_vt2, R, nintegrate=nintegrate)
        
    def __str__(self):
        s = "CompositeProfile:"
        for k, v in self.profiles.items():
            s += f"\n  {k}: {v}"
        s += f"\n  external: ({', '.join(self.external)})"
        return s