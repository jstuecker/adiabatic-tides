from .radial_profile import RadialProfile
from ..phasespace import EddingtonPhaseSpace
import numpy as np
from functools import partial
from .. import numerics
from ..numerics.search import maximize_scalar
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

    # Became unnecessary (delete soon)
    def drhodr(self, r, component="total"):
        return self._combine_profiles(self.profiles, 'drhodr', component, r)

    def m_of_r(self, r, component="total"):
        return self._combine_profiles(self.profiles, 'm_of_r', component, r)

    def potential(self, r, zero_at_zero=True, component="total"):
        return self._combine_profiles(self.profiles, 'potential', component, r, zero_at_zero=zero_at_zero)

    # Became unnecessary (delete soon)
    def daccdr(self, r, component="total"):
        return self._combine_profiles(self.profiles, 'daccdr', component, r)
    
    # Became unnecessary (delete soon)
    def vcirc(self, r, component="total"):
        return np.sqrt(np.clip(-self.accr(r, component=component) * r, 0., None))
    
    # Became unnecessary (delete soon)
    def accr(self, r, component="total"):
        """Radial Acceleration (negative means pull towards center)"""
        return  -self.G * self.m_of_r(r, component=component) / r**2

    # Became unnecessary (delete soon)
    def rmax_vmax(self, component="total"):
        """Radius and velocity where the circular velocity is maximal"""
        opt = maximize_scalar(lambda r: self.m_of_r(r, component=component)/r, (self.rmin(), self.rmax()))
        return opt.x, self.vcirc(opt.x, component=component)
    
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
            raise ValueError("This case not handled properly")
    
    def compute_pa_space_integral(self, r, f_of_rp_ra=None, vrmoment=0, vtmoment=0, vmoment=0, nintegrate=40, ramax=None, component="self"):
        if f_of_rp_ra is not None: # In this case it doesn't make sense to speak of separate components
            return super().compute_pa_space_integral(r, f_of_rp_ra=f_of_rp_ra, vrmoment=vrmoment, vtmoment=vtmoment, vmoment=vmoment, nintegrate=nintegrate, ramax=ramax)
        
        # Create a dictionary that includes all phase spaces
        fs = {}
        for label in self.profiles:
            fs[label] = partial(self.compute_pa_space_integral, f_of_rp_ra=partial(self.f_of_rperi_rapo, component=label))
        
        return combine_functions(fs, component, self.internal, self.external, r, vrmoment=vrmoment, vtmoment=vtmoment, vmoment=vmoment, nintegrate=nintegrate, ramax=ramax)
    
    def compute_vr2_vt2(self, r, component="self", nintegrate=40):
        """Returns the velocity dispersions vr2 and vt2 as a function of radius"""
        rho_x_vr2 = self.compute_pa_space_integral(r, vrmoment=2, nintegrate=nintegrate, component=component)
        rho_x_vt2 = self.compute_pa_space_integral(r, vtmoment=2, nintegrate=nintegrate, component=component)

        rho = self.compute_pa_space_integral(r, component=component, nintegrate=nintegrate)

        if isinstance(rho, dict):
            return {key: rho_x_vr2[key] / rho[key] for key in rho}, {rho_x_vt2[key] / rho[key] for key in rho}
        else:
            return rho_x_vr2 / rho, rho_x_vt2 / rho
        
    def compute_line_of_sight_vdisp2_and_dens(self, R, component="self", nintegrate=40, ninterp=100):
        """computes the line of sight velocity dispersion and the column density at projected radius R"""
        assert not "dict" in component, "This function does not support dict mode"

        rip = np.geomspace(self.rmin(), self.rmax(), ninterp+2)[1:-1]
        rho_x_vr2 = self.compute_pa_space_integral(rip, vrmoment=2, nintegrate=nintegrate, component=component)
        rho_x_vt2 = self.compute_pa_space_integral(rip, vtmoment=2, nintegrate=nintegrate, component=component)
        # We also integrate the density numerically to inherit the same discreteness error
        rho = self.compute_pa_space_integral(rip, nintegrate=nintegrate, component=component) 

        # Zero-densities can cause some errors with log-interpolation, let's remopve them and set the right boundary to zero
        rip, rho, rho_x_vr2, rho_x_vt2 = rip[rho > 0], rho[rho > 0], rho_x_vr2[rho > 0], rho_x_vt2[rho > 0]

        def ip_rho(r): return np.exp(np.interp(np.log(r), np.log(rip), np.log(rho), right=-np.inf))
        def ip_rho_x_vr2(r): return np.exp(np.interp(np.log(r), np.log(rip), np.log(rho_x_vr2), right=-np.inf))
        def ip_rho_x_vt2(r): return np.exp(np.interp(np.log(r), np.log(rip), np.log(rho_x_vt2), right=-np.inf))
        
        return numerics.integrate.integrate_line_of_sight_vdisp2_and_dens(ip_rho, ip_rho_x_vr2, ip_rho_x_vt2, R, nintegrate=nintegrate)
        
    def __str__(self):
        s = "CompositeProfile:"
        for k, v in self.profiles.items():
            s += f"\n  {k}: {v}"
        s += f"\n  external: ({', '.join(self.external)})"
        return s