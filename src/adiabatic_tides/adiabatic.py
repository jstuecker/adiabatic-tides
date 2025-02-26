import numpy as np
from . import profiles
from . import mathtools
from .config import Configureable, AdiabaticConfig, GeneralConfig
from .profiles import RadialProfile
from .composite import CompositeProfile
import time
from functools import partial

class AdiabaticTransformation(Configureable):
    DEFAULT_CONFIG = {
        'adiabatic': AdiabaticConfig()
    }

    def __init__(self, prof_initial : RadialProfile, prof_pert : RadialProfile, nr=None, verbose=1, **configs):
        # Combine with the config from the initial profile
        self.DEFAULT_CONFIG = {**self.DEFAULT_CONFIG, **prof_initial.cfg}
        super().__init__(**configs)

        self.prof_initial = prof_initial
        self.prof_pert = prof_pert
        self.verbose = verbose

        if nr is not None:
            self.cfg["adiabatic"].nr = nr

        r0 = np.logspace(np.log10(prof_initial.rmin()), np.log10(prof_initial.rmax()), self.cfg["adiabatic"].nr)
        self.history = [(r0, prof_initial.density(r0), prof_initial.density, prof_initial.m_of_r, prof_initial.potential)]

    def integrate_phasespace(self, rho, m, phi, mode=None, getf=False):
        raise NotImplementedError("This method should be implemented in a subclass")
    
    def solve_poisson(self, ri, rhoi, mode=None):
        lb = self.cfg["adiabatic"].lower_boundary
        if lb == "initial":
            lb = self.prof_initial.density, self.prof_initial.m_of_r, self.prof_initial.potential
            if mode is not None: # pass mode to each function
                lb = tuple(partial(lbi, mode=mode) for lbi in lb)
        
        rho, m, phi = mathtools.solve_poisson_via_spline_with_smart_boundaries(
            ri, np.clip(rhoi, 0, None), lower_boundary=lb, upper_boundary="vacuum")
        return rho, m, phi
    
    def iterate(self):
        ri, rhoi, rho0, m0, phi0 = self.history[-1]
        rnew, rhonew = self.integrate_phasespace(rho0, m0, phi0)

        rho, m, phi = self.solve_poisson(rnew, rhonew)
        
        self.history.append((rnew, rhonew, rho, m, phi))
        return rnew, rho0, rho
        
    def run(self, nitermax=None, eps=None, reset=True):
        if reset:
            self.history = self.history[:1]
        cfg : AdiabaticConfig = self.cfg["adiabatic"]
        nitermax = nitermax or cfg.nitermax
        eps = eps or cfg.eps_done * self.cfg["general"].scale_accuracy
        
        for i in range(nitermax):
            t0 = time.time()
            ri, rho, rhonew = self.iterate()
            rel_error = np.max(np.abs((rhonew(ri)-rho(ri))/self.prof_initial.density(ri)))
            if self.verbose:
                print(f"iteration {i} relative diff {rel_error:.2%} dt {time.time()-t0:.2f}s")
            if rel_error < eps:
                break
        return self
    
    def assemble_single_profile(self, mode=None, iter=-1):
        """Assemble the final profile, perturbation not included -- may be a sub-population"""
        _, _, rhotot, mtot, phitot = self.history[iter]
        ri, rhoi, fi = self.integrate_phasespace(rhotot, mtot, phitot, mode=mode, getf=True)
        rho,m,phi = self.solve_poisson(ri, rhoi, mode=mode)
        return AdiabaticResultProfile((ri, rhoi, rho, m, phi), fi)
    
    def assemble_total_profile(self, iter=-1):
        """Assemble the final profile, perturbation included"""
        if isinstance(self.prof_initial, CompositeProfile):
            remnants = {}
            for label in self.prof_initial.internal:
                remnants[label] = self.assemble_single_profile(mode=label, iter=iter)
            for label in self.prof_initial.external:
                remnants[label] = self.prof_initial.profiles[label]
            return CompositeProfile(**remnants, perturbation=self.prof_pert, external=self.prof_initial.external + ("perturbation",))
        else:
            remnant = self.assemble_single_profile(iter=iter)
            return CompositeProfile(remnant=remnant, perturbation=self.prof_pert, external=("perturbation",))

class AdiabaticTidalTransformation(AdiabaticTransformation):
    def __init__(self, prof_initial : RadialProfile, tide=1., nr=None, verbose=1, **configs):
        assert tide > 0, "Tide must be positive"
        self.tide = tide
        prof_pert = profiles.RadialTidalProfile(alpha=tide)
        super().__init__(prof_initial=prof_initial, prof_pert=prof_pert, nr=nr, verbose=verbose, **configs)

    @classmethod
    def from_rtid(cls, prof_initial : RadialProfile, rtid=1., **kwargs):
        return cls(prof_initial=prof_initial, tide=-prof_initial.accr(rtid)/rtid, **kwargs)

    def integrate_phasespace(self, rho, m, phi, mode=None, getf=False):
        cfg : AdiabaticConfig = self.cfg["adiabatic"]
        a, b = self.cfg["general"].scale_geometry, self.cfg["general"].scale_accuracy

        kwargs = dict(rpmin=self.prof_initial.rmin()*cfg.rminfac, nr=int(cfg.nr*b), ninterp=int(cfg.ninterp*b), nintegrate=int(cfg.nintegrate*b))
        if cfg.lower_boundary == "initial":
            fpa_below = self.prof_initial.f_of_rperi_rapo
            if mode is not None:
                fpa_below = partial(fpa_below, mode=mode)
        else:
            fpa_below = None

        f0_of_jl = self.prof_initial.f_of_jl
        if mode is not None:
            f0_of_jl = partial(f0_of_jl, mode=mode)

        return mathtools.adiabatic_tidal_iteration(f0_of_jl, rho, m, phi, tide=self.tide, fpa_below=fpa_below, getf=getf, **kwargs)

class AdiabaticResultProfile(RadialProfile):
    def __init__(self, result, f_of_rp_ra):
        super().__init__(phase_space=None)
        ri, rhoi, rho, m, phi = result

        self.q = dict(ri=ri, rhoi=rhoi)

        self.ip = dict(rho=rho, m=m, phi=phi, f_of_rp_ra=f_of_rp_ra)

    def density(self, r):
        return self.ip["rho"](r)
    def m_of_r(self, r):
        return self.ip["m"](r)
    def potential(self, r):
        return self.ip["phi"](r)
    def f_of_rperi_rapo(self, rp, ra):
        return self.ip["f_of_rp_ra"](rp, ra)
    def f_of_e(self, e):
        raise NotImplementedError("f_of_e is not meaningful for Adiabatic Remnants")
    def f_of_el(self, E, L):
        raise NotImplementedError("f_of_el is not implemented for Adiabatic Remnants, instead use f_of_rperi_rapo")
    def f_of_jl(self, j, l):
        raise NotImplementedError("f_of_jl makes sense for Adiabatic Remnants, but we first need to define the selection function")