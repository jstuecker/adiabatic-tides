import numpy as np
from . import profiles
from . import mathtools
from .config import Configureable, AdiabaticConfig, GeneralConfig
from .profiles import RadialProfile
import time

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

    def integrate_phasespace(self, rho, m, phi): # Abstract has to be implemented in child class
        raise NotImplementedError
    
    def iterate(self):
        cfg : AdiabaticConfig = self.cfg["adiabatic"]

        ri, rhoi, rho0, m0, phi0 = self.history[-1]
        rnew, rhonew = self.integrate_phasespace(rho0, m0, phi0)

        lb = cfg.lower_boundary
        if lb == "initial":
            lb = self.prof_initial.density, self.prof_initial.m_of_r, self.prof_initial.potential
        
        rho, m, phi = mathtools.solve_poisson_via_spline_with_smart_boundaries(
            rnew, np.clip(rhonew, 0, None), lower_boundary=lb, upper_boundary="vacuum")
        
        self.history.append((rnew, rhonew, rho, m, phi))
        return rnew, rho0, rho
        
    def run(self, nitermax=None, eps=None):
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

class AdiabaticTidalTransformation(AdiabaticTransformation):
    def __init__(self, prof_initial : RadialProfile, tide=1., nr=None, verbose=1, **configs):
        assert tide > 0, "Tide must be positive"
        self.tide = tide
        prof_pert = profiles.RadialTidalProfile(alpha=tide)
        super().__init__(prof_initial=prof_initial, prof_pert=prof_pert, nr=nr, verbose=verbose, **configs)

    @classmethod
    def from_rtid(cls, prof_initial : RadialProfile, rtid=1., **kwargs):
        return cls(prof_initial=prof_initial, tide=-prof_initial.accr(rtid)/rtid, **kwargs)

    def integrate_phasespace(self, rho, m, phi):
        cfg : AdiabaticConfig = self.cfg["adiabatic"]
        a, b = self.cfg["general"].scale_geometry, self.cfg["general"].scale_accuracy

        kwargs = dict(rpmin=self.prof_initial.rmin()*cfg.rminfac, nr=int(cfg.nr*b), ninterp=int(cfg.ninterp*b), nintegrate=int(cfg.nintegrate*b))
        if cfg.lower_boundary == "initial":
            fpa_below = self.prof_initial.f_of_rperi_rapo
        else:
            fpa_below = None

        ri, rho = mathtools.adiabatic_tidal_iteration(
            self.prof_initial.f_of_jl, rho, m, phi, tide=self.tide, fpa_below=fpa_below, **kwargs)
        return ri, rho