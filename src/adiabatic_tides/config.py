import yaml
from dataclasses import dataclass, asdict
from typing import Any, Dict, Callable, List
from collections.abc import Iterable
import copy
import numpy as np

@dataclass
class GeneralConfig:
    scale_accuracy: float = 1
    scale_geometry: float = 1
    rmin: float = 1e-20
    rmax: float = 1e20

@dataclass
class EddingtonConfig:
    nintegrate: int = 100
    nr: float = 2000

@dataclass
class ActionsConfig:
    # For finding peri/apocenter radii:
    niter_pa : int = 30
    search_method : str = "ridders"
    # For Action integral:
    nintegrate: int = 40
    # For mapping rp_ra(j, l):
    nbins_rp: int = 250
    nbins_ra: int = 100 # zero means that it uses the same as rp
    nsteps_newton: int = 5
    rafac_max: float = 1e10
    rpfac_eps: float = 1e-5

@dataclass
class AdiabaticConfig:
    # Performance critical parameters
    nr : int = 200
    ninterp : int = 50 
    nintegrate : int = 32
    # how to deal with contributions from radii < rmin
    lower_boundary : str = "initial" # "initial", "powerlaw" or "constant"
    rminfac : float = 1e2 # offset minimal reconstruction radius a little to avoid numerical issues
    # When to stop iterating:
    nitermax : int = 100
    eps_done : float = 1e-3

@dataclass
class SamplingConfig:
    nintegrate: int = 40
    ninterp : int = 1001
    nsteps_metropolis : int = 64

class Config():
    def __init__(self,
                 general : GeneralConfig | None = None,
                 eddington : EddingtonConfig | None = None,
                 actions : ActionsConfig | None = None,
                 adiabatic : AdiabaticConfig | None = None,
                 sampling : SamplingConfig | None = None):
        
        self.general = general or GeneralConfig()
        self.eddington = eddington or EddingtonConfig()
        self.actions = actions or ActionsConfig()
        self.adiabatic = adiabatic or AdiabaticConfig()
        self.sampling = sampling or SamplingConfig()

        self.configs = (self.general, self.eddington, self.actions, self.adiabatic, self.sampling)

    def scale_accuracy(self, scale: float):
        self.eddington.nintegrate = int(self.eddington.nintegrate * scale)
        self.eddington.nr = int(self.eddington.nr * scale)
        
        self.actions.niter_pa = int(self.actions.niter_pa * scale)
        self.actions.nintegrate = int(self.actions.nintegrate * scale)
        self.actions.nbins_rp = int(self.actions.nbins_rp * scale)
        self.actions.nbins_ra = int(self.actions.nbins_ra * scale)
        self.actions.nsteps_newton = int(self.actions.nsteps_newton * scale)
        
        self.adiabatic.nr = int(self.adiabatic.nr * scale)
        self.adiabatic.ninterp = int(self.adiabatic.ninterp * scale)
        self.adiabatic.nintegrate = int(self.adiabatic.nintegrate * scale)
        self.adiabatic.nitermax = int(self.adiabatic.nitermax * scale)
        self.adiabatic.eps_done = self.adiabatic.eps_done / scale

        self.sampling.nintegrate = int(self.sampling.nintegrate * scale)
        self.sampling.ninterp = int(self.sampling.ninterp * scale)
        self.sampling.nsteps_metropolis = int(self.sampling.nsteps_metropolis * scale)

    def scale_geometry(self, scale: float):
        self.general.rmin = self.general.rmin / scale
        self.general.rmax = self.general.rmax * scale

        self.actions.rafac_max = self.actions.rafac_max * scale
        self.actions.rpfac_eps = self.actions.rpfac_eps / scale

        self.adiabatic.rminfac = self.adiabatic.rminfac * np.sqrt(scale)

    def modified(self):
        modified_attrs = []
        for config in self.configs:
            for field in config.__dataclass_fields__:
                if getattr(config, field) != config.__dataclass_fields__[field].default:
                    modified_attrs.append((config.__class__.__name__, field, getattr(config, field)))
        return modified_attrs
    
    def print_modified(self):
        for config in self.configs:
            print(f"{config.__class__.__name__}:")
            for field in config.__dataclass_fields__:
                if getattr(config, field) != config.__dataclass_fields__[field].default:
                    print(f"  {field}: {getattr(config, field)}")

    def __str__(self):
        s = "Config:\n  "
        s += "\n  ".join((str(c) for c in self.configs))
        return s
    
    def __repr__(self):
        return str(self)
    
def only_on_change(attributes: Iterable[str] = ()):
    "Decorator to execute a method only on the first call or when any specified attribute changes."
    def decorator(method: Callable):
        def wrapper(self, *args, **kwargs):
            current_var = {var: copy.deepcopy(getattr(self, var)) for var in attributes}
            
            if not hasattr(self, f'_{method.__name__}_last_vars'):
                # If it's the first call, execute the method and store the configs
                setattr(self, f'_{method.__name__}_last_vars', current_var)
                return method(self, *args, **kwargs)
            
            last_vars = getattr(self, f'_{method.__name__}_last_vars')

            # Check if any of the groups have changed
            if any(getattr(self, var) != last_vars[var] for var in attributes):
                setattr(self, f'_{method.__name__}_last_vars', current_var)
                return method(self, *args, **kwargs)

            return None  # No changes, so no execution
        
        return wrapper
    return decorator