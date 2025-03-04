from dataclasses import dataclass
from typing import Any, Dict, Callable
from collections.abc import Iterable
import copy
import numpy as np

@dataclass
class GeneralConfig:
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
                 sampling : SamplingConfig | None = None,
                 scale_accuracy: float | None = None,
                 scale_geometry: float | None = None):

        self.general = general or GeneralConfig()
        self.eddington = eddington or EddingtonConfig()
        self.actions = actions or ActionsConfig()
        self.adiabatic = adiabatic or AdiabaticConfig()
        self.sampling = sampling or SamplingConfig()

        self.configs = (self.general, self.eddington, self.actions, self.adiabatic, self.sampling)

        if scale_accuracy is not None:
            self.scale_accuracy(scale_accuracy)
        if scale_geometry is not None:
            self.scale_geometry(scale_geometry)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        """Initialize the Config object from a dictionary"""
        return cls(
            general=GeneralConfig(**d.get("general", {})),
            eddington=EddingtonConfig(**d.get("eddington", {})),
            actions=ActionsConfig(**d.get("actions", {})),
            adiabatic=AdiabaticConfig(**d.get("adiabatic", {})),
            sampling=SamplingConfig(**d.get("sampling", {})),
            scale_accuracy=d.get("scale_accuracy", None),
            scale_geometry=d.get("scale_geometry", None)
        )
    
    @classmethod
    def from_yaml(cls, filename: str):
        """Initialize the Config object from a YAML file"""
        import yaml, re

        # The following is needed because of the slightly incorrect way
        # that floats are handled in the python yaml package
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))

        with open(filename, 'r') as ymlfile:
            cfg_dict = yaml.load(ymlfile, Loader=loader)

        return cls.from_dict(cfg_dict)

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

        self.actions.rafac_max = self.actions.rafac_max * np.sqrt(scale)
        self.actions.rpfac_eps = self.actions.rpfac_eps / np.sqrt(scale)

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