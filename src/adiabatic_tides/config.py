import yaml
from dataclasses import dataclass, asdict
from typing import Any, Dict, Callable, List
from collections.abc import Iterable

def only_on_change(attributes: Iterable[str] = (), cfg_groups: Iterable[str] = ()):
    """Decorator to execute a method only on the first call or when any specified config group changes."""
    def decorator(method: Callable):
        def wrapper(self: 'Configureable', *args, **kwargs):
            current_configs = {group: asdict(self.cfg[group]) for group in cfg_groups}
            current_var = {var: getattr(self, var) for var in attributes}
            
            if not hasattr(self, f'_{method.__name__}_last_configs'):
                # If it's the first call, execute the method and store the configs
                result = method(self, *args, **kwargs)
                setattr(self, f'_{method.__name__}_last_configs', current_configs)
                setattr(self, f'_{method.__name__}_last_vars', current_var)
                return result
            
            last_configs = getattr(self, f'_{method.__name__}_last_configs')
            last_vars = getattr(self, f'_{method.__name__}_last_vars')

            # Check if any of the groups have changed
            if (any(current_configs[group] != last_configs[group] for group in cfg_groups) 
                or any(getattr(self, var) != last_vars[var] for var in attributes)):
                # If any config has changed, execute the method and update the stored configs
                result = method(self, *args, **kwargs)
                setattr(self, f'_{method.__name__}_last_configs', current_configs)
                setattr(self, f'_{method.__name__}_last_vars', current_var)
                return result

            return None  # No changes, so no execution
        
        return wrapper
    return decorator

class Configureable:
    DEFAULT_CONFIG = {}
    
    def __init__(self, **configs):
        """Initialize the configurable with default values and provided configs."""
        self.cfg = {}
        
        for group, default_instance in self.DEFAULT_CONFIG.items():
            updated_values = {**asdict(default_instance), **configs.get(group, {})}
            self.cfg[group] = type(default_instance)(**updated_values)

    def config_to_dict(self) -> Dict[str, Any]:
        """Convert all config groups to a dictionary."""
        return {group: asdict(config) for group, config in self.cfg.items()}

    def update_config(self, group: str, **kwargs):
        """Update a single parameter within a config group."""
        if group not in self.cfg:
            raise ValueError(f"Unknown config group: {group}")

        updated_values = asdict(self.cfg[group])

        for key in kwargs:
            if key not in updated_values:
                raise ValueError(f"Unknown parameter '{key}' in group '{group}'")
            updated_values[key] = kwargs[key]

        self.cfg[group] = type(self.cfg[group])(**updated_values)  # Recreate instance

    def update_from_yaml(self, file_path: str):
        """Load configuration updates from a YAML file and apply them."""
        with open(file_path, "r") as f:
            file_config = yaml.safe_load(f) or {}
        self.update_config(**file_config)


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
    nintegrate : int = 32
    nsteps_metropolis : int = 64