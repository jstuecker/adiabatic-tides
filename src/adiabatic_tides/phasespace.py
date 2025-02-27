import numpy as np
from .config import Configureable, only_on_change, GeneralConfig, EddingtonConfig, ActionsConfig, SamplingConfig
from . import numerics

class PhaseSpace():
    def __init__(self, anisotropy=np.nan):
        self.anisotropy = anisotropy

    def f_of_e(self):
        raise NotImplementedError("This is an abstract class, please implement a subclass")

    def f_of_el(self):
        raise NotImplementedError("This is an abstract class, please implement a subclass")

class AnalyticPhaseSpace(PhaseSpace):
    def __init__(self, f_of_e=None, f_of_el=None, anisotropy=np.nan):
        super().__init__(anisotropy=anisotropy)

        assert f_of_e is not None or f_of_el is not None, "At least one of f_of_e or f_of_el must be defined"

        if f_of_e is None:
            def f_of_e(e): 
                return f_of_el(e, 1.) 
        if f_of_el is None:
            assert self.anisotropy == 0., "Please define f_of_el if anisotropy is not zero"
            def f_of_el(e, l): 
                return f_of_e(e)
        self.f_of_e = f_of_e
        self.f_of_el = f_of_el

class EddingtonPhaseSpace(PhaseSpace):
    def __init__(self, density, potential, cfg, anisotropy=0.):
        super().__init__()
        self.density = density
        self.potential = potential

        self.anisotropy = anisotropy
        self.cfg = cfg

        self.q = {}
        self.ip = {}

    def set_potential(self, potential):
        self.potential = potential

    def set_density(self, density):
        self.density = density

    @only_on_change(attributes=("potential","density"), cfg_groups=('eddington', 'general'))
    def _setup_f(self):
        cfg_ps : EddingtonConfig = self.cfg["eddington"]
        cfg_gen : GeneralConfig = self.cfg["general"]

        ri = np.geomspace(cfg_gen.rmin/cfg_gen.scale_geometry, cfg_gen.rmax*cfg_gen.scale_geometry, int(cfg_ps.nr*cfg_gen.scale_accuracy))
        phi = self.potential(ri)
        
        sel = np.roll(phi, -1) != phi # cancellation can lead to some energies being identical, let's avoid this
        
        e,f1 = numerics.integrate.anisotropic_inversion(ri[sel], self.density(ri[sel]), phi[sel], beta=self.anisotropy, nintegrate=cfg_ps.nintegrate)
        self.q["phasespace_r"] = ri[sel]
        self.q["phasespace_e"] = e
        self.q["phasespace_f"] = f1

        self.ip["f1"] = numerics.interpolate.define_interpolator(e, f1, method="pchip", bounds="zero")

    def f_of_e(self, e):
        self._setup_f()

        return self.ip["f1"](e)

    def f_of_el(self, e, l):
        self._setup_f()

        return self.ip["f1"](e) * l**(-2*self.anisotropy)

class ActionMap():
    def __init__(self, profile):
        self.profile = profile
        self.cfg = profile.cfg

    def rp_ra_of_jl(self, j, l):
        raise NotImplementedError("This is an abstract class, please implement a subclass")

class InterpolatorActionMap(ActionMap):
    def __init__(self, profile):
        super().__init__(profile)

        self.q = {}
        self.ip = {}

    @only_on_change(cfg_groups=('actions', 'general'))
    def setup_rp_ra_of_jl(self):
        cfg_gen : GeneralConfig = self.cfg["general"]
        cfg_act : ActionsConfig = self.cfg["actions"]

        rpmin, rpmax, facmax = self.profile.rmin(), self.profile.rmax(), cfg_act.rafac_max*cfg_gen.scale_geometry

        # table = mathtools.define_limited_peri_apo_table(ramax_of_rp=lambda r: rpmax, rpmin=rpmin, rlmax=rpmax, nbins=cfg_act.nbins_rp, nbins_apo=cfg_act.nbins_ra)
        table = numerics.interpolate.define_peri_apo_table(rpmin, rpmax, nbins=cfg_act.nbins_rp, nbins_apo=cfg_act.nbins_ra, facmax=facmax)
        self.ip["rp_ra_of_jl"] = numerics.interpolate.setup_rperi_rapo_of_jl(self.profile.potential, table, nsteps_newton=cfg_act.nsteps_newton, nintegrate_action=cfg_act.nintegrate)

    def rp_ra_of_jl(self, j, l):
        self.setup_rp_ra_of_jl()

        return self.ip["rp_ra_of_jl"](j, l)