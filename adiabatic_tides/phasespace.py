import numpy as np
from . import mathtools
from .config import Configureable, only_on_change, GeneralConfig, EddingtonConfig, ActionsConfig, SamplingConfig

class PhaseSpace():
    def __init__(self, profile):
        self.profile = profile
        self.cfg = profile.cfg
        self.anisotropy = np.nan
        
    def f_of_e(self):
        raise NotImplementedError("This is an abstract class, please implement a subclass")

    def f_of_el(self):
        raise NotImplementedError("This is an abstract class, please implement a subclass")

class EddingtonPhaseSpace(PhaseSpace):
    def __init__(self, profile, anisotropy=0.):
        super().__init__(profile)
        self.anisotropy = anisotropy

        self.q = {}
        self.ip = {}

    @only_on_change(['eddington', 'general'])
    def _setup_f(self):
        print("Recalculating phasespace")

        cfg_ps : EddingtonConfig = self.profile.cfg["eddington"]
        cfg_gen : GeneralConfig = self.profile.cfg["general"]

        ri = np.geomspace(cfg_gen.rmin/cfg_gen.scale_geometry, cfg_gen.rmax*cfg_gen.scale_geometry, int(cfg_ps.nr*cfg_gen.scale_accuracy))
        phi = self.profile.potential(ri, zero_at_zero=True)
        sel = np.roll(phi, -1) != phi # cancellation can lead to some energies being identical, let's avoid this
        
        e,f1 = mathtools.anisotropic_inversion(ri[sel], self.profile.density(ri[sel]), phi[sel], beta=self.anisotropy, nintegrate=cfg_ps.nintegrate)
        self.q["phasespace_r"] = ri[sel]
        self.q["phasespace_e"] = e
        self.q["phasespace_f"] = f1

        self.ip["f1"] = mathtools.define_interpolator(e, f1, method="pchip", bounds="zero")

    def f_of_e(self, e):
        self._setup_f()

        return self.ip["f1"](e)

    def f_of_el(self, e, l):
        self._setup_f()

        return self.ip["f1"](e) * l**(-2*self.anisotropy)
