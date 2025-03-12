import numpy as np
from . import profiles
from . import numerics
from .config import Config
from .profiles import RadialProfile, CompositeProfile
import time
from functools import partial

# === pure functions ===

def adiabatic_iteration(f_of_jl, rho, m, phi, fpa_below=None, rpmin=1e-11, nr=200, ninterp=50, nintegrate=32, G=43.0071057317063e-10, getf=False, rmax=1e10):
    def accr(r): return -G * m(r) / r**2
    def daccdr(r): return 2 * G * m(r) / r**3 - 4.*np.pi * rho(r) * G

    rperi, rapo, rlmax, rtid, ramax_of_rp = numerics.interpolate.define_paspace_boundaries(phi, accr, daccdr, rpmin=rpmin, rmax=rmax)
    table = numerics.interpolate.define_limited_peri_apo_table(ramax_of_rp, rpmin, rlmax, nbins=ninterp)
    f_of_rperi_rapo = numerics.interpolate.setup_adiabatic_f_of_rperi_rapo(f_of_jl, phi, table, fpa_below=fpa_below, accr=accr, daccdr=daccdr)
    rnew = np.geomspace(rpmin,rtid,nr)

    rhonew = numerics.integrate.integrate_f_paspace(f_of_rperi_rapo, phi, accr, rnew, N=nintegrate, rperirange=(0, rlmax), raporange=(0, ramax_of_rp))

    if getf:
        return rnew, rhonew, f_of_rperi_rapo
    else:
        return rnew, rhonew

def adiabatic_tidal_iteration(f_of_jl, rho, m, phi, tide, fpa_below=None, rpmin=1e-11, nr=200, ninterp=50, nintegrate=32, G=43.0071057317063e-10, getf=False, rmax=1e10):
    assert tide > 0, "Tide must be positive"
    
    def m_tot(r): return m(r) - tide/G * r**3
    def rho_tot(r): return rho(r) - 3.* tide / (4.*np.pi*G)
    def phi_tot(r): return phi(r) - 0.5 * tide* r**2
    
    return adiabatic_iteration(f_of_jl, rho_tot, m_tot, phi_tot, fpa_below=fpa_below, rpmin=rpmin, nr=nr, ninterp=ninterp, nintegrate=nintegrate, G=G, getf=getf, rmax=rmax)

def adiabatic_tidal_reconstruction(prof, tide, iter_max=200, eps=1e-3, rpmin=1e-20, rpmin2=None, get_all=False, verbose=1, nbins_fini=100, nintegrate=32, nr=200, ninterp=50, lower_boundary="initial", G=43.0071057317063e-10, eps_circ=1e-4, rmax=1e10):
    #Define Initial profile phase space
    # table = at.mathtools.define_peri_apo_table(rpmin, rpmax, nbins=nbins_fini)
    def pot_t(r): return prof.potential(r) - 0.5*tide*r**2
    def accr_t(r): return prof.accr(r) + tide*r
    rt0 = numerics.search.find_rphimax(pot_t)
    if verbose:
        print(f"Initial Tidal Radius {rt0:.2e}")
    rpmax = rt0*10

    table = numerics.interpolate.define_limited_peri_apo_table(ramax_of_rp=lambda r: rpmax, rpmin=rpmin, rlmax=rpmax, nbins=nbins_fini)
    # rp_ra_of_jl = numerics.interpolate.setup_rperi_rapo_of_jl(prof.potential, table, accr=prof.accr, daccdr=prof.daccdr, eps_circ=eps_circ)
    rp_ra_of_jl = numerics.interpolate.setup_rperi_rapo_of_jl_new(prof.potential, table, accr=prof.accr, daccdr=prof.daccdr, eps_circ=eps_circ)

    if rpmin2 is None:
        rpmin2 = rpmin*1e1

    def f_of_jl(j, l):
        rperi,rapo = rp_ra_of_jl(j,l)

        return prof.f_of_el(*prof.E_L_of_rperi_rapo(rperi, rapo))
    
    rho, m, phi = prof.density, prof.m_of_r, prof.potential
    if lower_boundary == "initial":
        lower_boundary = prof.density, prof.m_of_r, prof.potential
        fpa_below = prof.f_of_rperi_rapo
    else:
        fpa_below = None
    profiles = []
    for i in range(0,iter_max):
        rnew, rhonew = adiabatic_tidal_iteration(f_of_jl, rho, m, phi, tide=tide,  fpa_below=fpa_below, nr=nr, nintegrate=nintegrate, ninterp=ninterp, rpmin=rpmin2, G=G, rmax=rmax)
        rel_error = np.max(np.abs((rhonew-rho(rnew))/prof.density(rnew)))
        if verbose:
            print(f"iteration {i} relative diff {rel_error:.2%}")
        rho, m, phi = numerics.integrate.solve_poisson_via_spline_with_smart_boundaries(rnew, np.clip(rhonew, 0, None), lower_boundary=lower_boundary, upper_boundary="vacuum", G=G)
        if rel_error < eps:
            break
        if get_all:
            profiles.append((rnew, rhonew, rho, m, phi))
    
    if get_all:
        return profiles
    else:
        return rnew, rhonew, rho, m, phi
    
# === Object oriented interface ===

class AdiabaticTransformation():
    def __init__(self, prof_initial : RadialProfile, prof_pert : RadialProfile, nr=None, verbose=1):
        # Combine with the config from the initial profile
        self.cfg = prof_initial.cfg
        self.cfg.adiabatic.nr = nr or self.cfg.adiabatic.nr

        self.prof_initial = prof_initial
        self.prof_pert = prof_pert
        self.verbose = verbose

        r0 = np.logspace(np.log10(prof_initial.rmin()), np.log10(prof_initial.rmax()), self.cfg.adiabatic.nr)
        self.history = [(r0, prof_initial.density(r0), prof_initial.density, prof_initial.m_of_r, prof_initial.potential)]

    def integrate_phasespace(self, rho, m, phi, mode=None, getf=False):
        raise NotImplementedError("This method should be implemented in a subclass")
    
    def solve_poisson(self, ri, rhoi, mode=None):
        lb = self.cfg.adiabatic.lower_boundary
        if lb == "initial":
            lb = self.prof_initial.density, self.prof_initial.m_of_r, self.prof_initial.potential
            if mode is not None: # pass mode to each function
                lb = tuple(partial(lbi, mode=mode) for lbi in lb)
        
        rho, m, phi = numerics.integrate.solve_poisson_via_spline_with_smart_boundaries(
            ri, np.clip(rhoi, 0, None), lower_boundary=lb, upper_boundary="vacuum", G=self.cfg.G())
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
        nitermax = nitermax or self.cfg.adiabatic.nitermax
        eps = eps or self.cfg.adiabatic.eps_done
        
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
        return AdiabaticResultProfile((ri, rhoi, rho, m, phi), fi, config=self.cfg)
    
    def assemble_total_profile(self, iter=-1):
        """Assemble the final profile, perturbation included"""
        if isinstance(self.prof_initial, CompositeProfile):
            remnants = {}
            for label in self.prof_initial.internal:
                remnants[label] = self.assemble_single_profile(mode=label, iter=iter)
            for label in self.prof_initial.external:
                remnants[label] = self.prof_initial.profiles[label]
            return CompositeProfile(**remnants, perturbation=self.prof_pert, external=self.prof_initial.external + ("perturbation",), phase_space_mode="children", config=self.cfg)
        else:
            remnant = self.assemble_single_profile(iter=iter)
            return CompositeProfile(remnant=remnant, perturbation=self.prof_pert, external=("perturbation",), phase_space_mode="children", config=self.cfg)

class AdiabaticTidalTransformation(AdiabaticTransformation):
    def __init__(self, prof_initial : RadialProfile, tide=1., nr=None, verbose=1):
        assert tide > 0, "Tide must be positive"
        self.tide = tide
        prof_pert = profiles.RadialTidalProfile(tide=tide, config=prof_initial.cfg)
        super().__init__(prof_initial=prof_initial, prof_pert=prof_pert, nr=nr, verbose=verbose)

    @classmethod
    def from_rtid(cls, prof_initial : RadialProfile, rtid=1., **kwargs):
        return cls(prof_initial=prof_initial, tide=-prof_initial.accr(rtid)/rtid, **kwargs)

    def integrate_phasespace(self, rho, m, phi, mode=None, getf=False):
        cfg = self.cfg.adiabatic

        kwargs = dict(rpmin=self.prof_initial.rmin()*cfg.rminfac, nr=int(cfg.nr), ninterp=int(cfg.ninterp), nintegrate=int(cfg.nintegrate), rmax=self.prof_initial.rmax())
        if cfg.lower_boundary == "initial":
            fpa_below = self.prof_initial.f_of_rperi_rapo
            if mode is not None:
                fpa_below = partial(fpa_below, mode=mode)
        else:
            fpa_below = None

        f0_of_jl = self.prof_initial.f_of_jl
        if mode is not None:
            f0_of_jl = partial(f0_of_jl, mode=mode)

        return adiabatic_tidal_iteration(f0_of_jl, rho, m, phi, tide=self.tide, fpa_below=fpa_below, getf=getf, G=self.cfg.G(), **kwargs)

class AdiabaticResultProfile(RadialProfile):
    def __init__(self, result, f_of_rp_ra, config : Config | None = None):
        super().__init__(phase_space=None, anisotropy=None, config=config)
        ri, rhoi, rho, m, phi = result

        self.q = dict(ri=ri, rhoi=rhoi)
        self.ip = dict(rho=rho, m=m, phi=phi, f_of_rp_ra=f_of_rp_ra)

    def density(self, r):
        return self.ip["rho"](r)
    def m_of_r(self, r):
        return self.ip["m"](r)
    def potential(self, r, zero_at_zero=True):
        return self.ip["phi"](r)
    def f_of_rperi_rapo(self, rp, ra):
        return self.ip["f_of_rp_ra"](rp, ra)
    def f_of_e(self, e):
        raise NotImplementedError("f_of_e is not meaningful for Adiabatic Remnants")
    def f_of_el(self, E, L):
        raise NotImplementedError("f_of_el is not implemented for Adiabatic Remnants, instead use f_of_rperi_rapo")
    def f_of_jl(self, j, l):
        raise NotImplementedError("f_of_jl makes sense for Adiabatic Remnants, but we first need to define the selection function")
    
    def __str__(self):
        return f"AdiabaticResultProfile with {len(self.q['ri'])} points in ({self.q['ri'][0]:.5e}, {self.q['ri'][-1]:.5e})"