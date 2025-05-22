import numpy as np
from .config import only_on_change, GeneralConfig, EddingtonConfig, ActionsConfig
from . import numerics
from scipy.interpolate import PchipInterpolator, RectBivariateSpline

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
    def __init__(self, density, potential, cfg_gen : GeneralConfig, cfg_ed : EddingtonConfig, anisotropy=0.):
        super().__init__()
        self.density = density
        self.potential = potential

        self.anisotropy = anisotropy

        self.cfg_gen = cfg_gen
        self.cfg_ed = cfg_ed

        self.q = {}
        self.ip = {}

    def set_potential(self, potential):
        self.potential = potential

    def set_density(self, density):
        self.density = density

    @only_on_change(attributes=("cfg_gen","cfg_ed"))
    def _setup_f(self):
        ri = np.geomspace(self.cfg_gen.rmin, self.cfg_gen.rmax, int(self.cfg_ed.nr))
        phi = self.potential(ri)
        
        sel = np.roll(phi, -1) != phi # cancellation can lead to some energies being identical, let's avoid this
        
        e,f1 = numerics.integrate.anisotropic_inversion(ri[sel], self.density(ri[sel]), phi[sel], beta=self.anisotropy, nintegrate=self.cfg_ed.nintegrate)
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
        self.cfg_gen = profile.cfg.general
        self.cfg_act = profile.cfg.actions

    def orbit_valid_jl(self, j, l):
        raise NotImplementedError("This is an abstract class, please implement a subclass")
    
    def orbit_valid_rp_ra(self, rp, ra):
        raise NotImplementedError("This is an abstract class, please implement a subclass")

    def rp_ra_of_jl(self, j, l):
        raise NotImplementedError("This is an abstract class, please implement a subclass")

class InterpolatorActionMap(ActionMap):
    def __init__(self, profile):
        super().__init__(profile)

        self.q = {}
        self.ip = {}

    @only_on_change(attributes=("cfg_gen","cfg_act")) 
    def setup_rp_ra_of_jl(self):
        cfg_gen : GeneralConfig = self.cfg_gen
        cfg_act : ActionsConfig = self.cfg_act

        # Define boundaries of the orbit space
        rmin, rmax = self.profile.rmin(), self.profile.rmax()
        rperi, rapo, rlmax, rtid, ramax_of_rp = numerics.interpolate.define_paspace_boundaries(self.profile.potential, self.profile.accr, self.profile.daccdr, rpmin=rmin, rmax=rmax)

        jmax = self.profile.radial_action_of_rp_ra(rperi, rapo)
        ei, li = self.profile.e_l_of_rperi_rapo(rperi, rapo)

        self.q["lmax"] = self.profile.vcirc(rlmax) * rlmax

        self.ip["ramax_of_rp"] = ramax_of_rp
        self.ip["jmax_of_l"] = lambda l: np.interp(l, li, jmax)

        table = numerics.interpolate.define_limited_peri_apo_table(ramax_of_rp=ramax_of_rp, rpmin=rmin, rlmax=rlmax, nbins=cfg_act.nbins_rp, nbins_apo=cfg_act.nbins_ra)

        self.ip["rp_ra_of_jl"] = numerics.interpolate.setup_rperi_rapo_of_jl_new(self.profile.potential, table, nsteps_newton=cfg_act.nsteps_newton, accr=self.profile.accr, daccdr=self.profile.daccdr, eps_circ=self.cfg_act.eps_circ)

    def orbit_valid_jl(self, j, l):
        self.setup_rp_ra_of_jl()

        return (l <= self.q["lmax"]) & (j <= self.ip["jmax_of_l"](l))
    
    def orbit_valid_rp_ra(self, rp, ra):
        self.setup_rp_ra_of_jl()

        return (ra >= rp) & (ra <= self.ip["ramax_of_rp"](rp))

    def rp_ra_of_jl(self, j, l):
        self.setup_rp_ra_of_jl()

        jlvalid = self.orbit_valid_jl(j,l)

        rp, ra = np.zeros_like(j), np.zeros_like(j)
        rp[jlvalid], ra[jlvalid] = self.ip["rp_ra_of_jl"](j[jlvalid], l[jlvalid])
        rp[~jlvalid], ra[~jlvalid] = np.nan, np.nan

        return rp, ra

class ActionMapThroughLLines(ActionMap):
    """A new version of the ActionMap
    
    It first determines lines rp,ra (L=const) and then inverts rp(J | L) to J(rp | L) and ra(J | L) to J(ra | L)
    This reduces the rp,ra <-> J,L inversion to a 1D problem, which can nicely be sovled through 1D interpolators

    This is faster and works much more robustly for limtied spaces (as for tidally truncated profiles)
    """
    def __init__(self, profile):
        super().__init__(profile)

        self.q = {}
        self.ip = {}

    @only_on_change(attributes=("cfg_gen","cfg_act")) 
    def setup_rp_ra_of_jl(self, nl=400, ninvertj=400, nsteps_int=4, nj=2000, j0=1e-5):
        cfg_gen : GeneralConfig = self.cfg_gen
        cfg_act : ActionsConfig = self.cfg_act

        # Define boundaries of the orbit space
        rmin, rmax = self.profile.rmin(), self.profile.rmax()
        rperi, rapo, rlmax, rtid, ramax_of_rp = numerics.interpolate.define_paspace_boundaries(self.profile.potential, self.profile.accr, self.profile.daccdr, rpmin=rmin, rmax=rmax)

        prof = self.profile

        lmaxes = prof.e_l_of_rperi_rapo(rperi, rapo)[1]
        def ra_max_of_l(l):
            return np.exp(np.interp(np.log(l), np.log(lmaxes), np.log(rapo)))

        rc = np.exp(numerics.utility.tanh_space(np.log(rmin), np.log(rlmax), nl, tmax=3))
        rp_ev, ra_ev = numerics.integrate.rp_ra_with_rlcirc(rc, ra_max_of_l(prof.vcirc(rc) * rc)*1.01, prof.accr, nsteps=ninvertj, substeps=nsteps_int)
        rp_ev[-1], ra_ev[-1] = rlmax, rlmax
        li =  prof.e_l_of_rperi_rapo(rp_ev[:,0], ra_ev[:,0])[1]

        ramax_li, rpmin_li = np.nanmax(ra_ev, axis=1), np.nanmin(rp_ev, axis=1)
        jmax_of_li = prof.radial_action_of_rp_ra(rpmin_li, ramax_li)

        self.li = li
        self.jmin_of_l = lambda l: j0 * l
        self.jmax_of_l = lambda l: np.interp(l, li, jmax_of_li)

        self.ramax_of_rp = lambda rp: np.interp(rp, rpmin_li, ramax_li)
        self.ramax_of_rp_v2 = ramax_of_rp

        jmin, jmax = self.jmin_of_l(li), self.jmax_of_l(li)
        uj = np.linspace(0., 1., nj)
        jgrid = np.sinh(uj*np.arcsinh(jmax/jmin)[:,np.newaxis])*jmin[:,np.newaxis]

        assert np.all(~np.isnan(jgrid[:-1,:]))

        # Invert  rp <-> j and ra <-> j for each l-column individually
        rpgrid, ragrid = np.zeros_like(jgrid), np.zeros_like(jgrid)
        for i in range(0, jgrid.shape[0]-1):
            ji = prof.radial_action_of_rp_ra(rp_ev[i], ra_ev[i])
            sel = (~np.isnan(rp_ev[i])) & (~np.isnan(ra_ev[i])) & (~np.isnan(ji))

            # tiny numerical errors may break monotonicity, this does not matter, but it makes the interpolator
            sel[1:] &= (ji[1:] > np.maximum.accumulate(np.nan_to_num(ji[:-1], 0)))
            
            rpgrid[i] = np.clip(PchipInterpolator(ji[sel], rp_ev[i,sel])(jgrid[i]), np.min(rp_ev[i,sel]), np.max(rp_ev[i,sel]))
            ragrid[i] = np.clip(PchipInterpolator(ji[sel], ra_ev[i,sel])(jgrid[i]), np.min(ra_ev[i,sel]), np.max(ra_ev[i,sel]))

        rpgrid[-1], ragrid[-1] = rlmax, rlmax

        assert np.all((rpgrid > 0) & (ragrid > 0))

        assert (np.sum(np.isnan(rpgrid)) == 0) and (np.sum(np.isnan(ragrid)) == 0), f"Found nans: rpgrid {np.sum(np.isnan(rpgrid))} ragrid {np.sum(np.isnan(ragrid))}"

        self.ip_rp = RectBivariateSpline(np.log(li), uj, np.log(rpgrid), kx=3, ky=3)
        self.ip_ra = RectBivariateSpline(np.log(li), uj, np.log(ragrid), kx=3, ky=3)

    def orbit_valid_jl(self, j, l):
        self.setup_rp_ra_of_jl()

        valid = (l >= np.min(self.li)) & (l <= np.max(self.li))
        valid &= (j >= self.jmin_of_l(l)) & (j <= self.jmax_of_l(l))

        return valid

    def orbit_valid_rp_ra(self, rp, ra):
        self.setup_rp_ra_of_jl()

        return (ra >= rp) & (ra <= self.ramax_of_rp(rp))

    def rp_ra_of_jl(self, j, l):
        self.setup_rp_ra_of_jl()

        # print("lmax", np.max(l), np.max(self.li))
        if np.max(l) > np.max(self.li):
            print("some ls are too large", np.max(l), np.max(self.li))

        jmin, jmax = self.jmin_of_l(l), self.jmax_of_l(l)

        # assert np.all(j < jmax), f"jmax: {np.max(jmax)}, j: {np.max(j)}"
        # assert np.all(j > jmin), f"jmin: {np.min(jmin)}, j: {np.min(j)}"

        if np.any(np.isnan(jmax)):
            print("Got jmax nans", np.sum(np.isnan(jmax)))

        u = np.arcsinh(j/jmin)/np.arcsinh(jmax/jmin)

        if (np.min(u) < 0) | (np.max(u) > 1):
            print("Warning: Got u out of bounds: umax:", np.max(u), "umin:", np.min(u))
            u[(u < 0) | (u > 1)] = np.nan

        rp = np.exp(self.ip_rp(np.log(l), u, grid=False))
        ra = np.exp(self.ip_ra(np.log(l), u, grid=False))

        # assert np.all((ra > 0) & (rp > 0))

        # print("rpnans", np.mean(np.isnan(rp)), "ranans", np.mean(np.isnan(ra)))

        return rp, ra