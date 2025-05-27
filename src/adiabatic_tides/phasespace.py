import numpy as np
from .config import only_on_change, GeneralConfig, EddingtonConfig, ActionsConfig
from . import numerics
from scipy.interpolate import PchipInterpolator, RectBivariateSpline
from scipy.integrate import cumulative_trapezoid

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
        ri, phi = ri[sel], phi[sel]

        e,f1 = numerics.integrate.anisotropic_inversion(ri, self.density(ri), phi, beta=self.anisotropy, nintegrate=self.cfg_ed.nintegrate)

        if np.any(f1 < 0):
            neg = f1 < 0.
            print(f"Warning, the Eddington inversion returned a distribution function with {np.sum(neg):d}/{len(neg):d} negative values")
            print(f"-- min/max negative energy ({np.min(e[neg]):.3e}, {np.max(e[neg]):.3e})")
            print(f"-- min/max negative radii ({np.min(ri[neg]):.3e}, {np.max(ri[neg]):.3e})")
            print("-- This may happen due to round-off errors for cored profiles at small radii")
            print("-- Please consider giving the profile a small slope -- e.g. r**-0.01")
            print("-- I will discard the negative values, but no warranties!")

            ri, e, f1 = ri[~neg], e[~neg], f1[~neg]


        # assert np.all(self.q["phasespace_f"] >= 0)

        self.q["phasespace_r"] = ri
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
        pot = lambda r: self.profile.potential(r, component="total")
        accr = lambda r: self.profile.accr(r, component="total")
        daccdr = lambda r: self.profile.daccdr(r, component="total")
        rperi, rapo, rlmax, rtid, ramax_of_rp = numerics.interpolate.define_paspace_boundaries(pot, accr, daccdr, rpmin=rmin, rmax=rmax)

        jmax = self.profile.radial_action_of_rp_ra(rperi, rapo)
        ei, li = self.profile.e_l_of_rperi_rapo(rperi, rapo)

        self.q["lmax"] = self.profile.vcirc(rlmax, component="total") * rlmax

        self.ip["ramax_of_rp"] = ramax_of_rp
        self.ip["jmax_of_l"] = lambda l: np.interp(l, li, jmax)

        table = numerics.interpolate.define_limited_peri_apo_table(ramax_of_rp=ramax_of_rp, rpmin=rmin, rlmax=rlmax, nbins=cfg_act.nbins_l, nbins_apo=cfg_act.nbins_pa)

        self.ip["rp_ra_of_jl"] = numerics.interpolate.setup_rperi_rapo_of_jl_new(pot, table, nsteps_newton=cfg_act.nsteps_newton, accr=accr, daccdr=daccdr, eps_circ=self.cfg_act.eps_circ)

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
    def setup_interpolators(self):
        cfg_gen : GeneralConfig = self.cfg_gen
        cfg_act : ActionsConfig = self.cfg_act
        
        nbins_l, nbins_pa, substeps_pa = cfg_act.nbins_l, cfg_act.nbins_pa, cfg_act.substes_pa
        nj, j0fac = cfg_act.nj, cfg_act.j0fac

        # Define boundaries of the orbit space
        # (These may be complicated for tidally truncated profiles)
        rmin, rmax = self.profile.rmin(), self.profile.rmax()
        pot = lambda r: self.profile.potential(r, component="total")
        accr = lambda r: self.profile.accr(r, component="total")
        daccdr = lambda r: self.profile.daccdr(r, component="total")
        rperi, rapo, rlmax, rtid, ramax_of_rp = numerics.interpolate.define_paspace_boundaries(pot, accr, daccdr, rpmin=rmin, rmax=rmax)

        prof = self.profile

        lmaxes = prof.e_l_of_rperi_rapo(rperi, rapo)[1]
        def ra_max_of_l(l):
            return np.exp(np.interp(np.log(l), np.log(lmaxes), np.log(rapo)))

        # Find peri and apo-center lines of constant angular momentum
        rc = np.exp(numerics.utility.tanh_space(np.log(rmin), np.log(rlmax), nbins_l, tmax=3))
        rp_ev, ra_ev = numerics.integrate.rp_ra_with_rlcirc(rc, ra_max_of_l(prof.vcirc(rc, component="total") * rc)*1.02, accr, nsteps=nbins_pa, substeps=substeps_pa, eps_circ=self.cfg_act.eps_circ)
        rp_ev[-1], ra_ev[-1] = rlmax, rlmax
        ecirc, li =  prof.e_l_of_rperi_rapo(rp_ev[:,0], ra_ev[:,0])

        ramax_li, rpmin_li = np.nanmax(ra_ev, axis=1), np.nanmin(rp_ev, axis=1)
        jmax_of_li = prof.radial_action_of_rp_ra(rpmin_li, ramax_li)
        emax_of_li = prof.e_l_of_rperi_rapo(rpmin_li, ramax_li)[0]

        self.li = li
        self.jmin_of_l = lambda l: j0fac * l
        self.jmax_of_l = lambda l: np.interp(l, li, jmax_of_li)
        self.emin_of_l = lambda l: np.interp(l, li, ecirc)
        self.emax_of_l = lambda l: np.interp(l, li, emax_of_li)

        self.ramax_of_rp = lambda rp: np.interp(rp, rpmin_li, ramax_li)
        self.ramax_of_rp_v2 = ramax_of_rp

        jmin, jmax = self.jmin_of_l(li), self.jmax_of_l(li)
        u = np.linspace(0., 1., nj)
        jgrid = np.sinh(u*np.arcsinh(jmax/jmin)[:,np.newaxis])*jmin[:,np.newaxis]
        emin, emax = self.emin_of_l(li)[:,np.newaxis], self.emax_of_l(li)[:,np.newaxis]
        egrid = np.exp(np.log(emin) + u*np.log(emax/emin))

        assert np.all(~np.isnan(jgrid[:-1,:]))

        # Invert  rp <-> j and ra <-> j for each l-column individually
        rpgridj, ragridj = np.zeros_like(jgrid), np.zeros_like(jgrid)
        jgride = np.zeros_like(jgrid)
        for i in range(0, jgrid.shape[0]-1):
            ji = prof.radial_action_of_rp_ra(rp_ev[i], ra_ev[i])
            ei = prof.e_l_of_rperi_rapo(rp_ev[i], ra_ev[i])[0]
            sel = (~np.isnan(rp_ev[i])) & (~np.isnan(ra_ev[i])) & (~np.isnan(ji))
            sel = sel & numerics.utility.monotoneous_mask(np.nan_to_num(ji, 0), mode=">")

            rpgridj[i] = np.clip(PchipInterpolator(ji[sel], rp_ev[i,sel])(jgrid[i]), np.min(rp_ev[i,sel]), np.max(rp_ev[i,sel]))
            ragridj[i] = np.clip(PchipInterpolator(ji[sel], ra_ev[i,sel])(jgrid[i]), np.min(ra_ev[i,sel]), np.max(ra_ev[i,sel]))

            sel &= numerics.utility.monotoneous_mask(ei, mode=">")
            jgride[i] = np.clip(PchipInterpolator(ei[sel], ji[sel])(egrid[i]), np.min(ji), np.max(ji))

        rpgridj[-1], ragridj[-1], jgride[-1] = rlmax, rlmax, 0.

        assert np.all((rpgridj > 0) & (ragridj > 0))

        assert (np.sum(np.isnan(rpgridj)) == 0) and (np.sum(np.isnan(ragridj)) == 0), f"Found nans: rpgrid {np.sum(np.isnan(rpgridj))} ragrid {np.sum(np.isnan(ragridj))}"

        self.ip_rp = RectBivariateSpline(np.log(li), u, np.log(rpgridj), kx=3, ky=3)
        self.ip_ra = RectBivariateSpline(np.log(li), u, np.log(ragridj), kx=3, ky=3)
        self.ip_j_of_e = RectBivariateSpline(np.log(li), u, np.log(jgride+1e-10), kx=3, ky=3)

        self.jgrid, self.rpgrid, self.ragrid = jgrid, rpgridj, ragridj

    def orbit_valid_jl(self, j, l):
        self.setup_interpolators()

        valid = (l >= np.min(self.li)) & (l <= np.max(self.li))
        valid &= (j >= self.jmin_of_l(l)) & (j <= self.jmax_of_l(l))

        return valid

    def orbit_valid_rp_ra(self, rp, ra):
        self.setup_interpolators()

        return (ra >= rp) & (ra <= self.ramax_of_rp(rp))

    def rp_ra_of_jl(self, j, l):
        self.setup_interpolators()

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
    
    def j_of_e_l(self, e, l):
        self.setup_interpolators()

        emin, emax = self.emin_of_l(l), self.emax_of_l(l)

        u = np.log(e/emin) / np.log(emax/emin)

        if (np.min(u) < 0) | (np.max(u) > 1):
            print("Warning: Got u out of bounds: umax:", np.max(u), "umin:", np.min(u))
            u[(u < 0) | (u > 1)] = np.nan

        return np.clip(np.exp(self.ip_j_of_e(np.log(l), u, grid=False)) - 1e-10, 0, None)
    
    def sample_jl(self, nsamp=1000, get_rp_ra=False, f=None, nf=1000):
        self.setup_interpolators()
        
        if f is None:
            f = self.profile.f

        # A grid used for getting j given the cumulative distribution function at fixed l
        ftarget = np.linspace(0, 1, nf)

        l, fl, j_of_fc_grid, rp_of_fc_grid, ra_of_fc_grid = [],[],[],[],[]

        for i in range(0, len(self.li)):
            # Integrate f(j,l) over j
            fjl = f(j=self.jgrid[i], l=self.li[i], rp=self.rpgrid[i], ra=self.ragrid[i])
            assert np.all(fjl >= 0)
            if np.all(fjl == 0) or np.std(self.jgrid[i]) == 0:
                continue  #Zero mass at this angular momentum (May happen e.g. at lmax or when some orbits are set to zero through f)

            fcum_of_j_givenl = (2.*np.pi)**3*cumulative_trapezoid(fjl, x=self.jgrid[i], initial=0)

            assert fcum_of_j_givenl[-1] > 0

            # Make the arrray strictly monotoneous -- otherwise the interpolator will complain
            # (Physically this should be the case, but numerically some equal cases may happen)
            sel = numerics.utility.monotoneous_mask(fcum_of_j_givenl/fcum_of_j_givenl[-1], mode=">")

            j_of_fc_grid.append(PchipInterpolator(fcum_of_j_givenl[sel]/fcum_of_j_givenl[-1], self.jgrid[i,sel])(ftarget))
            if get_rp_ra:
                rp_of_fc_grid.append(PchipInterpolator(fcum_of_j_givenl[sel]/fcum_of_j_givenl[-1], self.rpgrid[i,sel])(ftarget))
                ra_of_fc_grid.append(PchipInterpolator(fcum_of_j_givenl[sel]/fcum_of_j_givenl[-1], self.ragrid[i,sel])(ftarget))

            l.append(self.li[i])
            fl.append(fcum_of_j_givenl[-1])

        l, fl, j_of_fc_grid = np.array(l), np.array(fl), np.array(j_of_fc_grid)

        # Now do the l integral. The 2*l is from the lz integral, another l from the log integral
        fcum_l = cumulative_trapezoid(2.*l**2*fl, x=np.log(l), axis=0, initial=0)

        assert np.all(~np.isnan(fcum_l))

        # Sample and interpolate
        usamp, vsamp = np.random.uniform(0, 1, size=(2, nsamp))
        
        msamp = fcum_l[-1] / nsamp * np.ones(nsamp)

        mask = numerics.utility.monotoneous_mask(fcum_l, mode=">")
        lsamp = PchipInterpolator(fcum_l[mask]/fcum_l[-1], l[mask])(usamp)
        jsamp = RectBivariateSpline(np.log(l), ftarget, j_of_fc_grid, kx=1, ky=1)(np.log(lsamp), vsamp, grid=False)

        assert np.all(jsamp >= 0)
        assert np.all(lsamp >= 0)

        if get_rp_ra:
            rp_samp = RectBivariateSpline(np.log(l), ftarget, rp_of_fc_grid, kx=1, ky=1)(np.log(lsamp), vsamp, grid=False)
            ra_samp = RectBivariateSpline(np.log(l), ftarget, ra_of_fc_grid, kx=1, ky=1)(np.log(lsamp), vsamp, grid=False)

            return msamp, jsamp, lsamp, rp_samp, ra_samp
        

        return msamp, jsamp, lsamp