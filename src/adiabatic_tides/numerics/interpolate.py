import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator, RectBivariateSpline, NearestNDInterpolator, LinearNDInterpolator, CloughTocher2DInterpolator, RBFInterpolator, griddata

from . import search
from . import utility
from .integrate import calculate_radial_action_tanh_peri_apo, calculate_dj_de_tanh_peri_apo, calculate_jel_and_dj_dl_drp_dra

# ==================== Generic Interpolation Functions ===================== #

def define_interpolator(x, y, method="pchip", bounds="constant", **kwargs):
    if method == "pchip":
        ip = PchipInterpolator(x, y, extrapolate=False, **kwargs)
    elif method == "linear":
        ip = lambda xev : np.interp(xev, x, y, left=np.nan, right=np.nan)
    else:
        raise ValueError("Method not recognized")
    
    if bounds == "nan":
        return ip
    elif bounds == "constant":
        return lambda xev: ip(np.clip(xev, x[0], x[-1]))
    elif bounds == "zero":
        return lambda xev: np.nan_to_num(ip(xev), 0)
    else:
        raise ValueError("bounds has to be nan, constant or zero")

def flexible_interpolator(xi, yi, logx=False, logy=False, eps_for_logx=0., eps_for_logy=0., kind="cubic", bounds_error=False, fill_value=None):
    """Returns an interpolator function y(x) that flexibly can handle log-interpolation
    
    xi : x evaluation points for creating the interpolator
    yi : y evaluation points for creating the interpolator
    logx : whether to interpolate in logarithmic x-space
    logy : whether to interpolate in logarithmic y-space
    eps_for_logx : will interpolate in log(x + eps). Setting this is
           useful to exactly represent x=0
    eps_for_logy : same for x
    kind : interpolation method. Can be same as scipy.interpolate.interp1d
    bounds_error : whether to throw errors when out of bounds
    fill_value : Two values that should be return for x out of bounds. Defaults
           to (y[0], y[-1])
           
    returns : A function y(x). Note that this function always uses x and y
              in linear space. The previous arguments just modify the internal
              behavior of the interpolator (to improve accuracy).
    """
    xi, iduq = np.unique(xi, return_index=True)
    yi = yi[iduq]
    
    def xmod(x):
        if logx:
            assert np.min(x) >= 0.
            return np.log10(x+eps_for_logx)
        else:
            return x
    def ymod(y):
        if logy:
            assert np.min(y) >= 0.
            return np.log10(y+eps_for_logy)
        else:
            return y
    def yinvmod(y):
        if logy:
            return 10.**(y) - eps_for_logy
        else:
            return y

    if fill_value is None:
        fill_value = tuple(ymod(np.array((yi[0], yi[-1]))))
        if(xi[-1] < xi[0]):
            fill_value = (fill_value[1], fill_value[0])
    else:
        fill_value = tuple(ymod(fill_value))

    if logy:
        assert np.min(yi) >= 0.
    if logx:
        assert np.min(xi) >= 0.
    ip = interp1d(xmod(xi), ymod(yi), kind=kind, bounds_error=bounds_error, fill_value=fill_value)

    def f(x):
        return yinvmod(ip(xmod(x)))
    
    return f

def vectorized_interp(x, xi, yi):
    """Like a np.interp that broadcasts along first axis for x,xi and yi"""
    i1 = np.argmax(xi > x[:,np.newaxis], axis=-1)
    ar = np.arange(0, len(xi))

    i0 = np.clip(i1-1, 0, xi.shape[1]-1)
    i1 = np.clip(i1, 0, xi.shape[1]-1)

    x0, y0 = xi[(ar,i0)], yi[(ar,i0)]
    x1, y1 = xi[(ar,i1)], yi[(ar,i1)]

    eps = 1e-30
    dx = np.clip(x1 - x0, eps, None)

    return y0 + (y1 - y0) * (x - x0) / dx

# ===================== Generic Space Transformations ====================== #

def bins_log_lin_log(xlogmin, xlinmin, xlinmax, xlogmax, n1=50, n2=50, n3=50, dlogmin=-7, dlogmin_up=None, include_xminxmax=False):
    """Concatenates a logspace with a linspace and an inverted logspace. 
    
    (1) logspace, starting at xlogmin+eps1, ending at xlinmin
    (2) linspace, starting at xlinmin, ending at xlinmax
    (3) logspace, starting at xlinmax, ending at xlogmax-eps2
    
    n1 : number of bins in first logspace
    n2 : number of bins in linspace
    n3 : number of bins in second logspace
    
    dlogmin : eps1 = (xlinmin-xlogmin) * 10**(dlogmin)
    dlogmin_up : eps2 = (xlogmax-xlinmax) * 10**(dlogmin_up), defaults to dlogmin
    
    include_xminxmax : if True, slightly modifies the behavior so that xlogmin
                       and xlogmax are included in the results

    returns : an array with values in the specified ranges
    """
    if dlogmin_up is None:
        dlogmin_up = dlogmin
    if include_xminxmax:
        xi1 = xlogmin+(np.logspace(dlogmin,0., n1, endpoint=True)-10.**dlogmin)*(xlinmin - xlogmin)
        xi2 = np.linspace(xlinmin,xlinmax, n2, endpoint=True)
        xi3 = xlogmax-(np.logspace(0., dlogmin_up, n2)-10.**dlogmin_up)*(xlogmax-xlinmax)
    else:
        xi1 = xlogmin+np.logspace(dlogmin,0., n1, endpoint=False)*(xlinmin - xlogmin)
        xi2 = np.linspace(xlinmin,xlinmax, n2, endpoint=False)
        xi3 = xlogmax-np.logspace(0., dlogmin_up, n2)*(xlogmax-xlinmax)
    return np.concatenate([xi1, xi2, xi3])

def bins_log_both_ends(xmin, xmax, n1=50, n2=50, dlogmin=-7):
    """Concatenates a logspace and an inverted logspace
    
    useful when dealing with functions with two singularities
    first space starts at xmin + eps and second space ends at  xmax - eps
    where eps = (xmax-xmin)*10**dlogmin
    
    xmin : start of interval
    xmax : end of interval
    n1 : number of points in the first logspace
    n2 : number of points in the second logspace
    
    returns : an array with values in the specified range
    """
    dx = 0.5*(xmax-xmin)
    xi1 = xmin+np.logspace(dlogmin,0, n1, endpoint=False)*dx
    xi2 = xmax-np.logspace(0., dlogmin, n2)*dx
    return np.concatenate([xi1, xi2])

def logspace_map(xmin, xmax):
    def x_of_u(u):
        return xmin * (xmax/xmin)**u
    def u_of_x(x):
        return np.log(x/xmin) / np.log(xmax/xmin)
    return x_of_u, u_of_x

def logspace_map_offset(xmin, xmax, xoff):
    def x_of_u(u):
        return (xmin+xoff) * ((xmax+xoff)/(xmin+xoff))**u - xoff
    def u_of_x(x):
        return np.log((x+xoff)/(xmin+xoff)) / np.log((xmax+xoff)/(xmin+xoff))
    return x_of_u, u_of_x

def coshspace_map(fmax, pow=1.):
    def x_of_u(u):
        return np.cosh(u**pow*np.arccosh(fmax))
    def u_of_x(x):
        return (np.arccosh(x)/np.arccosh(fmax))**(1./pow)
    return x_of_u, u_of_x

# ========= Functions for defining special interpolation spaces ============ #

def map_peri_apo_space_log_log(rpmin, rpmax, facmax=None, rpoff=0., facmin=1e-4):
    if facmax is None:
        facmax = rpmax/rpmin

    def rpra_of_uv(u,v):
        rp = (rpmin+rpoff) * ((rpmax+rpoff)/(rpmin+rpoff))**u - rpoff
        fac = 1. + facmin*np.exp(v*np.log(facmax/facmin))
        return rp, rp*fac
    def uv_of_rpra(rp, ra):
        u = np.log((rp+rpoff)/(rpmin+rpoff)) / np.log((rpmax+rpoff)/(rpmin+rpoff))
        x = np.clip(ra/rp - 1., facmin, None)
        v = np.log(x/facmin) / np.log(facmax/facmin)
        return u,v
    return rpra_of_uv, uv_of_rpra

def map_limited_peri_apo_space_log_tanh(ramax_of_rp, rpmin, rpmax, rpoff=0., tmax=5):
    def rpra_of_uv(u,v):
        rp = (rpmin+rpoff) * ((rpmax+rpoff)/(rpmin+rpoff))**u - rpoff
        
        logramax = np.log(ramax_of_rp(rp))
        t = (v - 0.5) * 2 * tmax
        logra = 0.5*(logramax+np.log(rp)) + 0.5*(logramax-np.log(rp)) * np.tanh(t)
        
        return rp, np.exp(logra)
    
    def uv_of_rpra(rp, ra):
        u = np.log((rp+rpoff)/(rpmin+rpoff)) / np.log((rpmax+rpoff)/(rpmin+rpoff))
        
        logramax = np.log(ramax_of_rp(rp))
        with np.errstate(divide='ignore', invalid='ignore'):
            # Nans happen when out of range here, that is ok, those cases should be nan and caught elsewhere
            t =  np.arctanh((np.log(ra) - 0.5*(logramax+np.log(rp))) / (0.5*(logramax-np.log(rp))))
        v = 0.5*(t/tmax + 1.)

        return u,v
    
    return rpra_of_uv, uv_of_rpra

def define_peri_apo_table(rpmin, rpmax, nbins=200, facmax=None, nbins_apo=None, rpoff=0., facmin=1e-3):
    if (nbins_apo is None) or (nbins_apo == 0):
        nbins_apo = nbins

    # Set up a uniform domain
    u = np.linspace(0, 1, nbins)
    v = np.linspace(0, 1, nbins_apo)
    uvgrid = np.stack(np.meshgrid(u, v, indexing="ij"), axis=-1)

    # Set up functions that map between peri/apo centers and the uniform domain
    rpra_of_uv,uv_of_rpra = map_peri_apo_space_log_log(rpmin, rpmax, facmax, rpoff=rpoff, facmin=facmin)
    rpgrid, ragrid = rpra_of_uv(uvgrid[...,0], uvgrid[...,1])

    return u,v,uvgrid,rpgrid,ragrid,rpra_of_uv,uv_of_rpra

def define_limited_peri_apo_table(ramax_of_rp, rpmin, rlmax, nbins=200, nbins_apo=None, rpoff=0., tmax=5):
    """like define_peri_apo_table, but for profiles where valid apo centers are limited"""
    if (nbins_apo is None) or (nbins_apo == 0):
        nbins_apo = nbins

    # Set up a uniform domain
    u = np.linspace(0, 1, nbins)
    v = np.linspace(0, 1, nbins_apo)
    uvgrid = np.stack(np.meshgrid(u, v, indexing="ij"), axis=-1)

    # Set up functions that map between peri/apo centers and the uniform domain
    rpra_of_uv,uv_of_rpra = map_limited_peri_apo_space_log_tanh(ramax_of_rp, rpmin, rlmax*(1-np.exp(-np.cbrt(nbins_apo))), rpoff=rpoff, tmax=tmax) # 1+np.cbrt(nbins_apo)
    rpgrid, ragrid = rpra_of_uv(uvgrid[...,0], uvgrid[...,1])

    return u,v,uvgrid,rpgrid,ragrid,rpra_of_uv,uv_of_rpra

def define_paspace_boundaries(pot, accr, daccdr, rpmin=1e-10, nbins=1000, eps=1e-6, rmax=1e10):
    rlmax, rtid = search.find_rlmax(accr, rmin=rpmin, rmax=rmax), search.find_rphimax(pot, rmin=rpmin, rmax=rmax)
    if np.isfinite(rlmax) and np.isfinite(rtid):
        rperi = np.geomspace(rpmin, rlmax, nbins)
        rapo = np.append(search.find_rapo_max_of_rperi(pot, accr, rperi[:-1], rlmax, rtid), rlmax)
        
        def ramax_of_rp(rp):
            return np.interp(rp, rperi, rapo)

        return rperi, rapo, rlmax, rtid, ramax_of_rp
    else: # profile is not actually limited
        rlmax, rtid = rmax, rmax
        rperi = np.geomspace(rpmin, rmax, nbins)
        rapo = np.ones_like(rperi)*rmax

        return rperi, rapo, rlmax, rtid, lambda rp: rmax

# =========== Functions for calculating interpolation tables =============== #

def rp_ra_of_j_l_near_circ(accr, daccdr, j,l, r0, niter=30):
    """Find the peri- and apo-centers of a nearly circular orbit with actions j and l
    
    consider the function integrals.vr_integral_near_circ to see what we need to invert
    """
    # First, find the circular orbit radius of l
    # dvr2_dr =  2*accr(r) + 2*l**2/r**3
    # dvr2_dr = 0 <=>  r0 = (l**2/accr(r0))**(1/3)
    for i in range(niter):
        r0 = (-l**2/accr(r0))**(1/3)

    c = -daccdr(r0) + 3*l**2/r0**4

    # invert j = (1/8.)*c**0.5 * (ra - rp)**2
    drap = np.sqrt(j * 8 / c**0.5)

    return r0 - drap*0.5, r0 + drap*0.5

def setup_rperi_rapo_of_jl(pot, table, nsteps_newton=5, nintegrate_action=40, k=3, accr=None, daccdr=None, eps_circ=1e-3):
    """ sets up a function that returns the peri- and apo-centric radii for a given action and angular momentum """
    u,v,uvgrid,rpgrid,ragrid,rpra_of_uv,uv_of_rpra = table

    j = calculate_radial_action_tanh_peri_apo(pot, rpgrid, ragrid, nintegrate=nintegrate_action, accr=accr, daccdr=daccdr, eps_circ=eps_circ)
    e,l = utility.e_l_of_rp_ra(pot, rpgrid, ragrid, accr=accr, eps_circ=eps_circ)

    l0, j0, facl = np.min(l[l>0]), np.min(j[j>0]), 1e-5

    xy_nn = NearestNDInterpolator(np.stack((np.log(j+j0+l*facl),np.log(l+l0)), axis=-1).reshape(-1,2), uvgrid.reshape(-1,2))
    
    logj_spline = RectBivariateSpline(u, v, np.log(j+j0+l*facl), kx=k, ky=k)
    logl_spline = RectBivariateSpline(u, v, np.log(l+l0), kx=k, ky=k)

    def rpra_of_jl(j, l):
        # Use NN interpolator for first guess
        ftarget, gtarget = np.log(j+l*facl+j0), np.log(l+l0)
        xy0 = xy_nn(np.stack((ftarget, gtarget), axis=-1))

        if nsteps_newton == 0:
            return rpra_of_uv(xy0[...,0], xy0[...,1])

        def F(xy):
            return np.stack((logj_spline.ev(xy[...,0], xy[...,1]) - ftarget, logl_spline.ev(xy[...,0], xy[...,1]) - gtarget), axis=-1)
        def Jac(xy):
            res = np.array([[logj_spline.ev(xy[...,0], xy[...,1],dx=1), logj_spline.ev(xy[...,0], xy[...,1],dy=1)], 
                            [logl_spline.ev(xy[...,0], xy[...,1],dx=1), logl_spline.ev(xy[...,0], xy[...,1],dy=1)]])
            return np.einsum("ij...->...ij", res) # convenient transpose
        
        xynew = search.newton_raphson(F, Jac, xy0, niter=nsteps_newton)

        failed = np.linalg.norm(F(xynew), axis=-1) > np.linalg.norm(F(xy0), axis=-1)
        failed |= (xynew[...,0] < np.min(u)) | (xynew[...,0] > np.max(u)) | (xynew[...,1] < np.min(v)) | (xynew[...,1] > np.max(v))
        if np.sum(failed) > 0:
            # print("Warning, Newton Raphson failed for %d/%d points" % (np.sum(failed), failed.size))
            xynew[failed] = xy0[failed]

        return rpra_of_uv(xynew[...,0], xynew[...,1])
    
    return rpra_of_jl

def setup_rperi_rapo_of_jl_new(pot, table, nintegrate_action=40, nsteps_newton=2, accr=None, daccdr=None, eps_circ=1e-3):
    """ sets up a function that returns the peri- and apo-centric radii for a given action and angular momentum """
    u,v,uvgrid,rpgrid,ragrid,rpra_of_uv,uv_of_rpra = table

    if accr is not None: 
        # Set up lcirc interpolator for handling near-circular orbits
        # Note that this is only used as a guess in the search and doesn't have to be very accurate
        ri = np.unique(rpgrid)
        lc = np.sqrt(np.clip(-accr(ri)*ri**3, 0, None))

        def rcirc_of_lcirc(l):
            return np.interp(l, lc, ri)

    j = calculate_radial_action_tanh_peri_apo(pot, rpgrid, ragrid, nintegrate=nintegrate_action, accr=accr, daccdr=daccdr, eps_circ=eps_circ)
    e,l = utility.e_l_of_rp_ra(pot, rpgrid, ragrid, accr=accr, eps_circ=eps_circ)

    rpmin, ramax = np.min(rpgrid), np.max(ragrid)
    lmax, jmax = np.max(l), np.max(j)

    l0, j0, facl = np.min(l[l>0]), np.min(j[j>0]), 0

    sel = (j > 0) & (l > 0)
    xy_ip = CloughTocher2DInterpolator(np.stack((np.log(j+j0+l*facl),np.log(l+l0)), axis=-1)[sel], uvgrid[sel])

    def rpra_of_jl(j, l):
        # Use NN interpolator for first guess
        ftarget, gtarget = np.log(j+l*facl+j0), np.log(l+l0)
        xynew = xy_ip(np.stack((ftarget, gtarget), axis=-1))
        rp, ra = rpra_of_uv(xynew[...,0], xynew[...,1])

        # For almost circular orbits we use a more accurate method that avoids cancellation
        if (accr is not None) and (daccdr is not None):
            sel =  (j <= eps_circ*l) & (l < lmax) & (j < jmax) & (l > 0)
            rp[sel], ra[sel] = rp_ra_of_j_l_near_circ(accr, daccdr, j[sel], l[sel], r0=rcirc_of_lcirc(l[sel]))

            # Optionally improve the result with Newton-Raphson
            # This makes the result practically independent of the table discretization
            # assuming that the starting point is close enough to the true solution
            # Even a single step dramatically improves accuracy in that case
            # accr is additionally needed as an input to evaluate the gradient
            if nsteps_newton > 0:
                # Avoid circular orbits, they lead to cancellation
                sel = (ra > rp*(1. + eps_circ)) & (j > 0) & (l > 0) & (rp > 0) & (ra > 0)
                rp[sel], ra[sel] = newton_improve_rp_ra_of_j_l(pot,accr,j[sel],l[sel], rp[sel], ra[sel], nsteps_newton=nsteps_newton, nintegrate_action=nintegrate_action, daccdr=daccdr, eps_circ=eps_circ)

        invalid = (rp < rpmin) | (ra > ramax) | (rp > ra)
        rp[invalid], ra[invalid] = np.nan, np.nan

        return rp, ra

    return rpra_of_jl

def newton_improve_rp_ra_of_j_l(pot,accr,j0,l0, rp0, ra0, nsteps_newton=0, nintegrate_action=40, daccdr=None, eps_circ=1e-3):
    def F_and_Jac(log_rpra):
        rp, ra = np.exp(log_rpra[...,0]), np.exp(log_rpra[...,1])
        rp, ra = np.minimum(rp, ra), np.maximum(rp, ra)
        j,e,l,dj_drp, dj_dra, dl_drp, dl_dra = calculate_jel_and_dj_dl_drp_dra(pot,accr, rp, ra, nintegrate=nintegrate_action, daccdr=daccdr, eps_circ=eps_circ)

        F = np.stack((np.log(j/j0), np.log(l/l0)), axis=-1)
        Jac = np.stack((dj_drp*(rp/j), dj_dra*(ra/j), dl_drp*(rp/l), dl_dra*(ra/l)), axis=-1).reshape(j0.shape+(2,2))

        return F, Jac

    if nsteps_newton == 0:
        return rp0, ra0
    else:
        log_rpra = search.newton_raphson_FJ(F_and_Jac, np.stack((np.log(rp0), np.log(ra0)), axis=-1), niter=nsteps_newton)
        rp,ra = np.exp(log_rpra[...,0]), np.exp(log_rpra[...,1])
        rp, ra = np.minimum(rp, ra), np.maximum(rp, ra)
        return rp, ra

def setup_adiabatic_f_of_rperi_rapo(f_of_jl, pot, table, nintegrate_action=40, fpa_below=None, k=3, accr=None, daccdr=None, eps_circ=1e-3):
    ui,vi,uvgrid,rpgrid,ragrid,rpra_of_uv,uv_of_rpra = table

    j = calculate_radial_action_tanh_peri_apo(pot, rpgrid, ragrid, nintegrate=nintegrate_action, accr=accr, daccdr=daccdr, eps_circ=eps_circ)
    e,l = utility.e_l_of_rp_ra(pot, rpgrid, ragrid, accr=accr, eps_circ=eps_circ)
    
    
    f = f_of_jl(j,l)

    f0 = np.min(f[f>0])

    ip = RectBivariateSpline(ui, vi, np.log(f+f0), kx=k, ky=k)

    umin, umax, vmin, vmax = np.min(ui), np.max(ui), np.min(vi), np.max(vi)

    def f_of_rperi_rapo(rp, ra):
        u,v = uv_of_rpra(rp, ra)

        # assert np.all(~np.isnan(v))

        # For nearly circular orbits f(j,l) only depends on l and therefore it is fine to 
        # approximate by the closest resolved orbit
        v = np.clip(v, vmin, None) 

        res = np.exp(ip.ev(u,v)) - f0

        valid = (u >= umin) & (u <= umax) & (v >= vmin) & (v <= vmax)
        res[~valid] = 0

        if fpa_below is not None: 
            # The contribution of orbits with pericenters below rpmin may be relevant
            # it is possible to define a distribution function that we assume for those
            shape = np.broadcast(rp,ra).shape
            res[u < 0] = fpa_below(np.broadcast_to(rp, shape)[u < 0], np.broadcast_to(ra, shape)[u < 0])

        # assert np.all(res > 0)

        return res
    
    return f_of_rperi_rapo
