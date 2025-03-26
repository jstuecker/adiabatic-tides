import numpy as np
from .utility import save_divide
from scipy.optimize import minimize_scalar

# ================== Generic Root Finding functions ======================== #

def vectorized_binary_search(f, xlow, xhigh, niter=100, mode="sqrt", return_err=False, exceptions=True, xfallback=None, **kwargs):
    """A vectorized binary search which searches the zero-point of f
    
    f : function with signature f(x, **kwargs)
    xlow : any location left of the zero-point
    xhigh : any location right of the zero-point f(xlow) * f(xhigh) < 0
    niter : number of iterations, typically ~30 is already enough
    mode : can be 'mean' or 'sqrt', decides how xlow and xhigh are combined
    return_err : if True, an estimate of the error is append to the output
    exceptions : can be True, "warning" or "silent". Controls behavior if
                 if cases without signflip are encountered
    xfallback : return these values for cases without signflip (array like)
    **kwargs : other keyword arguments will be passed through to the function
    
    returns : the location x of zero-crossing (and an error estimate if return_err)
    """
    flow, fhigh = f(xlow, **kwargs), f(xhigh, **kwargs)
    
    if np.max(np.sign(flow) * np.sign(fhigh)) > 0.: # Ttest whether we got valid limits
        msg = ("Not all cases have a sign flip between xlow and xhigh (%d / %d don't)"
               % (np.sum(np.sign(flow) * np.sign(fhigh) > 0.), np.size(xlow)))
        if exceptions == True:
            raise ValueError(msg)
        elif exceptions == "warning":
            print("Warning: ", msg)
        else:
            pass
            
        if xfallback is not None:
            sel = flow * fhigh > 0.
            xlow[sel] = xhigh[sel] = xfallback[sel]
    
    def choose(x1, x2, choosex1): # a selection function which avoids if/else switches
        return choosex1 * x1 + (~choosex1) * x2
    
    for i in range(niter):
        if mode == "mean":
            xnew = 0.5*(xlow + xhigh)
        elif mode == "sqrt":
            xnew = np.sqrt(xlow * xhigh)
        else:
            raise ValueError("Invalid mode=%s, can only be 'mean' or 'sqrt'" % mode)
        
        fnew = f(xnew, **kwargs)
        
        change_low = fnew * fhigh < 0

        xlow = choose(xnew, xlow, change_low)
        flow = choose(fnew, flow, change_low)
        xhigh = choose(xnew, xhigh, ~change_low)
        fhigh = choose(fnew, fhigh, ~change_low)
    
    xnew = choose(xlow, xhigh, flow >= 0.)
    
    if return_err:
        return xnew, xhigh - xlow
    else:
        return xnew

def ridders_method(f, xlow, xup, niter=10, mode="both", logspace=False, invalid_val=None, **kwargs):
    """Finds the root f(x) = 0 using Ridder's method.
    mode : can be "both", "positive" or "negative"
    """
    x0, x2 = xlow, xup

    if logspace:
        x0, x2 = np.log(x0), np.log(x2)
        fin = f
        def f(x, **kwargs): return fin(np.exp(x), **kwargs)

    f0 = f(x0, **kwargs)
    f2 = f(x2, **kwargs)

    if invalid_val is None: # Raise an error for cases without zero-points in the intevral
        assert np.all(np.sign(f0*f2) <= 0)
        assert np.all(np.abs(x0) >= 1e-12 * np.abs(x2)), "Initial interval too large, expecting cancellation..."
    else: # Continue only with the valid cases
        valid = np.sign(f0*f2) <= 0.
        x0, x2 = np.where(valid, x0, invalid_val), np.where(valid, x2, invalid_val)

    for i in range(0, niter):
        x1 = (x0 + x2)/2
        f1 = f(x1, **kwargs)

        sqr = np.sqrt(f1**2 - f0*f2)
        x3 = x1 + (x1 - x0) * np.sign(f0) * np.divide(f1, sqr, out=np.ones_like(f1)*0.5, where=sqr > 0.)
        f3 = f(x3, **kwargs)

        keep1 = np.sign(f1*f3) < 0
        keep0 = (~keep1) & (np.sign(f0*f3) <= 0)
        keep2 = (~keep1) & (~keep0)

        x0 = x0*keep0 + x1*keep1 + x2*keep2
        f0 = f0*keep0 + f1*keep1 + f2*keep2

        x2 = x3
        f2 = f3

    if logspace:
        x0, x2 = np.exp(x0), np.exp(x2)

    if invalid_val is None:
        assert np.all(f2*f0 <= 0)
    else:
        x0, x2 = np.where(valid, x0, invalid_val), np.where(valid, x2, invalid_val)    

    if mode == "both":
        return x0, x2
    elif mode == "positive":
        return np.where(f2 >= 0, x2, x0)
    elif mode == "negative":
        return np.where(f2 <= 0, x2, x0)
    else:
        raise ValueError("Unknown mode")
    
def newton_raphson(F, Jac, x0, niter=10):
    if niter == 0:
        return x0

    x = x0
    for i in range(niter):
        myJ = Jac(x)
        Jinv = np.linalg.inv(myJ)
        dx = -np.einsum("...ij,...j", Jinv, F(x))
        x = x + dx
    return x

def newton_raphson_FJ(f_jac, x0, niter=10):
    if niter == 0:
        return x0

    x = x0
    for i in range(niter):
        f,jac = f_jac(x)
        jac_inv = np.linalg.inv(jac)
        dx = -np.einsum("...ij,...j", jac_inv, f)
        x = x + dx
    return x

def find_single_root(acc, r0=1., maxiter=100, eps=1e-8, warning=True, mode="negative"):
    """Assume acc is a function that is < 0 at small radii and > 0 at large radii"""
    r = r0

    # Find a radius where the sign of the acceleration is positive and negative
    if acc(r) == 0.:
        return r
    elif acc(r) > 0.:
        rpos = r
        for i in range(0, maxiter):
            r = r / 2.
            if acc(r) < 0.:
                rneg = r
                break
            if i == maxiter-1:
                raise ValueError("I couldn't find any radius where the profile is attractive")
    else:
        rneg = r
        for i in range(0, maxiter):
            r = r * 2.
            if acc(r) > 0.:
                rpos = r
                break
            if i == maxiter-1:
                if warning:
                    print("Warning, I couldn't find any radius where the profile is repulsive, rtid=infty")
                
                return np.infty

    for i in range(0, maxiter):
        r = 0.5*(rpos + rneg)
        if acc(r) > 0.:
            rpos = r
        else:
            rneg = r
            
        if (rpos-rneg)/r < eps:
            break
    
    if mode == "negative":
        return rneg
    elif mode == "positive":
        return rpos
    elif mode =="both":
        return rneg, rpos
    else:
        raise ValueError("mode has to be either negative, positive or both")

# =============== Functions for finding special roots ====================== #

def find_rlmax(accr,  rmin=1e-10, rmax=1e10, boundary_eps=0.5):
    return maximize_scalar(lambda r: -accr(r)*r**3, (rmin, rmax), boundary_eps=boundary_eps).x

def find_rphimax(pot, r0=1., rmin=1e-10, rmax=1e10, boundary_eps=0.5):
    #return find_single_root(accr, r0, eps=1e-10, mode="negative", warning=False)
    return maximize_scalar(pot, (rmin, rmax), boundary_eps=boundary_eps).x

def rperi_rapo_valid(pot, accr, rperi, rapo):
    """To have a valid apo-center we need to fulfill two conditions
    (1) pot(rapo) >= pot(rperi)
    (2) acc(rapo) + L**2/rapo**3 <= 0
    """
    rperi, rapo = np.broadcast_arrays(rperi, rapo)

    # Check for circular obits, these can be a problem...
    # We slightly perturb ther apo-center to avoid having to deal
    # with the exact solution, that requires a higher derivative
    circ = rperi == rapo
    if np.sum(circ) > 0:
        rapo = np.copy(rapo)
        rapo[circ] = rperi[circ] * (1. + 1e-8)

    phip, phia = pot(rperi), pot(rapo)

    valid = (rapo > rperi) & (phia >= phip)
    if np.sum(valid) > 0:
        L2 = 2. * (phia[valid] - phip[valid]) / (rperi[valid]**-2 - rapo[valid]**-2)
        valid[valid] = valid[valid] & (accr(rapo[valid]) + L2/rapo[valid]**3 <= 0)

    return valid

def rperi_rapo_valid_continuous(pot, accr, rperi, rapo):
    """Like rperi_rapo_valid but returns a float that is >= 0 if valid and < 0 if not"""
    phip, phia = pot(rperi), pot(rapo)

    f1 = phia - phip
    # L2 = np.clip(2. * save_divide(phia - phip, rperi**-2 - rapo**-2), 0, None)
    L2 = np.clip(2. * save_divide((phia - phip)*rperi**2*rapo**2, rapo**2 - rperi**2), 0, None)
    f2 = -(accr(rapo) + L2/rapo**3)

    f = f1*f2 * (0.5 - 1.0*((f1 < 0)&(f2<0)))
    return f #f2 * (0.5 - 1.0*((f2 > 0)&(f1<0)))

def find_rapo_max_of_rperi(pot, accr, rperi, rlmax, rtid):
    def valid(rapo):
        return rperi_rapo_valid_continuous(pot, accr, rperi, rapo)
    
    return ridders_method(valid, np.sqrt(rperi*rlmax), rtid*1.1, mode="positive", niter=10, invalid_val=np.nan)

def profile_is_disrupted(accr, rpmin=1e-10):
    return (accr(rpmin) > 0)

def profile_is_limited(accr, rpmin=1e-10):
    rtid = find_single_root(accr, rpmin, warning=False)
    return (rtid < np.infty)

def maximize_scalar(f, bounds, boundary_eps=1e-1):
    opt = minimize_scalar(lambda x: -f(x), bounds=bounds)
    opt.fun = -opt.fun

    if not opt.success:
        pass
    elif opt.x < bounds[0] +  np.abs(bounds[0]) * boundary_eps:
        opt.x, opt.fun, opt.succes = np.nan, np.nan, False
        opt.message = "Lower boundary reached during optimization"
    elif opt.x > bounds[1] -  np.abs(bounds[1]) * boundary_eps:
        opt.x, opt.fun, opt.succes = np.infty, np.nan, False
        opt.message = "Upper boundary reached during optimization"
    
    return opt