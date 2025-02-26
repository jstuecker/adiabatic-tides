import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline, NearestNDInterpolator
from scipy.integrate import simps, trapezoid
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.special import gamma

# ====================== Generic helper functions ========================== #

def save_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=(b!=0)) # &(a!=np.infty)&(~np.isnan(a))&(~np.isnan(b)

def cosh_space(fmax, n, pow=1.):
    x = np.linspace(0., np.arccosh(fmax), n)
    return np.cosh(x[-1] * (x/x[-1])**pow)

# ===================== Differentiation functions ========================== #

def second_deriv(f, x):
    """Second order second derivative"""
    h1 = x[1:-1] - x[:-2]
    h2 = x[2:] - x[1:-1]

    yl, yc, yr = f[:-2], f[1:-1], f[2:]

    fderiv2 = np.zeros_like(f)
    fderiv2[1:-1] = 2*(yl*h2 + yr*h1 - yc*(h1+h2)) / (h1*h2*(h1+h2))
    
    fderiv2[0] = fderiv2[1]
    fderiv2[-1] = fderiv2[-2]

    return fderiv2

def second_deriv_avoid_cancelation(f, x, degree=1e-10):
    """Second order second derivative"""
    il = np.arange(0, len(x)-2)
    ic = np.arange(1, len(x)-1)
    ir = np.arange(2, len(x))

    def cancelation_degree():
        h1, h2 = x[ic] - x[il], x[ir] - x[ic]
        with np.errstate(divide='ignore', invalid='ignore'):
            f2d = 2*(f[il]*h2 + f[ir]*h1 - f[ic]*(h1+h2)) / (h1+h2)
            return np.nanmin((np.abs(f2d)/(f[ic]), np.abs((h1+h2)/np.abs(x[ic]))), axis=0)

    for i in range(0, len(x)//2):
        sel = cancelation_degree() < degree
        if np.sum(sel) == 0:
            break

        ir[sel] = ir[sel]+1
        il[sel] = il[sel]-1

        iadd = np.zeros_like(il)

        # hit a boundary, shift everything 1 extra
        iadd[il < 0] = 1
        iadd[ir >= len(x)] = -1

        ir = ir + iadd
        il = il + iadd
        ic = ic + iadd
        
        if (i == len(x)//2 - 1):
            raise ValueError("Could not find a non-canceling neighbor")

    h1, h2 = x[ic] - x[il], x[ir] - x[ic]

    fderiv2 = np.zeros_like(f)
    fderiv2[1:-1] = 2*(f[il]*h2 + f[ir]*h1 - f[ic]*(h1+h2)) / (h1*h2*(h1+h2))
    
    fderiv2[0] = fderiv2[1]
    fderiv2[-1] = fderiv2[-2]

    return fderiv2

# ===================== Coordinate map functions =========================== #

def e_l_of_rp_ra(pot, rp, ra, perturb_circular=False, eps=1e-6):
    if perturb_circular:
        # for circular orbits the formula below don't work
        # However, in a "finite-diffrences" sense they are stll correct, so we can
        # get aways by perturbing ra a little
        circular = (ra > rp) & (ra <= rp*(1+eps))
        ra = np.copy(ra)
        ra[circular] = rp[circular]*(1+eps) 

    phip, phia = pot(rp), pot(ra)
    E = phip + (phia - phip)*ra**2 / (ra**2 - rp**2)
    L = np.sqrt(2. * (phia - phip) / (rp**-2 - ra**-2))

    assert np.all(~np.isnan(L))

    return E,L

def Jacobian_ldlde_drpdra(pot, accr, rp, ra, get_el=False):
    """The jacobian need for a substitution of angular momentum and energy through
    peri and apocenter radii multiplied with the angular moment L
    """
    acca, accp = accr(ra), accr(rp)
    phia, phip = pot(ra), pot(rp)
    res = ra*rp*(ra**3*acca - ra*rp**2*acca + 2*rp**2*phia - 2*rp**2*phip)
    res = res *(ra**2*rp*accp + 2*ra**2*phia - 2*ra**2*phip - rp**3*accp)
    res = save_divide(res, (ra - rp)**3*(ra + rp)**3)
    
    res = np.nan_to_num(res, 0)
    if get_el:
        e = save_divide(phia*ra**2 - phip*rp**2, ra**2 - rp**2)
        l = np.sqrt(2*save_divide(phia - phip,rp**-2 - ra**-2))
        return e,l,np.abs(res)
    else:
        return np.abs(res)

# ============================== Other ===================================== #

def fit_powerlaw(x1,x2,y1,y2):
    slope = (np.log(y2) - np.log(y1)) / (np.log(x2) - np.log(x1))
    amp = y2 / x2**slope
    return amp, slope
