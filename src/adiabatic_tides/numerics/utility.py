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

def piecewise_2_2(x1, x2, cond, f1, f2):
    fo1, fo2 = np.zeros(x1.shape), np.zeros(x1.shape)

    fo1[cond], fo2[cond] = f1(x1[cond], x2[cond])
    fo1[~cond], fo2[~cond] = f2(x1[~cond], x2[~cond])
    
    return fo1, fo2

def e_l_of_rp_ra(pot, rp, ra, accr=None, eps=1e-3, get_de=False):
    """energy and angular momentum as function of peri- and apo-center
    
    you may provide accr for handling close to circular orbits accurately

    de: If true, return de = e - phi(rp) instead of E
    """
    facphip = 0. if get_de else 1.
    def el(rp, ra): 
        phip, phia = pot(rp), pot(ra)
        de = (phia - phip)*ra**2 / (ra**2 - rp**2)
        l = np.sqrt(2. * (phia - phip) / (rp**-2 - ra**-2))
        return de + facphip*phip,l
    
    if accr is None:
        return el(rp, ra)
    
    # Expansion of the equations above around ra=rp (avoids cancellation)
    def el_expansion(rp, ra):
        phigrad = -accr(0.5*(ra+rp))
        de = phigrad * ra**2/(ra + rp)
        l2 = 2.*(ra**2*rp**2)/(ra + rp) * phigrad
        return de + facphip*pot(rp),np.sqrt(l2)
    
    e,l = piecewise_2_2(rp, ra, ra>=rp*(1+eps), el, el_expansion)

    return e,l

def Jacobian_det_ldlde_drpdra(pot, accr, rp, ra, get_el=False):
    """The jacobian determinant needed for a substitution of angular momentum and energy through
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
    
def dedl2_drpdra(pot, accr, rp, ra):
    phia, phip = pot(ra), pot(rp)
    phigrad_rp, phigrad_ra = -accr(rp), -accr(ra)

    inv_A = 1./(ra**2 - rp**2)
    e = (phia*ra**2 - phip*rp**2)*inv_A
    de_dra = (phigrad_ra*ra**2 - 2*(e-phia)*ra)*inv_A
    de_drp = (-phigrad_rp*rp**2 + 2*(e-phip)*rp)*inv_A

    inv_B = (ra**2*rp**2)*inv_A # == 1/(rp**-2 - ra**-2)
    
    l2 = 2*(phia - phip)*inv_B
    dl2_dra =  2*phigrad_ra*inv_B - 4*(phia - phip)*ra**-3*inv_B**2
    dl2_drp = -2*phigrad_rp*inv_B + 4*(phia - phip)*rp**-3*inv_B**2

    # jac = np.stack([de_drp, de_dra, dl2_drp, dl2_dra], axis=-1).reshape(rp.shape + (2,2))

    return de_drp, de_dra, dl2_drp, dl2_dra

# ============================== Other ===================================== #

def fit_powerlaw(x1,x2,y1,y2):
    slope = (np.log(y2) - np.log(y1)) / (np.log(x2) - np.log(x1))
    amp = y2 / x2**slope
    return amp, slope

# ======================== Cosmology related =============================== #

def fmax_wdm(h=0.68, omega_dm=0.26, gx=1.5, mx=1., G=43.0071057317063e-10):
    """Following https://arxiv.org/pdf/2207.05082.pdf

    the phase space density of a thermal relic WDM
    """
    def v0_wdm(omega_dm=0.28, h=0.678, gx=1.5, mx=1., a=1.):
        """omgega_dm: dark matter (not full matter) density paramater, mx in kev
        result : velocity in km/s
        """
        # Bode (2001), arXiv:astro-ph/0010389
        v0 = 0.012 * a**(-1) * (omega_dm / 0.3)**(1./3.) * (h/0.65)**(2./3.) * (1.5/gx)**(1./3.) * (1./mx)**(4./3.)
        return v0

    rho_dm = 3. * (h * 100.)**2 / (8.*np.pi*G) * omega_dm

    v0 = v0_wdm(mx=mx, omega_dm=omega_dm, gx=gx, h=h)

    return 0.0221 * v0**-3 * rho_dm

def fmax_wimp(h=0.68, omega_dm=0.26,  mx=100, Td=30., ad=5.332e-12, G=43.0071057317063e-10):
    """Following https://arxiv.org/pdf/2207.05082.pdf

    mx : WIMP mass in GeV
    Td : Decoupling Temperature in MeV
    omgega_dm: dark matter (not full matter) density paramater, mx in kev
    ad : scale factor of decoupling (where the temperature of the universe is Td)
            This can be put to None to use an approximation by the neutrino temperature
            which may have errors of order 10% if the wimp decoupled a bit before the
            neutrinos
    
    the phase space density of WIMP's in Msol (km/s)**3 Mpc**3
    """
    c = 299792458.0

    mev, Tdev = mx*1e9, Td*1e6

    if ad is None:
        print("Approximating ad by assuming evaluating T(a_d)=Td while using the temperature T(a) of the Neutrino background.\n"
                "This may give inaccurate results by 10-20%. For full accuracy use a full thermal history and determine ad")
        Tcmb = 2.725 #K
        kb = 8.617333262e-5 # eV/kelvin
        Tnu = Tcmb*(4./11.)**(1./3.) * kb   # in eV
        ad = (Tnu/Tdev)
    
    v0 =  np.sqrt(Tdev * mev)*ad / mev * c / 1e3  # velocity today in km/s

    rho_dm = 3. * (h * 100.)**2 / (8.*np.pi*G) * omega_dm

    return (2.*np.pi)**(-3./2.) * v0**-3 * rho_dm