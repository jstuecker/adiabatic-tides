import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.integrate import simps, trapezoid
from scipy.interpolate import CubicSpline
from . import integrals

def RvirOfMvir(mvir, mode="crit", delta=200., h=0.679, omega_m=0.30):
    """Returns the virial radius of a halo with a given virial mass
    
    It is assumed that mvir is the mass enclosed inside that radius and
    that the object is 'delta' times as dense as the critical/mean
    density of the universe
    
    mvir : virial mass in solar masses
    mode : can be 'crit' or 'mean' to use the critical or mean density
    delta : the over-density. Typically 200 is used here
    h : reduced hubble parameter. If set to 1 units will change to
        mass in Msol/h and radius in Mpc/h
    omega_m : matter density parameter, only relevant when using mode='mean'
    
    returns : virial radius in Mpc
    """
    G = 43.0071057317063 * 1e-10  #  Grav. constant in Mpc (km/s)^2 / Msol
    rhocrit = 3.0 / (8.0 * np.pi * G) * (1e2*h)**2
    
    if mode == "crit":
        rhoref = rhocrit
    elif mode == "mean":
        rhoref = omega_m * rhocrit
    else:
        raise ValueError("Unknown mode=%s, can be 'crit' or 'mean'" % mode)

    return  np.cbrt(mvir / (rhoref * 4.*np.pi/3. * delta))

def MvirOfRvir(rvir, mode="crit", delta=200., h=0.679, omega_m=0.30):
    """Returns the virial mass of a halo with a given virial radius
    
    It is assumed that mvir is the mass enclosed inside rvir and
    that the object is 'delta' times as dense as the critical/mean
    density of the universe
    
    rvir : virial radius in Mpc
    mode : can be 'crit' or 'mean' to use the critical or mean density
    delta : the over-density. Typically 200 is used here
    h : reduced hubble parameter. If set to 1 units will change to
        mass in Msol/h and radius in Mpc/h
    omega_m : matter density parameter, only relevant when using mode='mean'
    
    returns : virial mass in Msol
    """
    G = 43.0071057317063 * 1e-10  #  Grav. constant in Mpc (km/s)^2 / Msol
    rhocrit = 3.0 / (8.0 * np.pi * G) * (1e2*h)**2
    
    if mode == "crit":
        rhoref = rhocrit
    elif mode == "mean":
        rhoref = omega_m * rhocrit
    else:
        raise ValueError("Unknown mode=%s, can be 'crit' or 'mean'" % mode)
        
    return rhoref * 4.*np.pi/3. * delta * rvir**3

def get_mass_profile(ri, mi, rbins):
    """Returns the mass profile, given some particle radii
    
    ri : radii of particles
    mi : masses of the particles
    rbins : radial bins to use
    
    returns : (rho, mprof) where rho is the density profile and 
              mprof is the cumulative mass profile
              each has shape len(rbins) -1
    """
    vbins = 4./3.*np.pi*(rbins[1:]**3 - rbins[:-1]**3)
    m,_ = np.histogram(ri, bins=rbins, weights=mi)

    # Estimate the error in the mass profile that results
    # from the numpy histogram being based on a cumulative sum
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_inc = np.nanmin(np.clip(m,np.min(mi),None)/np.cumsum(m))
    if rel_inc <= 1e-13:
        print("Warning: I expect cancellation in the mass-profile calculation, worst mass ratio = %.2e" % (rel_inc))

    rho = m / vbins
    mprof = np.concatenate([[0.], np.cumsum(m)])

    return rho, mprof

def random_direction(size, ndim):
    """Samples random unit vectors
    
    size : number of vectors to sample, can be tuple
    ndim : the dimension of the space
    
    returns : array with shape (*size, ndim)
    """
    x = np.random.normal(size=tuple(size) + (ndim,))
    r = np.sqrt(np.sum(x**2,axis=-1))
    return x/r[...,np.newaxis]


def trapez_integral_cumulative(xi, fi):
    """Calculates the cumulative integral of a function using the trapez-rule
    
    xi : locations of the function points. Shape (Nbins,...)
    fi : f(xi) the values of the function at these points. Shape (Nbins,...)
    
    returns: the integral between (xi[0] and xi[:]). Starts with 0 at index 0.
         Shape (Nbins,...)"""
    
    Ii = np.cumsum(0.5 * (fi[...,1:] + fi[...,:-1]) * (xi[...,1:] - xi[...,:-1]), axis=-1)
    
    return np.insert(Ii, 0, 0, axis=-1)

def trapez_integral_lastax(xi, fi):
    """Calculates the integral of a function using the trapez-rule over the last axis
    
    xi : locations of the function points. Shape (..., Nbins)
    fi : f(xi) the values of the function at these points. Shape (..., Nbins)
    
    returns: The integral. Shape (...)"""
    Ii = np.sum(0.5 * (fi[...,1:] + fi[...,:-1]) * (xi[...,1:] - xi[...,:-1]), axis=-1)
    
    return Ii

def powerlaw_trapez_integral_cumulative(xi, fi):
    """Approximates the integrand as a powerlaw on each interval"""
    slopes = (np.log(fi[1:]) - np.log(fi[:-1]))/(np.log(xi[1:]) - np.log(xi[:-1]))
    amps = fi[1:] / xi[1:] ** slopes

    Ipl = amps * (xi[1:]**(slopes+1) - xi[:-1]**(slopes+1)) / (slopes+1)
    sel = np.isnan(Ipl) | (slopes == 0.)

    if(np.sum(sel) > 0):
        Itrapez = 0.5 * (fi[1:] + fi[:-1]) * (xi[1:] - xi[:-1])
        Ipl[sel] = Itrapez[sel]
        print("Warning %d out of %d points undefined as powerlaws and replaced by linear trapezes" % (np.sum(sel), len(sel)))

    Ii = np.cumsum(Ipl)
    
    return np.concatenate([[0.], Ii])

def extended_simpson_cumulative(fi, dx):
    """Adapted From Numerical Recipes 4.1.14
    only works for constant dx"""
    Ii = np.cumsum(fi)

    # Correct contributions from start boundaries
    Ii += -5./8. * fi[0] + 1./6. * fi[1] - 1./24. * fi[2]
    
    # Handle upper boundaries leading up to each point
    Ii -= 5./8. * fi
    Ii[1:] += 1./6. * fi[:-1]
    Ii[2:] -= 1./24. * fi[:-2]

    # For first points use trapezoidal rule
    Ii[0] = 0
    Ii[1] = 0.5 * (fi[0] + fi[1])

    return Ii * dx

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


def cum_simpson(f, xi, fi=None, **kwargs):
    """Returns the cumulative integral of a function using the simpson rule
    
    f : a function
    xi : evaluation points (the midpoint of each interval will as well be evaluated)
    fi : function values at xi, providing these is just an optimization
    ** kwargs : key word arguments that will be passed through to the function
    
    returns : cumulative integral from xi[0] to xi[:]. Shape is same as xi
    """
    if fi is None:
        fi = f(xi, **kwargs)
    
    xh = 0.5*(xi[1:] + xi[:-1])
    
    Is = (xi[1:] - xi[:-1])/6. * (fi[1:] + 4.*f(xh, **kwargs) + fi[:-1])
    
    return np.insert(np.cumsum(Is, axis=0), 0, 0.)

def remove_zero_intervals(xi):
    # adds shift in xi so that there is a dx of zero nowhere
    dx = xi[...,1:] - xi[...,:-1]

    mindx = np.min(dx[dx>0.])
    dx[dx == 0.] = mindx + np.abs(xi[...,1:][dx == 0.])*1e-12 
    
    newx = np.concatenate([xi[...,0:1]*0., np.cumsum(dx, axis=-1)], axis=-1)

    assert np.min(newx[...,1:] - newx[...,:-1]) > 0.
    
    return newx

def simpson_2d(fgrid, xi, ygrid, axisx=0, axisy=1):
    """Applies the simpson rule over a 2d integration domain of a function f(x,y)
    The boundaries in y can vary as a function of x, but the x-boundaries need to be
    fixed
    
    fgrid : the function evaluation locations. Shape (...,nx,ny)
    xi : x evaluation locations. Shape (...,nx)
    yi : y evaluation locations. Shape (...,nx,ny)
    axis0 : the axis that should be summed for x-integration
    axis1 : the axis that should be summed for y-integration
    
    returns : the integral. shape(...)
    """

    newy = remove_zero_intervals(ygrid)

    Iy = simps(fgrid, x=newy, axis=axisy) # x = ygrid
    
    assert np.max(np.isnan(Iy)) == False

    newx  = remove_zero_intervals(xi)
    Ixy = simps(Iy, x=newx, axis=axisx) # x = xi

    assert np.max(np.isnan(Ixy)) == False
    
    return Ixy

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

def sample_metropolis_hastings(f, x0, stepsize=1., nsteps=1000):
    """Does an mcmc sampling of a probability distribution function
    only returns the last step of each chain.
    
    f : a pdf to sample
    x0 : start locations
    sig : standard deviation(s) for step sizes
    nsteps : total number of steps to perform
    """
    
    x = np.copy(x0)
    f0 = f(x)

    if np.sum(f0 <= 0) > 0:
        print("Warning, I am starting with invalid points... %d" % np.sum(f0 <= 0))
        #raise ValueError("Invalid starting points")

    for i in range(0, nsteps):
        dx = np.random.normal(loc=0., scale=stepsize, size=x.shape)

        f1 = f(x + dx)

        alpha = f1 / f0
        u = np.random.uniform(0., 1., size=x.shape[0])
        accept = u <= alpha

        x[accept] = (x+dx)[accept]
        f0[accept] = f1[accept]

    if np.sum(f0 <= 0) > 0:
        print("Warning, I ended with invalid points... %d" % np.sum(f0 <= 0))
        #raise ValueError("Invalid starting points")
        
    return x

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

def fit_powerlaw(x1,x2,y1,y2):
    slope = (np.log(y2) - np.log(y1)) / (np.log(x2) - np.log(x1))
    amp = y2 / x2**slope
    return amp, slope

def solve_poisson(ri, rho, boundary="powerlaw", integration_mode="trapez", G=43.0071057317063e-10):
    """Solve Poisson's equation returning m(<r) and phi (normalized to 0 at 0)
        
    ri : radius sampling points
    rhoi : densities
    boundary : How to handle radii r < min(ri) if min(ri) > 0. 
                Can be "constant" or "powerlaw"
                For the powerlaw case a powerlaw profile is fitted based on the
                two smallest radii. This is the recommended mode if applicable.
    integration_mode : don't change for now
    """

    assert np.all(ri[1:] > ri[:-1])

    if ri[0] == 0.:
        m0 = phi0 = 0.
    elif boundary == "powerlaw":
        rhoc, alpha = fit_powerlaw(ri[0], ri[1], rho[0], rho[1])
        # See profiles.PowerlawProfile for seeing the powerlaw normalization
        
        m0 = 4.*np.pi * rhoc  / (3. + alpha) * ri[0]**(3.+alpha)
        if alpha > -2:
            phi0 = 4.*np.pi * G * rhoc / ( (3. + alpha) * (2. + alpha) ) * ri[0]**(2.+alpha)
        else:
            raise ValueError("Potential is not normalizable at 0 consider using boundary=constant (inner slope %.2f)" % alpha)
    elif boundary == "constant":
        m0 = 4.*np.pi/3. * rho[0] * ri[0]**3
        phi0 = 0.
    else:
        raise ValueError("Unknown boundary mode %s" % boundary)
    
    if integration_mode == "trapez":
        m = m0 + trapez_integral_cumulative(ri, 4.*np.pi*rho*ri**2)
        with np.errstate(divide='ignore', invalid='ignore'):
            phi = phi0 + trapez_integral_cumulative(ri, np.nan_to_num(G * m / ri**2, 0))
    elif integration_mode == "powerlaw_trapez":
        m = m0 + powerlaw_trapez_integral_cumulative(ri, 4.*np.pi*rho*ri**2)
        phi = phi0 + powerlaw_trapez_integral_cumulative(ri, G * m / ri**2)
    else:
        raise ValueError("Unknown integration mode %s" % integration_mode)
    
    return m, phi

def cosh_space(fmax, n, pow=1.):
    x = np.linspace(0., np.arccosh(fmax), n)
    return np.cosh(x[-1] * (x/x[-1])**pow)

def eddington_inversion(ri, rho, phi=None, integrator=None):
    """Does the Eddington inversion at discrete energies ei=phi
    
    phi : potential -- if not provided a simple Poisson solver is used
          assuming that the distribution rho generates the potential

    to avoid the singularity we use a substitution t = sqrt(phi - E)

    f(E) &= \frac{1}{\sqrt{8} \pi^2} \frac{d}{dE} \int_E^{E_{max}} \frac{d \rho}{d \Phi} (\Phi - E)^{-1/2} d \Phi \\
        &= \frac{1}{\sqrt{8} \pi^2} \frac{d}{dE} \int_0^{\sqrt{E_{max} - E}} 2 \frac{d \rho}{d \Phi} (\phi = E + t^2)  d t
    """
    if integrator is None:
        integrator = trapezoid
    if phi is None:
        m, phi = solve_poisson(ri, rho)

    d2rhodphi2 = second_deriv_avoid_cancelation(rho, phi)

    integrand = d2rhodphi2 * (2/(np.sqrt(8.) * np.pi**2))

    f = np.zeros_like(phi)
    for i,E in enumerate(phi):
        t = np.sqrt(np.clip(phi - E, 0, None))
        f[i] = integrator(integrand, x=t)
    return phi, f

def eddington_inversion_adaptive(ri, prof, nintegrate=None):
    """Does the Eddington inversion for discrete energies ei=phi(ri), but using
       adaptively spaced integration points.
       nintegrate: number of integration steps. Defaults to len(ri)
    """
    if nintegrate is None:
        nintegrate = len(ri)
    
    rho, phi = prof.density(ri), prof.potential(ri)

    d2rhodphi2 = second_deriv_avoid_cancelation(rho, phi)

    spl_d2rhodphi2 = CubicSpline(phi, d2rhodphi2)

    def integrand(v):
        E = prof.potential(ri)[:,np.newaxis] + 0.5*v**2
        return spl_d2rhodphi2(E) * (E <= phi[-1])
    
    vscale = prof.vcirc(ri)
    I = integrals.integrate_exp_0_inf(integrand, nintegrate, xscale=vscale)

    return phi, I / (2. * np.pi**2)

def eddington_inversion_diff_last(ri, rho, phi=None, integrator=None):
    """Does the Eddington inversion at discrete energies ei=phi
    using a different approach where the parent function is calculated first
    
    phi : potential -- if not provided a simple Poisson solver is used
          assuming that the distribution rho generates the potential

    to avoid the singularity we use a substitution t = sqrt(phi - E)

    f(E) &= \frac{1}{\sqrt{8} \pi^2} \frac{d}{dE} \int_E^{E_{max}} \frac{d \rho}{d \Phi} (\Phi - E)^{-1/2} d \Phi \\
         &= \frac{1}{\sqrt{8} \pi^2} \frac{d}{dE} \int_0^{\sqrt{E_{max} - E}} 2 \frac{d \rho}{d \Phi} (\phi = E + t^2)  d t
    """
    if integrator is None:
        integrator = trapezoid
    if phi is None:
        m, phi = solve_poisson(ri, rho)

    drhodphi = np.gradient(rho, phi, edge_order=1)

    integrand = -drhodphi * (2/(np.sqrt(8.) * np.pi**2))

    fparent = np.zeros_like(phi)
    for i,E in enumerate(phi):
        t = np.sqrt(np.clip(phi - E, 0, None))
        fparent[i] = integrator(integrand, x=t)

    return phi, -np.gradient(fparent, phi, edge_order=1)

def anisotropic_inversion(ri, rho, phi=None, beta=0.):
    """Assuming a profile with constant anisotropy beta, calculates f1(E)
    assuming that f(E,L) = f1(E) * L**(-2beta)
    
    phi : potential -- if not provided a simple Poisson solver is used
          assuming that the distribution rho generates the potential
    """
    if phi is None:
        m, phi = solve_poisson(ri, rho)

    rho_rbeta2 = rho * ri**(2*beta)

    d2rb2 = second_deriv_avoid_cancelation(rho_rbeta2, phi)

    integrand = d2rb2

    f = np.zeros_like(phi)
    for i,E in enumerate(phi):
        t = (np.clip(phi - E, 0, None))**(beta + 0.5)
        f[i] = trapezoid(integrand, x=t)

    from scipy.special import gamma

    Ibeta = np.sqrt(np.pi) * gamma(1. - beta) / gamma(1.5 - beta)
    fac = 2**(beta - 0.5) * np.cos(beta*np.pi) 
    fac /= 2.*np.pi**2 * Ibeta * (0.5 - beta) * (0.5 + beta)

    return phi, f*fac

def integrate_f_to_density(ei, fi):
    """Integrates a phase space distribution to obtain rho(phi)
    See Binney and Tremaine (4.43)
    """
    rho_phi = np.zeros_like(fi)
    for i,phi in enumerate(ei):
        integrand = fi * np.sqrt(np.clip(ei - phi, 0, None)) 
        rho_phi[i] = trapezoid(integrand, ei) * (np.sqrt(2.)*4.*np.pi)
    
    return rho_phi

def integrate_f_to_density_adaptive(f_of_e, phi, nintegrate=200):
    """Integrates a phase space distribution to obtain rho(phi)
    See Binney and Tremaine (4.43)
    Chooses the evaluation points adaptively
    """
    assert np.min(phi) > 0, "Please normalize potential to zero at zero"

    def integrand(de):
        return f_of_e(phi[...,np.newaxis] + de) * np.sqrt(de)
    
    fphi = f_of_e(phi)
    phiscale = np.interp(fphi*0.5, fphi[::-1], phi[::-1])
    
    return integrals.integrate_exp_0_inf(integrand, nintegrate, xscale=phiscale)* (np.sqrt(2.)*4.*np.pi)

def integrate_f_to_density_perisplit_adaptive(ri, pot, f_of_e, rp1=0, rp2=np.infty, nintegrate=None, rmaxfac=1e10):
    """Integrates a phase space distribution to obtain the density
    but limits to orbits which have pericenters in rp1 < rp < rp2
    """
    if nintegrate is None:
        nintegrate = len(ri)

    phip1, phip2 = pot(rp1), pot(rp2)

    assert not np.isnan(phip1)

    rho = np.zeros_like(ri)
    for i,r in enumerate(ri):
        reval = r * cosh_space(rmaxfac, nintegrate, 2)
        phi, eeval = pot(r), pot(reval)

        with np.errstate(divide='ignore', invalid='ignore'):
            q1 = np.clip(eeval - phi - np.clip(eeval - phip1, 0, None) * (rp1**2 / r**2), 0, None) * (r >= rp1)
            q2 = np.clip(eeval - phi - np.clip(eeval - phip2, 0, None) * (rp2**2 / r**2), 0, None) * (r >= rp2)

            integrand = f_of_e(eeval) * (np.nan_to_num(np.sqrt(q1),0) - np.nan_to_num(np.sqrt(q2),0))

        rho[i] = trapezoid(integrand, eeval) * (np.sqrt(2.)*4.*np.pi)
    
    return rho

def sample_from_Finv(Finv, size):
    Fs = np.random.uniform(0., 1., size)

    return Finv(Fs)

def sample_radii(ri, mi, size=1, rmax=None):
    """sample radii from a given mass profile

    returns sampled radii and total mass
    """
    mi = mi-mi[0]

    if rmax is None:
        Mmax = mi[-1]
    else:
        Mmax = np.interp(rmax, ri, mi)

    def FcumInv(f):
        return np.interp(f, mi/Mmax, ri)
    
    return sample_from_Finv(FcumInv, size)

def sample_rimi_from_density(ri, rhoi, size=1, rmax=None, weights=None):
    """ sample radii and masses from a density profile
        weights : can be provided to sample more particles (with lower weights)
                  at different radii. The number of particles at a radius will
                  be proportional to rhoi*weights, but the masses to 1/weights
    """
    mi = trapez_integral_cumulative(ri, 4.*np.pi*rhoi*ri**2)
    if rmax is None:
        Mmax = mi[-1]
    else:
        Mmax = np.interp(rmax, ri, mi)

    if weights is None:
        rsamp = sample_radii(ri, mi, size=size, rmax=rmax)
        msamp = np.ones_like(rsamp) * Mmax / len(rsamp)
    else:
        miwi = trapez_integral_cumulative(ri, 4.*np.pi*rhoi*ri**2*weights)
        rsamp = sample_radii(ri, miwi, size=size, rmax=rmax)
        msamp = 1. / np.interp(rsamp, ri, weights)
        msamp *= Mmax / np.sum(msamp) # normaliz

    return rsamp, msamp

def integrate_fiso_cumulative_phi_e(ei, fi):
    """Integrates a phase space distribution to obtain rho(phi, <E)
    This is useful for sampling the distribution with particles

    \\rho(\\phi, < E) = 4 \\pi \\int_\\phi^{E} f(E) \\sqrt{2E' - 2\\phi} \\text{\\quad}  dE'
    """
    rho_phi = np.zeros((len(ei), len(ei)), dtype=ei.dtype)
    for i,phi in enumerate(ei):
        integrand = fi * np.sqrt(np.clip(ei - phi, 0, None))  * (np.sqrt(2.)*4.*np.pi)
        rho_phi[i] = trapez_integral_cumulative(ei, integrand)
    
    return rho_phi

def sample_conditional_energy(phisamp, ei, fi, emaxsamp=None):
    """Samples the energy, given that the particle is at a radius where the potential is phi
    phisamp : potential energies of sampled particles
    ei, fi: phase space distribution as function of energy
    """
    print("Warning this method is deprecated, use adaptive one!")
    
    rho_phi_e = integrate_fiso_cumulative_phi_e(ei, fi)
    with np.errstate(divide='ignore', invalid='ignore'):
        Fcum = rho_phi_e / rho_phi_e[:,-1:]
    Fsamp = np.random.uniform(0., 1., phisamp.shape)

    itab = np.interp(phisamp, ei, np.arange(len(ei)))

    esamp = np.zeros_like(phisamp)
    for i in range(0, len(phisamp)):
        #Fcum_sel = np.interp(phisamp, ei, Fcum) # select the correct row of our table
        #i0 = np.floor(itab[i]).astype(int)
        #di = itab[i] - i0
        Fcum_sel = Fcum[itab[i].astype(int)]
        #Fcum_sel = Fcum[i0] * (1-di) + Fcum[i0+1] * di
        esamp[i] = np.interp(Fsamp[i], Fcum_sel, ei) # do  the inversion sampling
    
    return esamp

def sample_conditional_energy_adaptive(phisamp, f_of_e, emax=None, nintegrate=1000):
    """Samples the energy, given that the particle is at a radius where the potential is phi
    phisamp : potential energies of sampled particles
    ei, fi: phase space distribution as function of energy
    """

    assert np.min(phisamp) > 0, "Please normalize potential to zero at zero"
    
    Fsamp = np.random.uniform(0., 1., phisamp.shape)

    if emax is None:
        emax = np.max(phisamp)*1e3

    Esamp = np.zeros_like(phisamp)
    for i,phi in enumerate(phisamp):
        eeval = phi * cosh_space(emax/phi, nintegrate, 2)
        assert ~np.isnan(np.max(eeval))
        
        integrand = f_of_e(eeval) * np.sqrt(np.clip(eeval - phi, 0, None)) 
        fcum = trapez_integral_cumulative(eeval, integrand)

        Esamp[i] = np.interp(Fsamp[i], fcum/fcum[-1], eeval)
    
    return Esamp

def sample_conditional_energy_perisplit_adaptive(rsamp, pot, f_of_e, rp1=0, rp2=np.infty, nintegrate=1000, rmaxfac=1e10):
    """Samples the energy, given that the particle is at a radius where the potential is phi
    but limits to orbits which have pericenters in rp1 < rp < rp2
    phisamp : potential energies of sampled particles
    ei, fi: phase space distribution as function of energy
    """
    Fsamp = np.random.uniform(0., 1., rsamp.shape)

    phip1, phip2 = pot(rp1), pot(rp2)

    rfacs = cosh_space(rmaxfac, nintegrate, 2)

    Esamp = np.zeros_like(rsamp)
    for i,r in enumerate(rsamp):
        reval = r * rfacs

        phi, eeval = pot(r), pot(reval)

        q1 = np.clip(eeval - phi - np.clip(eeval - phip1, 0, None) * (rp1**2 / r**2), 0, None) * (r >= rp1)
        q2 = np.clip(eeval - phi - np.clip(eeval - phip2, 0, None) * (rp2**2 / r**2), 0, None) * (r >= rp2)
        q2 = np.nan_to_num(q2, 0)
        integrand = f_of_e(eeval) * (np.sqrt(q1) - np.nan_to_num(np.sqrt(q2),0))

        fcum = trapez_integral_cumulative(eeval, integrand)

        Esamp[i] = np.interp(Fsamp[i], fcum/fcum[-1], eeval)
    
    return Esamp

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

def sample_conditional_energy_adaptive_batched(phisamp, f_of_e, emax=None, nintegrate=1000, batchsize=500):
    """Samples the energy, given that the particle is at a radius where the potential is phi
    phisamp : potential energies of sampled particles
    ei, fi: phase space distribution as function of energy
    """

    assert np.min(phisamp) > 0, "Please normalize potential to zero at zero"
    
    Fsamp = np.random.uniform(0., 1., phisamp.shape)

    if emax is None:
        emax = np.max(phisamp)*1e3

    facspace = cosh_space(emax/np.min(phisamp), nintegrate, 2)

    Esamp = np.zeros_like(phisamp)

    nlow = np.arange(0, len(phisamp), batchsize)
    nup = np.clip(nlow + batchsize, 0, len(phisamp))

    for ilow,iup in zip(nlow, nup):
        phi = phisamp[ilow:iup,np.newaxis]
        eeval = phi * facspace
        assert ~np.isnan(np.max(eeval))
        
        integrand = f_of_e(eeval) * np.sqrt(np.clip(eeval - phi, 0, None)) 
        fcum = trapez_integral_cumulative(eeval, integrand)

        Esamp[ilow:iup] = vectorized_interp(Fsamp[ilow:iup], fcum/fcum[:,-1:], eeval)
    
    return Esamp

def sample_conditional_vr_L_isotropic(r, dE):
    """Input: dE=E-phi(r)
    Output: vr, L
    """
    vel= random_direction(dE.shape, 3) * np.sqrt(2.*dE)[...,np.newaxis]
    # assume r = x-axis
    vr = vel[...,0]

    L = np.linalg.norm([0.*r, -r * vel[...,2], r * vel[...,1]], axis=0)

    return vr, L

def sample_conditional_L_vr_perisplit(r, E, pot, rp1, rp2):
    """Samples angular momenta for a given (r,E) and
    assuming that the peri center has to be in the range (rp1, rp2)
    """
    Fsamp = np.random.uniform(0., 1., r.shape)

    phip1, phip2 = pot(rp1), pot(rp2)
    phi = pot(r)

    assert np.all(E >= phi)
    assert np.all(r >= rp1), "Your input radii are not quite consistent. Maybe a numerical error from radial bins not lining up with peri center radii?"

    def Fu(L2): # Parent function of ang-mom distr. (for isotropic systems)
        return -np.sqrt(np.clip(2.*(E - phi) - L2/r**2,0,None))
    
    Lmin2 = np.clip(2.*(E - phip1) * rp1**2, 0, None)
    Lmax2 = 2.*(E - phi) * r**2
    Lmaxb2 = 2.*(E - phip2) * rp2**2
    Lmax2[r >= rp2] = Lmaxb2[r >= rp2]

    # Cumulative Function is F = (Fu(L) - Fu(Lmin))/(Fu(Lmax) - Fu(Lmin))
    # Invert this
    FuL = Fsamp * (Fu(Lmax2) - Fu(Lmin2)) + Fu(Lmin2)
    L2 = (2. * (E-phi) - FuL**2) * r**2

    vr = np.sqrt(np.clip(2.*(E - phi) - L2/r**2,0,None))
    vr = vr * np.sign(np.random.uniform(-1,1, r.shape))

    return np.sqrt(L2), vr

def integrate_radial_orbits(acc_func, r, vr, L, t, nsteps=1000):
    # Hamiltonian = phi(r) + 0.5 vr**2 + 0.5 L**2 / r**2
    # dvr/dt = -dphi/dr - L**2 / r**3
    
    dt = t/nsteps

    for i in range(nsteps):    
        # Drift Kick Drift Integrator
        r = r + vr*dt*0.5
        vr = vr + (acc_func(r) + L**2/r**3) * dt
        r = r + vr*dt*0.5

    return r, vr

def integrate_radial_orbits_with_snaps(acc_func, r, vr, L, t, nsnaps=10, nsteps_per_snap=100):
    for i in range(0, nsnaps):
        r, vr = integrate_radial_orbits(acc_func, r, vr, L, t/nsnaps, nsteps=nsteps_per_snap)
        yield r, vr

def integrate_to_density_of_states(ri, phii, rmax=np.infty):
    """Calculates the density of states
    See Binney and Tremaine (4.56)
    """
    gE = np.zeros_like(ri)
    for i,phi in enumerate(phii):
        integrand = np.sqrt(np.clip(phi - phii, 0, None)) * ri**2 * (ri < rmax)
        gE[i] = trapezoid(integrand, ri) * (4.*np.pi)**2 * np.sqrt(2.)
        
    return gE

def integrate_to_density_of_states_adaptive(phi, ri, nintegrate=200):
    """Calculates the density of states
    See Binney and Tremaine (4.56)
    """
    E = phi(ri)

    def integrand(r):
        return np.sqrt(E[...,np.newaxis] - phi(r)) * r**2
    
    return integrals.integrate_tanh_a_b(integrand, 0., ri, N=nintegrate) * (4.*np.pi)**2 * np.sqrt(2.)

def integrate_fofel_adaptive(f_of_el, phi, r, N=100):
    r = np.array(r)

    Escale = np.clip(phi(r*2.) - phi(r), 0, None)
    def integrate_vl(f_of_el, phi, vr, r, N=100):
        vlscale = np.clip(np.abs(vr), np.sqrt(Escale)[...,np.newaxis], None)
        def integrand(vl):
            E = (phi(r)[...,np.newaxis] + 0.5*vr**2)[...,np.newaxis] + 0.5*vl**2
            L = vl*r[...,np.newaxis,np.newaxis]
            return 2.*np.pi*vl * f_of_el(E, L)
        return integrals.integrate_exp_0_inf(integrand, N=N, xscale=vlscale)
    
    def integrand(vr):
        return integrate_vl(f_of_el, phi, vr, r, N=N)
    
    vrscale = np.sqrt(phi(r*2.) - phi(r))
    
    return 2.*integrals.integrate_exp_0_inf(integrand, N=N, xscale=vrscale)

def integrate_fofel_adaptive_rperi_lim(f_of_el, phi, r, rp1=1e-10, rp2=1e10, N=100):
    r = np.array(r)
    rho = np.zeros_like(r)
    if rp1 == 0.:
        rp1 = np.min(r) * 1e-5

    assert rp2 > rp1

    sel = r > rp1
    r = r[sel]

    def integrate_vl(f_of_el, phi, vr, r, N=100):
        def integrand(vl):
            E = (phi(r)[...,np.newaxis] + 0.5*vr**2)[...,np.newaxis] + 0.5*vl**2
            L = vl*r[...,np.newaxis,np.newaxis]
            return 2.*np.pi*vl * f_of_el(E, L)
        
        phip1, phip2 = phi(rp1), phi(rp2)
        phir = phi(r)

        # Lmin2 = np.clip((vr**2 + 2*(phir-phip1)[...,np.newaxis])/(rp1**-2 - r**-2)[...,np.newaxis], 0, None)
        # Lmax2 = np.clip((vr**2 + 2*(phir-phip2)[...,np.newaxis])/(rp2**-2 - r**-2)[...,np.newaxis], 0, None)
        vlmin2 = np.clip((vr**2 + 2*(phir-phip1)[...,np.newaxis])/(r**2/rp1**2 - 1.)[...,np.newaxis], 0, None)
        with np.errstate(divide='ignore', invalid='ignore'):
            vlmax2 = np.clip((vr**2 + 2*(phir-phip2)[...,np.newaxis])/(r**2/rp2**2 - 1.)[...,np.newaxis], 0, None)

        vlmax2[r <= rp2] = (phip2-phip1)*1e5

        return integrals.integrate_tanh_a_b(integrand, np.sqrt(vlmin2), np.sqrt(vlmax2),  N=N)
    
    
    def integrand(vr):
        return integrate_vl(f_of_el, phi, vr, r, N=N)
    
    vrscale = np.sqrt(phi(r*2.) - phi(r))

    
    rho[sel] = 2.*integrals.integrate_exp_0_inf(integrand, N=N, xscale=vrscale)
    
    return rho

def sample_E_L_vr_given_r_metropolis(f_of_el, pot, vcirc, rs, nsteps_chain=100):
    phis = pot(rs)
    vref = vcirc(rs)
    
    def likelihood_of_vel_given_r(logvtheta):
        # Likelihood in polar coordinates in velocity space
        vs,thetas = np.exp(logvtheta[...,0]), logvtheta[...,1]

        es = phis + 0.5*vs**2
        ls = vs * rs * np.abs(np.sin(thetas))
        
        return f_of_el(es,ls) * vs**3 * np.abs(np.sin(thetas))

    logv0 = np.random.uniform(-2., 2., rs.shape) + np.log(vref)
    theta0 = np.random.uniform(0., np.pi, rs.shape)
    
    logvtheta = np.stack([logv0,theta0], axis=-1)
    stepsize = np.stack([2.0, 0.2*np.pi], axis=-1)
    
    logvtheta = sample_metropolis_hastings(likelihood_of_vel_given_r, logvtheta, stepsize=stepsize, nsteps=nsteps_chain)
    vs, thetas = np.exp(logvtheta[...,0]), logvtheta[...,1]
    Ls = vs *rs * np.abs(np.sin(thetas))
    vrs = vs * np.cos(thetas)
    Es = phis + 0.5*vs**2

    return Es, Ls, vrs

def sample_E_L_vr_given_r_metropolis_perisplit(f_of_el, pot, accr, rs, nsteps_chain=40, rp1=None, rp2=None):
    phis = pot(rs)

    assert rp2 > rp1

    # we have to sample from
    # f(E,L) v^2 sin(theta) dv dtheta
    # L = v r sin(theta)
    # vr = v cos(theta)

    # parameterize v in terms of u_p = r_p/r
    # where r_p is the pericenter radius
    # this way it is easy to predict the relevant
    
    def dv_du_overv(u, sintheta2, rs, phis):
        rp = rs * u
        f = u**-3 * sintheta2 / (u**-2 * sintheta2 - 1)
        f = f + accr(rp)*rs / (2*phis - 2*pot(rp))
        return f

    def likelihood_of_vel_given_r(logutheta):
        # Likelihood in polar coordinates in velocity space
        us,thetas = np.exp(logutheta[...,0]), logutheta[...,1]

        rp = rs * us
        sintheta2 = np.square(np.sin(thetas))
        vs2 = 2.*(phis - pot(rp)) / (us**-2 * sintheta2 - 1.)

        valid = (rp >= rp1) & (rp <= rs) & (rp <= rp2) & (vs2 > 0)

        es = phis[valid] + 0.5*vs2[valid]
        ls = rs[valid] * np.sqrt(vs2[valid] * sintheta2[valid])

        dvol = vs2[valid] * np.sqrt(sintheta2[valid])
        dvol *= dv_du_overv(us[valid], sintheta2[valid], rs[valid], phis[valid]) * us[valid] * np.sqrt(vs2[valid])

        f = np.zeros_like(rs)
        f[valid] = f_of_el(es,ls) * dvol
        
        return f
    
    umax = np.clip(rp2/rs,None,1.)
    umin = rp1 / rs
    logu0 = np.random.uniform(np.log(umin), np.log(umax), rs.shape)
    u0 = np.exp(logu0)
    thmin = np.arcsin(u0)
    theta0 = np.random.uniform(thmin, np.pi-thmin, rs.shape)
    
    logutheta = np.stack([logu0,theta0], axis=-1)
    stepsize = np.stack([np.log(umax/umin)*0.25, (np.pi/2.-np.arcsin(rp1/rs))/4.], axis=-1)
    
    logutheta = sample_metropolis_hastings(likelihood_of_vel_given_r, logutheta, stepsize=stepsize, nsteps=nsteps_chain)

    # Transform back
    us,thetas = np.exp(logutheta[...,0]), logutheta[...,1]
    vs = np.sqrt(2.*(phis - pot(us*rs)) / (us**-2 * np.sin(thetas)**2 - 1.))

    es = phis + 0.5*vs**2
    ls = rs * vs * np.abs(np.sin(thetas))
    vrs = vs * np.cos(thetas)

    return es, ls, vrs