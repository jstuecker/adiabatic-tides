import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.integrate import simps

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
    
    Ii = np.cumsum(0.5 * (fi[1:] + fi[:-1]) * (xi[1:] - xi[:-1]))
    
    return np.concatenate([[0.], Ii])

def trapez_integral_lastax(xi, fi):
    """Calculates the integral of a function using the trapez-rule over the last axis
    
    xi : locations of the function points. Shape (..., Nbins)
    fi : f(xi) the values of the function at these points. Shape (..., Nbins)
    
    returns: The integral. Shape (...)"""
    Ii = np.sum(0.5 * (fi[...,1:] + fi[...,:-1]) * (xi[...,1:] - xi[...,:-1]), axis=-1)
    
    return Ii

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

    for i in range(0, nsteps):
        dx = np.random.normal(loc=0., scale=stepsize, size=x.shape)

        f1 = f(x + dx)

        alpha = f1 / f0
        u = np.random.uniform(0., 1., size=x.shape[0])
        accept = u <= alpha

        x[accept] = (x+dx)[accept]
        f0[accept] = f1[accept]
        
    return x