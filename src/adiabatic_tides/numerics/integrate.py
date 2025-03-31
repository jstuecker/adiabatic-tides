import numpy as np
from scipy.integrate import simps, trapezoid
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.special import gamma
from . import utility
from .utility import save_divide

# ===================== Generic Integration Functions ====================== #

def scaled_tanh(t, a, b):
    """"
    Returns x = 0.5*(b+a) + 0.5*(b-a) * tanh(t)
    (Like a tanh function, but reaching a at -inf and b at +inf)
    Avoids cancellation by using different expressions on different intervals
    """

    t1, t2, t3 = t[t < -10], t[(t >= -10) & (t <= 10)], t[t > 10]
    x1 = a + (b-a) / (np.exp(-2*t1) + 1)
    x2 = 0.5*(b+a) + 0.5*(b-a) * np.tanh(t2)
    x3 = b - (b-a) / (np.exp(2*t3) + 1)

    x = np.concatenate([x1, x2, x3], axis=-1)

    return x

def integrate_tanh_a_b(f, a, b, N=100):
    """ Integrates f over the interval (a,b) using a tanh substitution.
    x = 0.5*(b+a) + 0.5*(b-a) * tanh(t)
    dxdt = (2/(b-a)) * (b-x)*(x-a)
    See numerical recipes (4.5.3)
    This is especially useful if there are singularities at a or b
    """
    a,b = np.array(a)[...,np.newaxis], np.array(b)[...,np.newaxis]

    h = np.pi/np.sqrt(2.*N)
    t = (np.arange(N)-N/2.)*h

    x = scaled_tanh(t, a, b)

    with np.errstate(divide='ignore', invalid='ignore'):
        dxdt = np.nan_to_num((2./(b-a)) * (b-x)*(x-a), 0.)
    
    return np.trapz(f(x)*dxdt, t, axis=-1)

def integrate_exp_tanh_a_b(f, a, b, N=100, tmax=None, c=1.):
    """Like integrate_tanh_a_b, but in log-space"""
    a,b = np.array(a)[...,np.newaxis], np.array(b)[...,np.newaxis]
    loga, logb = np.log(a), np.log(b)

    if tmax is None:
        tmax = np.clip(np.sqrt(N),0,14)
    t = np.linspace(-tmax, tmax, N)
    with np.errstate(under='ignore'):
        q = scaled_tanh(t, loga, logb)
        x = np.exp(q)

    with np.errstate(divide='ignore', invalid='ignore'):
        dqdt = np.nan_to_num((2./(logb-loga)) * (logb-q)*(q-loga), 0.)

    return np.trapz(f(x)*x*dqdt, t, axis=-1)

def integrate_double_exponential_a_b(f, a, b, N=100, tmax=4., c=1.):
    """ Double exponential integration of f(x) from a to b.
    See Numerical Recipes 4.5.2
    Converges exponentially for most functions.
    If the function is singular at a or b, make sure to return
    zero if evaluated exactly at a or b 
    This may happen due to round-off errors.

    tmax : 4 -> dxmin = (b-a) * 1e-24, 5 -> 1e-65
    """
    a,b = np.array(a)[...,np.newaxis], np.array(b)[...,np.newaxis]
    Nhalf = N//2

    h = tmax/Nhalf
    t = np.arange(1,Nhalf+1)*h
    with np.errstate(under='ignore'):
        q = np.exp(-2.*np.sinh(t)*c)

        delta = (b - a) * q / (1 + q)
        dxdt = 2*(b-a)*q/(1+q)**2 * np.cosh(t) * c

        I = np.sum(f(a + delta) * dxdt, axis=-1)
        I += np.sum(f(b - delta) * dxdt, axis=-1)
        I += (f(0.5*(a+b)) * ((b-a)* c/2))[...,0]
        I = I * h
        
    return I

def integrate_exp_double_exp_a_b(f, a, b, N=100, tmax=4., c=1.):
    """Like integrate_tanh_a_b, but in log-space"""
    loga, logb = np.log(a), np.log(b)

    def newf(t):
        x = np.exp(t)
        return f(x)*x
    
    return integrate_double_exponential_a_b(newf, loga, logb, N, tmax, c)

def integrate_exp_a_inf(f, a=0., N=100, xscale=1.):
    """ Integrates f over the interval (a,inf) using a log/exp substitution.
    x = exp(t)
    dxdt = x
    """
    a,xscale = np.array(a)[...,np.newaxis], np.array(xscale)[...,np.newaxis]

    h = np.pi/np.sqrt(2.*N)
    t = (np.arange(N)-N/2.)*h
    x = np.exp(t)*xscale
    dxdt = x
    
    return np.trapz(f(x+a)*dxdt, t, axis=-1)

def integrate_exp_a_b(f, a=1e-3, b=1., N=100):
    """ Integrates f over the interval (a>0,b) using a log/exp substitution.
    x = exp(t)
    dxdt = x
    """
    t = np.linspace(np.log(a), np.log(b), N, axis=-1)
    x = np.exp(t)
    dxdt = x
    
    return np.trapz(f(x)*dxdt, t, axis=-1)

def integrate_double_exponential_a_inf(f, a=0, N=100, tmax=4., c=1., xscale=1.):
    """ Double exponential integration of f(x) from a to infinity
    See Numerical Recipes 4.5.3
    
    x = exp(2 c sinh(t)) + a
    dxdt = 2c exp(2c sinh(t)) cosh(t)

    tmax : 4 -> xmax ~ 1e23, 5 -> xmax ~ 1e64 
    """
    a,xscale = np.array(a)[...,np.newaxis], np.array(xscale)[...,np.newaxis]

    t = np.linspace(-tmax, tmax, N)

    q = np.exp(2*c*np.sinh(t))
    dxdt = 2*c*q*np.cosh(t)*xscale

    return np.sum(f(q*xscale + a) * dxdt, axis=-1) * (t[1]-t[0])

def integrate_double_exponential_a_infb(f, a=0, b=1, N=100, tmax=4., c=1., xscale=1.):
    """ Double exponential integration of f(x) from a to b which lies almost at infinity
    this is different to integrate_double_exponential_a_b in that it does not place
    many points near b. This only makes sense if f is 0 beyond b!
    This function is almost equivalent to integrate_double_exponential_a_inf, but places 
    no points beyond b
    See Numerical Recipes 4.5.3
    
    x = exp(2 c sinh(t)) + a
    dxdt = 2c exp(2c sinh(t)) cosh(t)

    tmax : 4 -> xmax ~ 1e23, 5 -> xmax ~ 1e64 
    """
    a,b,xscale = np.array(a), np.array(b), np.array(xscale)

    with np.errstate(divide='ignore'):
        tmaxup = np.nan_to_num(np.arcsinh(np.log((b - a)/xscale) / (2.*c)), tmax)

    t = np.linspace(-tmax, np.clip(tmaxup, -tmax+0.1, tmax), N, axis=-1)

    q = np.exp(2*c*np.sinh(t))
    dxdt = 2*c*q*np.cosh(t)*xscale[...,np.newaxis]

    return np.sum(f(q*xscale[...,np.newaxis] + a[...,np.newaxis]) * dxdt, axis=-1) * (t[...,1]-t[...,0])

def integrate_double_exponential_inf_inf(f, N=100, tmax=4.5, c=1., xscale=1.):
    """ Double exponential integration of f(x) from -inf to inf
    See Numerical Recipes 4.5.3
    
    x = sinh(c sinh(t))
    dxdt = c cosh(t) cosh(c sinh(t))

    tmax : 4 -> xmax ~ 1e11, 5 -> xmax ~ 1e31 
    """
    xscale = np.array(xscale)[...,np.newaxis]

    t = np.linspace(-tmax, tmax, N)

    u = np.sinh(t)
    x = np.sinh(c*u)*xscale
    dxdt = c * np.cosh(t) * np.cosh(c*u)*xscale

    return np.sum(f(x) * dxdt, axis=-1) * (t[1]-t[0])

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

# =========== Methods for Inverting density to phase space  ================ #

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

    d2rhodphi2 = utility.second_deriv_avoid_cancelation(rho, phi)

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

    d2rhodphi2 = utility.second_deriv_avoid_cancelation(rho, phi)

    spl_d2rhodphi2 = CubicSpline(phi, d2rhodphi2)

    def integrand(v):
        E = prof.potential(ri)[:,np.newaxis] + 0.5*v**2
        return spl_d2rhodphi2(E) * (E <= phi[-1])
    
    vscale = prof.vcirc(ri)
    I = integrate_exp_a_inf(integrand, N=nintegrate, xscale=vscale)

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

def anisotropic_inversion_old(ri, rho, phi=None, beta=0.):
    """Assuming a profile with constant anisotropy beta, calculates f1(E)
    assuming that f(E,L) = f1(E) * L**(-2beta)
    
    phi : potential -- if not provided a simple Poisson solver is used
          assuming that the distribution rho generates the potential
    """
    if phi is None:
        m, phi = solve_poisson(ri, rho)

    rho_rbeta2 = rho * ri**(2*beta)

    d2rb2 = utility.second_deriv_avoid_cancelation(rho_rbeta2, phi)

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

def anisotropic_inversion(ri, rho, phi, beta=0., spline_class=PchipInterpolator, nintegrate=100):
    """Assuming a profile with constant anisotropy beta, calculates f1(E)
    assuming that f(E,L) = f1(E) * L**(-2beta)
    """
    if np.min(phi) <= 0:
        raise ValueError("For Eddington inversion potential should approach 0 at 0 and be negative nowhere")
    
    assert (beta >= -0.5) and (beta <= 0.5), f"Anisotropy of beta = {beta:.2f}. Valid range is (-0.5,0.5)"

    rho_rbeta2 = rho * ri**(2*beta)
    Ei = phi

    d2rb2 = utility.second_deriv_avoid_cancelation(rho_rbeta2, Ei)
    ip_d2rb2 = spline_class(Ei, d2rb2)

    def integrand(phi):
        return save_divide(ip_d2rb2(phi), np.clip(phi - Ei[...,np.newaxis], 0, None)**(0.5 - beta))

    f = integrate_exp_double_exp_a_b(integrand, Ei,  phi[-1], N=nintegrate, tmax=4.)

    Ibeta = np.sqrt(np.pi) * gamma(1. - beta) / gamma(1.5 - beta)
    fac = 2**(beta - 0.5) * np.cos(beta*np.pi) 
    fac /= 2.*np.pi**2 * Ibeta * (0.5 - beta) #* (0.5 + beta)

    return Ei, f*fac


# ======= Methods for Integrating phase space (e.g. to density)  ========== #

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
    
    return integrate_exp_a_inf(integrand, N=nintegrate, xscale=phiscale)* (np.sqrt(2.)*4.*np.pi)

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
        reval = r * utility.cosh_space(rmaxfac, nintegrate, 2)
        phi, eeval = pot(r), pot(reval)

        with np.errstate(divide='ignore', invalid='ignore'):
            q1 = np.clip(eeval - phi - np.clip(eeval - phip1, 0, None) * (rp1**2 / r**2), 0, None) * (r >= rp1)
            q2 = np.clip(eeval - phi - np.clip(eeval - phip2, 0, None) * (rp2**2 / r**2), 0, None) * (r >= rp2)

            integrand = f_of_e(eeval) * (np.nan_to_num(np.sqrt(q1),0) - np.nan_to_num(np.sqrt(q2),0))

        rho[i] = trapezoid(integrand, eeval) * (np.sqrt(2.)*4.*np.pi)
    
    return rho

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
    
    return integrate_tanh_a_b(integrand, 0., ri, N=nintegrate) * (4.*np.pi)**2 * np.sqrt(2.)

def integrate_fofel_adaptive(f_of_el, phi, r, N=100):
    r = np.array(r)

    Escale = np.clip(phi(r*2.) - phi(r), 0, None)
    def integrate_vl(f_of_el, phi, vr, r, N=100):
        vlscale = np.clip(np.abs(vr), np.sqrt(Escale)[...,np.newaxis], None)
        def integrand(vl):
            E = (phi(r)[...,np.newaxis] + 0.5*vr**2)[...,np.newaxis] + 0.5*vl**2
            L = vl*r[...,np.newaxis,np.newaxis]
            return 2.*np.pi*vl * f_of_el(E, L)
        return integrate_exp_a_inf(integrand, N=N, xscale=vlscale)
    
    def integrand(vr):
        return integrate_vl(f_of_el, phi, vr, r, N=N)
    
    vrscale = np.sqrt(phi(r*2.) - phi(r))
    
    return 2.*integrate_exp_a_inf(integrand, N=N, xscale=vrscale)

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

        return integrate_tanh_a_b(integrand, np.sqrt(vlmin2), np.sqrt(vlmax2),  N=N)
    
    
    def integrand(vr):
        return integrate_vl(f_of_el, phi, vr, r, N=N)
    
    vrscale = np.sqrt(phi(r*2.) - phi(r))

    
    rho[sel] = 2.*integrate_exp_a_inf(integrand, N=N, xscale=vrscale)
    
    return rho

def integrate_f_paspace(f_of_rp_ra, pot, accr, r, N=32, N2=None, rperirange=(0, np.infty), raporange=(0, np.infty), vrmoment=0, vtmoment=0, vmoment=0):
    """Integrates a distribution function, discretizing the integral in "paspace"
    paspace is the space of possible peri- and apocenter radii and maps one to one
    to (E,L) space

    raporange: each component can be a constant or a function depending on rp

    f : function f(rp, ra)
    """
    r = np.array(r)
    phir = pot(r)
    if N2 is None:
        N2 = N

    def integrate_ra_given_rp(rp):
        def integrand(ra):
            with np.errstate(divide='ignore', invalid='ignore'):
                e,l,ldlde = utility.Jacobian_det_ldlde_drpdra(pot, accr, rp[...,np.newaxis], ra, get_el=True)
                vr = np.sqrt(np.clip(2*e - 2*phir[...,np.newaxis,np.newaxis] - l**2/r[...,np.newaxis,np.newaxis]**2, 0, None))

                valid = (ldlde > 0.) & (vr > 0.) & (l > 0.)

            fval = np.zeros_like(e)

            rps, ras = np.broadcast_arrays(rp[...,np.newaxis], ra)
            fval[valid] = f_of_rp_ra(rps[valid], ras[valid])

            if vmoment != 0:
                v = np.sqrt(np.clip(2*e - 2*phir[...,np.newaxis,np.newaxis], 0, None))
                fval *= v**vmoment
            if vrmoment != 0:
                fval *= vr**vrmoment
            if vtmoment != 0:
                fval *= (l/r[...,np.newaxis,np.newaxis])**vtmoment

            return np.divide(fval * ldlde, vr, out=np.zeros_like(fval), where=valid)

        if callable(raporange[0]):
            a = np.clip(raporange[0](rp), r[...,np.newaxis], None)
        else:
            a = np.clip(raporange[0], r, None)[...,np.newaxis]
        if callable(raporange[1]):
            b = np.clip(raporange[1](rp), r[...,np.newaxis], None)
        else:
            b = np.clip(raporange[1], r, None)[...,np.newaxis]

        if np.max(raporange[1]) == np.infty:
            I = integrate_double_exponential_a_inf(integrand, a=a, N=N2,c=1, tmax=4., xscale=a)
        else: # We have a finite upper limit
            I = integrate_exp_double_exp_a_b(integrand, a, b, N=N2)

        return I
    
    a,b = np.clip(rperirange[0], 0, r), np.clip(rperirange[1], 0, r)
    I = integrate_double_exponential_a_b(integrate_ra_given_rp, a, b, N=N, tmax=4)
    return 4.*np.pi*I  / r**2

def integrate_line_of_sight(f, R, Rmax=np.infty, nintegrate=100):
    """Integrates a function along the line of sight with minimal distance R"""
    def integrand(r):
        return 2. * utility.save_divide(r*f(r), np.sqrt(np.clip(r**2 - R[...,np.newaxis]**2, 0, None)))

    if Rmax == np.infty:
        return integrate_double_exponential_a_inf(integrand, R, N=nintegrate)
    else:
        return integrate_double_exponential_a_infb(integrand, R, Rmax, N=nintegrate)

def integrate_line_of_sight_vdisp2_and_dens(density, rho_x_sigmar2, rho_x_sigmat2, R, nintegrate=100):
    """Calculates the line-of-sight velocity dispersion and column density

    rho_x_sigmar2: function that returns rho(r) * sigmar2(r) 
    rho_x_sigmat2: rho(r) * sigmat2(r)
    
    density and sigmar2_sigmat2 are functions of r
    """
    I = integrate_line_of_sight(density, R, nintegrate=nintegrate)
    def integrand(r):
        # All of these are already density weighted
        sigmar2, sigmat2 = rho_x_sigmar2(r), rho_x_sigmat2(r)

        sigmaz2 = sigmar2 + (R[...,np.newaxis]**2/r**2) * (0.5*sigmat2 - sigmar2)
        return sigmaz2
    Iv2 = integrate_line_of_sight(integrand, R, nintegrate=nintegrate)
    return Iv2/I, I

# ================= Methods for calculating Actions  ======================= #

def vr_near_circ(daccdr, r, rp, ra, l):
    """Expansion of vr for nearly circular orbits (avoids cancellation)"""
    rc = 0.5*(rp + ra)
    dphieff_dr2 = 2*daccdr(rc) - 6*l**2/rc**4
    return np.sqrt(-0.5*dphieff_dr2*(r-rp)*(ra-r))

def vr_integral_near_circ(accr, daccdr, rp, ra, p=0.5):
    """integrate vr**(2p) from rp to ra analytically with an expansion for nearly circular orbits"""
    rm = 0.5*(rp + ra)
    l2 = 2.*(ra**2*rp**2)/(ra + rp) * (-accr(rm))
    c = -daccdr(rm) + 3*l2/rm**4

    # To second order vr can be approximated as vr**2 = c*(r-rp)*(ra-r)
    # Then  can integrate vr**2p analytically

    if p == 0.5: # integral over sqrt(vr2) as needed for radial action
        return (1/8. * np.pi)*c**0.5 * (ra - rp)**2
    elif p == -0.5: # integral over 1/sqrt(vr2) as needed for dj/de
        return np.pi/np.sqrt(c)
    else: # for other cases get the pre-factor numerically
        from scipy.special import gamma
        fac = gamma(p+1)**2 / gamma(2*p+2)
        return fac * c**p * (ra - rp)**(2*p+1)

def vr_integral_tanh_peri_apo(pot, rperi, rapo, p=0.5, pr=0., nintegrate=40, accr=None, daccdr=None, eps_circ=1e-3):
    """Calculates an integral over vr**(2p)*r**pr dr from rp to ra.
    
    Provide accr and daccdr to improve accuracy for near circular orbits"""
    I = np.zeros(np.broadcast(rperi, rapo).shape)

    assert np.all(rapo >= rperi), "rapo must be larger than rperi"
    I[rapo < rperi] = np.nan

    if (accr is not None) and (daccdr is not None): # Use expansion for nearly circular orbits
        sel_circ = (rapo > rperi) & (rapo <= rperi*(1+eps_circ))
        I[sel_circ] = vr_integral_near_circ(accr, daccdr, rperi[sel_circ], rapo[sel_circ], p=p)
        if pr != 0.: 
            I[sel_circ] *= (0.5*(rperi[sel_circ]+rapo[sel_circ]))**pr
        sel = (rapo > rperi*(1+eps_circ))
    else:
        sel = rapo > rperi

    rperi, rapo = rperi[sel], rapo[sel]
    e, l = utility.e_l_of_rp_ra(pot, rperi, rapo, accr=accr, eps_circ=eps_circ)

    def integrand(r):
        vr2 = 2*(e[...,np.newaxis] - pot(r)) - l[...,np.newaxis]**2/r**2
        with np.errstate(divide='ignore', invalid='ignore'):
            if pr == 0.:
                return np.nan_to_num(vr2**p, 0) # Roundoff errors may lead to vr2 <= 0
            else:
                return np.nan_to_num(vr2**p * r**pr, 0)

    I[sel] = integrate_tanh_a_b(integrand, rperi, rapo, nintegrate)
    return I

def calculate_radial_action_tanh_peri_apo(pot, rperi, rapo, nintegrate=40, accr=None, daccdr=None, eps_circ=1e-3):
    """Calculate radial action. Provide accr and daccdr to improve accuracy for near circular orbits"""
    return vr_integral_tanh_peri_apo(pot, rperi, rapo, p=0.5, nintegrate=nintegrate, accr=accr, daccdr=daccdr, eps_circ=eps_circ) / np.pi

def calculate_dj_de_tanh_peri_apo(pot, rperi, rapo, nintegrate=40, accr=None, daccdr=None, eps_circ=1e-3):
    return vr_integral_tanh_peri_apo(pot, rperi, rapo, p=-0.5, nintegrate=nintegrate, accr=accr, daccdr=daccdr, eps_circ=eps_circ) / np.pi

def calculate_dj_dl2_tanh_peri_apo(pot, rperi, rapo, nintegrate=40, accr=None, daccdr=None, eps_circ=1e-3):
    return vr_integral_tanh_peri_apo(pot, rperi, rapo, p=-0.5, pr=-2., nintegrate=nintegrate, accr=accr, daccdr=daccdr, eps_circ=eps_circ) / (-2.*np.pi)

def calculate_jel_and_dj_dl_drp_dra(pot, accr, rperi, rapo, nintegrate=40, daccdr=None, eps_circ=1e-3):
    j = calculate_radial_action_tanh_peri_apo(pot, rperi, rapo, nintegrate, accr=accr, daccdr=daccdr, eps_circ=eps_circ)
    dj_de = calculate_dj_de_tanh_peri_apo(pot, rperi, rapo, nintegrate, accr=accr, daccdr=daccdr, eps_circ=eps_circ)
    dj_dl2 = calculate_dj_dl2_tanh_peri_apo(pot, rperi, rapo, nintegrate, accr=accr, daccdr=daccdr, eps_circ=eps_circ)

    de_drp, de_dra, dl2_drp, dl2_dra = utility.dedl2_drpdra(pot, accr, rperi, rapo)

    dj_drp = dj_de * de_drp + dj_dl2 * dl2_drp
    dj_dra = dj_de * de_dra + dj_dl2 * dl2_dra

    e,l = utility.e_l_of_rp_ra(pot, rperi, rapo, accr=accr)

    return j,e,l,dj_drp, dj_dra, dl2_drp/(2.*l), dl2_dra/(2.*l)

# ============= Methods for Solving Poisson's equation  ==================== #

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
        rhoc, alpha = utility.fit_powerlaw(ri[0], ri[1], rho[0], rho[1])
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

def solve_poisson_via_spline(ri, rhoi, spline_class=PchipInterpolator, mbelow=0., phibelow=0., G=43.0071057317063e-10, **kwargs):
    """ Fits a spline to the density profile and evaluates the parent function through the spline class
    spline_class: has to be a interpolator as in scipy.interpolate -- needs to define the .antiderivative() method
    recommended for robustness is scipy.interpolate.PchipInterpolator (3rd order, preserves monotonicity)
    you might improve convergence, e.g. with something like scipy.interpolate.UnivariateSpline, k=5,
    but I do not recommend this as it may fail catastrophically for some cases
    """
    spl_dm_dr = spline_class(ri, 4*np.pi*rhoi*ri**2, **kwargs)
    spl_m = spl_dm_dr.antiderivative()

    spl_dphdir = spline_class(ri, G * (spl_m(ri)+mbelow) / ri**2, **kwargs)
    spl_phi = spl_dphdir.antiderivative()
    
    # For the density it is much better to interpolate rho*r**2 than rho
    # since this removes effectively any singularities in the function
    rho = lambda r: spl_dm_dr(r) / (4*np.pi*r**2)
    m = lambda r: spl_m(r) + mbelow
    phi = lambda r: spl_phi(r) + phibelow

    return rho, m, phi

def describe_lower_boundary(ri, rhoi, boundary="powerlaw", G=43.0071057317063e-10):
    if ri[0] == 0.:
        return lambda r: 0.*r, lambda r: 0.*r, lambda r: 0.*r

    if isinstance(boundary, (list, tuple)) and len(boundary) == 3 and all(callable(f) for f in boundary):
        rho, m, phi = boundary
        return rho, m, phi
    elif boundary == "powerlaw":
        rhoc, alpha = utility.fit_powerlaw(ri[0], ri[1], rhoi[0], rhoi[1])
        rho = lambda r: rhoc * r**alpha
        m = lambda r: 4.*np.pi * rhoc  / (3. + alpha) * r**(3.+alpha)
        phi = lambda r: 4.*np.pi * G * rhoc / ( (3. + alpha) * (2. + alpha) ) * r**(2.+alpha)
    elif boundary =="zero":
        rho = lambda r: 0.*r
        m = lambda r: 0.*r
        phi = lambda r: 0.*r
    elif boundary =="constant":
        rho = lambda r: rhoi[0] + 0.*r
        m = lambda r: 4*np.pi/3. * np.clip(r, 0., ri[0])**3
        phi = lambda r: G * m(r) / r
    else:
        raise ValueError("Unknown lower boundary mode")

    return rho, m, phi

def describe_upper_boundary(rho, m, phi, ri, mode="exp", G=43.0071057317063e-10):
    rmax = ri[-1]
    rhomax, mmax, phimax = rho(rmax),m(rmax), phi(rmax)
    if mode =="vacuum":
        def rho(r): return 0.*r
        def m(r): return mmax + 0.*r
        def phi(r): return phimax + G * mmax * (1./np.clip(r, rmax, None) - 1./np.clip(r, rmax, None))
    elif mode =="exp":
        def rho(r): return rhomax*np.exp((-r + rmax)/rmax)
        def m(r): return mmax + 20*np.pi*rhomax*rmax**3 + (-4*np.pi*r**2*rhomax*rmax - 8*np.pi*r*rhomax*rmax**2 - 8*np.pi*rhomax*rmax**3)*np.exp((-r + rmax)/rmax)
        def phi(r): return phimax + G *(12*np.pi*rhomax*rmax**2 + (mmax + 20*np.pi*rhomax*rmax**3)/rmax - (mmax + 20*np.pi*rhomax*rmax**3)/r + (4*np.pi*r*rhomax*rmax**2 + 8*np.pi*rhomax*rmax**3)*np.exp((-r + rmax)/rmax)/r)
        # def rho(r): return rhomax * np.exp(-np.clip(r,rmax, None)/rmax)
        # def m(r): return mmax + 0.*r
    else:
        raise ValueError("Unknown upper boundary mode")

    return rho, m, phi

def solve_poisson_via_spline_with_smart_boundaries(ri, rhoi, spline_class=PchipInterpolator, lower_boundary="powerlaw", upper_boundary="exp", G=43.0071057317063e-10, **kwargs):
    """
    Solves the Poisson equation with splines and assuming smart boundary conditions
    returns functions rho, m, phi that implement the smart boundary conditions
    lower_boundary : can be "powerlaw", "zero", "constant" or a tuple of 3 functions rho,m,phi
    upper_boundary : so far, can be "vacuum" or "exp" (recommended)
    """
    rhobelow, mbelow, phibelow = describe_lower_boundary(ri, rhoi, boundary=lower_boundary, G=G)
    spl_rho, spl_m, spl_phi = solve_poisson_via_spline(ri, rhoi, spline_class=spline_class, mbelow=mbelow(ri[0]), phibelow=phibelow(ri[0]), G=G, **kwargs)
    rhoabove, mabove, phiabove = describe_upper_boundary(spl_rho, spl_m, spl_phi, ri, mode=upper_boundary, G=G)

    rho = lambda r: np.where(r<ri[0], rhobelow(r), np.where(r>ri[-1], rhoabove(r), spl_rho(r)))
    m = lambda r: np.where(r<ri[0], mbelow(r), np.where(r>ri[-1], mabove(r), spl_m(r)))
    phi = lambda r: np.where(r<ri[0], phibelow(r), np.where(r>ri[-1], phiabove(r), spl_phi(r)))

    return rho,m,phi
