import numpy as np


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