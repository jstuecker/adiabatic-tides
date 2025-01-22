import numpy as np


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

    # We have x = 0.5*(b+a) + 0.5*(b-a) * np.tanh(t)
    # but to avoid cancellation (e.g. of tanh - 1),
    # we use different formulations of tanh in different intervals
    t1, t2, t3 = t[t < -10], t[(t >= -10) & (t <= 10)], t[t > 10]
    x1 = a + (b-a) / (np.exp(-2*t1) + 1)
    x2 = 0.5*(b+a) + 0.5*(b-a) * np.tanh(t2)
    x3 = b - (b-a) / (np.exp(2*t3) + 1)
    
    x = np.concatenate([x1, x2, x3], axis=-1)

    dxdt = (2./(b-a)) * (b-x)*(x-a)
    
    return np.trapz(f(x)*dxdt, t, axis=-1)

def integrate_exp_0_inf(f, N=100, xscale=1.):
    """ Integrates f over the interval (a,inf) using a log/exp substitution.
    x = exp(t)
    dxdt = x
    """
    xscale = np.array(xscale)[...,np.newaxis]

    h = np.pi/np.sqrt(2.*N)
    t = (np.arange(N)-N/2.)*h
    x = np.exp(t)*xscale
    dxdt = x
    
    return np.trapz(f(x)*dxdt, t, axis=-1)