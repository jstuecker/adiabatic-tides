import numpy as np

def finite_differences_n(x, f, deriv=1, h=1e-5, **kwargs):
    """Calculate the nth order derivatives of a function
    
    x : locations where to evaluate the derivatives shape (..., ndim)
    f : function f(x, **kwargs)
    deriv : derivative, if > 1 this function will be called recursively
    h : step-size that will be used for finite differences
    **kwargs: will be passed through to the function
    
    returns : n-th derivative tensor, e.g. for deriv=2 shape = (..., ndim, ndim) 
    """
    if deriv==0:
        return f(x, **kwargs)
    
    grad = np.zeros(np.shape(f(x, **kwargs)) + (3,)*deriv)
    
    dim = x.shape[-1]
    eij = np.diag(np.ones(dim))
    
    for ax in range(0, dim):
        xp = x + h * eij[ax]
        xm = x - h * eij[ax]
        
        fp = finite_differences_n(xp, f, deriv=deriv-1, h=h, **kwargs)
        fm = finite_differences_n(xm, f, deriv=deriv-1, h=h, **kwargs)
        
        grad[...,ax] = (fp - fm) / (2*h)

    return grad

def potential_miyamoto_nagai(x, M=2e10, b=300e-6, a=3000e-6, G=43.0071057317063e-10):
    """Miyamoto Nagai (1975) potential (https://articles.adsabs.harvard.edu/pdf/1975PASJ...27..533M)
    see also http://astro.utoronto.ca/~bovy/AST1420/notes/notebooks/07.-Flattened-Mass-Distributions.html
    
    x : positions to evaluate, shape (...,3)
    
    returns : potential at locations x
    """
    # assuming the disk lies in the x-y plane
    R = np.sqrt(x[...,0]**2 + x[...,1]**2)
    z = x[...,2]
    
    return - G*M / np.sqrt(R**2 + (np.sqrt(z**2 + b**2) + a)**2)

def tidal_tensor_miyamoto_nagai(x, M=2e10, b=300e-6, a=3000e-6, G=43.0071057317063e-10, h=5e-6):
    # use finite differencing, analytic derivative is complicated
    # h = 5 pc seems reasonable
    return -finite_differences_n(x, potential_miyamoto_nagai, deriv=2, G=G, M=M, b=b, a=a, h=h)

def random_vector(length=1., size=1):
    randvec = np.random.normal(0., 1., tuple(np.atleast_1d(size)) + (3,))
    randnorm = np.sqrt(np.sum(randvec**2, axis=-1))
    return randvec * (length/randnorm)[...,np.newaxis]

def tide_percentiles_miyamoto_nagai(r, nsamp=50, percentile=50):
    """Percentiles of tidal eigen value distribution
    
    Takes a set of radii with unknown directions and randomly samples
    directions to create a distribution of eigenvalues for each direction.
    Then returns the precentiles of this distribution
    
    r : radii
    nsamp : number of samples to do per radius
    percentile : percentiles to return
    
    returns : eigenvalue percentiles, shape ((npercentiles), nradii, 3)
    """
    uvec = random_vector(size=nsamp)
    xvec = r[...,np.newaxis,np.newaxis] * uvec
    T = tidal_tensor_miyamoto_nagai(xvec)
    tev = np.sort(np.linalg.eigvalsh(T), axis=-1)[...,::-1]

    return np.percentile(tev, percentile, axis=1)
    