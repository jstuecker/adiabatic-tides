import numpy as np
import adiabatic_tides as at

import io
import pytest
import base64
import pytest_html

import parse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

@pytest.fixture
def embed_plot(extras):
    def save_plot(fig):
        # Save the plot to a buffer as a base64 string
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()

        # Attach the plot to the current test item
        #request.node.extra_image = image_base64
        extras.append(pytest_html.extras.png(image_base64))  
        return image_base64

    return save_plot

def standard_profiles(name):
    if name == "nfw": # nfw with rs ~ 1
        return at.profiles.NFWProfile(conc=1., r200c=1.)
    elif name[0:4] == "anfw": # anisotropic nfw
        beta = float(name[4:])
        return at.profiles.NFWProfile(conc=1., r200c=1., anisotropy=beta)
    elif name == "plummer":
        return at.profiles.PlummerProfile()
    elif name == "isotherm":
        return at.profiles.IsothermalSphere()
    elif name[0:8] == "powerlaw":
        slope = float(name[8:])
        print("Powerlaw with slope = %.2f" % slope)
        return at.profiles.PowerlawProfile(alpha=slope, anisotropy=0.)
        # return at.profiles.PowerlawProfile(slope=-slope, rhoc=1.)
    elif name[0:5] == "aniso":
        result = parse.parse("aniso{beta}pow{alpha}", name)
        return at.profiles.PowerlawProfile(alpha=float(result["alpha"]), anisotropy=float(result["beta"]), rhoc=1.)
    else:
        raise ValueError("Unknown profile %s" % name)
    
def standard_profile_radii(name, n = 1000, ntest=231):
    if name=="plummer":
        return np.linspace(0, 20, n), np.linspace(1e-1,10,ntest)
    else:
        return np.logspace(-10, 10, n), np.logspace(-8, 8, ntest)

def check_max_relative_error(f, fref, tolerance=1e-3, fscale=None):
    if fscale is None:
        fscale = fref
    rel_err = np.max(np.abs((f-fref)/fscale))
    print("Maximal relative error is %.2e" % rel_err)
    assert rel_err < tolerance, "error is too large %.2e > %.2e" % (rel_err, tolerance)

def check_perc_relative_error(f, fref, tolerance=1e-3, percentile=90):
    rel_err = np.abs(f/fref - 1.)
    maxerr, perr = np.max(rel_err), np.percentile(rel_err, percentile)
    print("Maximal relative error is %.2e, %.1fth percentile is %.2e" % (maxerr, percentile, perr))
    assert perr < tolerance, "error is too large %.2e > %.2e" % (perr, tolerance)

def plot_relative_error(f, fref, tolerance=1e-3, fscale=None):
    if fscale is None:
        fscale = fref
    rel_err = np.abs((f-fref)/fscale)
    fig = plt.figure()

    fig, axs = plt.subplots(2,1, figsize=(6,5), sharex=True)
    axs[0].semilogy(f)
    axs[0].semilogy(fref, ls="dashed")
    axs[1].semilogy(rel_err)
    axs[1].axhline(tolerance, linestyle="--", color="black")
    axs[1].set_ylim(min(tolerance**3, 1e-3), 1.)
    return fig