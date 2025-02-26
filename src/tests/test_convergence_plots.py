import adiabatic_tides as at
import numpy as np
import pytest
import os

from . import test_config as tc
from .test_config import embed_plot, standard_profiles
from adiabatic_tides.convergence_plots import plot_perisplit_integration, plot_poisson_convergence

# Tests in this module are slow and do not make assertions
# rather they add plots to the report that can be checked manually

basedir = "tests/plots"
def savefig(fig, dir, filename):
    path = os.path.join(basedir, dir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")

@pytest.mark.plot
@pytest.mark.veryslow
@pytest.mark.parametrize("profile", ["nfw", "plummer",  "powerlaw1.0", "powerlaw1.4", "powerlaw1.8", "aniso-0.3pow1", "aniso0.4pow1.4"])
def test_plot_single_perisplit_integration(profile, embed_plot):
    prof = standard_profiles(profile)

    ri = np.logspace(-10,10,2000)
    nprof = at.profiles.NumericalProfile(ri, prof.density(ri))
    fig, axs = plot_perisplit_integration(nprof, norb=20, npart=100000, steps_per_orb=100)
    savefig(fig, "single_perisplit", f"{profile}.pdf")

    embed_plot(fig)

@pytest.mark.plot
@pytest.mark.veryslow
@pytest.mark.parametrize("profile", ["powerlaw1.0", "powerlaw1.4", "powerlaw1.8", "aniso-0.3pow1", "aniso0.2pow1.5", "aniso-0.3pow1.5", "aniso0.1pow1."])
def test_plot_perisplit_integration_pow(profile, embed_plot):
    prof = standard_profiles(profile)
    # ri = np.logspace(-10,10,2000)
    # nprof = at.profiles.NumericalProfile(ri, prof.density(ri))

    from adiabatic_tides.convergence_plots import plot_perimultisplit_integration

    fig, axs, res = plot_perimultisplit_integration(prof, 100000, norb=20, steps_per_orb=200)
    savefig(fig, "perisplit_pow", f"{profile}.pdf")

    embed_plot(fig)

@pytest.mark.plot
@pytest.mark.veryslow
@pytest.mark.parametrize("anisotropy", [-0.48,-0.4,-0.2,0.,0.2,0.4,0.48])
def test_plot_perisplit_integration_aniso_nfw(anisotropy, embed_plot):
    prof = standard_profiles("nfw")
    ri = np.logspace(-10,10,2000)
    nprof = at.profiles.NumericalProfile(ri, prof.density(ri), anisotropy=anisotropy)

    from adiabatic_tides.convergence_plots import plot_perimultisplit_integration

    fig, axs, res = plot_perimultisplit_integration(nprof, 100000, norb=20, steps_per_orb=200)
    savefig(fig, "perisplit_nfw", f"nfw_aniso{anisotropy}.pdf")

    embed_plot(fig)

@pytest.mark.plot
@pytest.mark.parametrize("profile", ["nfw", "plummer",  "powerlaw1.0", "powerlaw1.4", "powerlaw1.8", "aniso0.4pow1.4"])
def test_poisson_convergence(profile, embed_plot):
    prof = standard_profiles(profile)

    fig, axs = plot_poisson_convergence(prof)
    savefig(fig, "poisson", f"{profile}_pchip.pdf")

    embed_plot(fig)