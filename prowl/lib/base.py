from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.optimize import least_squares


def __init__(self, n, H, G, dtype=None, p=None, pspace=None, seq=None, x=None):
    """
    Initialize the base wavefunction class.

    Parameters
    ----------
    n : int
        The number of electrons.
    H : 2-index np.ndarray, dtype=np.float64
        The one-electron Hamiltonian array in the MO basis.
    G : 4-index np.ndarray, dtype=np.float64
        The two-electron Hamiltonian array in the MO basis.
    x : 1-index np.ndarray, dtype=np.float64, optional
        The coefficient vector with initial guess values.
        If not specified, an appropriate guess `x` is generated.
    pspace : list, optional
        A list of Python ints representing Slater Determinants.
        If not specified, an appropriate `pspace` is generated.

    """

    # System attributes
    self.n = n
    self.k = H.shape[0]
    self.H = H
    self.G = G

    # Algorithm attributes
    self.p = n // 2 if p is None else p
    if seq:
        self.seq = seq

    # Coefficient attributes
    self.dtype = self.dtype if dtype is None else dtype
    self.x = self.generate_guess() if x is None else x
    self.C = self.generate_view()

    # Projection space attributes
    self.pspace = self.generate_pspace() if pspace is None else pspace
    self.ground = min(self.pspace)


def energy(self, sd=None, dict=False):
    """
    Calculate the energy of the wavefunction.

    Parameters
    ----------
    sd : int, optional
       The Slater Determinant against which to project.
       If not specified, the projection is done against `self.ground`.
    dict : bool, optional
        If True, the energy is returned as a dict of its separate components.
        If False, the energy is returned as a single float.

    """

    sd = self.ground if sd is None else sd
    energies = self.hamiltonian(sd)

    if dict:
        energy = {
            "kinetic": energies[0],
            "coulomb": energies[1],
            "exchange": energies[2],
        }
    else:
        energy = sum(energies)

    return energy


def solve(self, **kwargs):
    """
    Optimize `self.objective(x)` to solve the coefficient vector.

    Parameters
    ----------
    jacobian : bool, optional
        If False, the Jacobian is not used in the optimization.
        If False, it is not used.
    kwargs : dict, optional
        Keywords to pass to the internal solver, `scipy.optimize.leastsq`.

    Returns
    -------
    result : tuple
        See `scipy.optimize.leastsq` documentation.

    """

    # Update solver options
    options = {
        "jac": self.jacobian,
        "bounds": self.bounds,
        "xtol": 1.0e-15,
        "ftol": 1.0e-15,
        "gtol": 1.0e-15,
    }
    options.update(kwargs)

    # Use appropriate Jacobian approximation if necessary
    if not options["jac"]:
        if self.dtype == np.complex128:
            options["jac"] = "cs"
        else:
            options["jac"] = "3-point"

    # Solve
    result = least_squares(self.objective, self.x, **options)

    return result
