from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.optimize import least_squares


def __init__(self, n, H, G, x=None, pspace=None, dtype=None):
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

    self.n = n
    self.k = H.shape[0]
    self.p = int(n // 2)

    self.H = H
    self.G = G

    self.dtype = self.dtype if dtype is None else dtype
    self.x = self.generate_guess() if x is None else x
    self.C = self.generate_view()

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


def generate_guess(self):
    """
    Generate an appropriate random guess for the coefficient vector `x`.

    Raises
    ------
    NotImplementedError

    """

    raise NotImplementedError


def generate_pspace(self):
    """
    Generate an appropriate projection space for solving the coefficient vector `x`.

    Raises
    ------
    NotImplementedError

    """

    raise NotImplementedError


def generate_view(self):
    """
    Generate a view of `x` corresponding to the shape of the coefficient array.

    Raises
    ------
    NotImplementedError

    """

    raise NotImplementedError


def hamiltonian(self, sd):
    """
    Compute the Hamiltonian of the wavefunction projected against `sd`.

    Parameters
    ----------
    sd : int
        The Slater Determinant against which to project.

    Raises
    ------
    NotImplementedError

    """

    raise NotImplementedError


def jacobian(self, x):
    """
    Compute the Jacobian of `self.objective(x)`.

    Parameters
    ----------
    x : 1-index np.ndarray
        The coefficient vector.

    Raises
    ------
    NotImplementedError

    """

    raise NotImplementedError


def objective(self, x):
    """
    Compute the objective function for solving the coefficient vector.
    The function is of the form:
    "<sd|H + G|Psi> - E<sd|Psi> == 0 for all sd in pspace".

    Parameters
    ----------
    x : 1-index np.ndarray
        The coefficient vector.

    Raises
    ------
    NotImplementedError

    """

    raise NotImplementedError


def overlap(self, sd):
    """
    Compute the overlap of the wavefunction with `sd`.

    Parameters
    ----------
    sd : int
        The Slater Determinant against which to project.

    Raises
    ------
    NotImplementedError

    """

    raise NotImplementedError


def solve(self, **kwargs):
    """
    Optimize `self.objective(x)` to solve the coefficient vector.

    Parameters
    ----------
    jacobian : bool, optional
        If True, the Jacobian is used in the optimization.
        If False, it is not used.
    kwargs : dict, optional
        Keywords to pass to the internal solver, `scipy.optimize.leastsq`.

    Returns
    -------
    result : tuple
        See `scipy.optimize.leastsq` documentation.

    """

    options = {
        #"full_output": True,
        "jac": None,
        "bounds": self.bounds,
    }
    options.update(kwargs)

    if options["jac"]:
        options["jac"] = self.jacobian
    elif self.dtype == np.complex128:
        options["jac"] = "cs"
    else:
        options["jac"] = "3-point"
    result = least_squares(self.objective, self.x, **options)

    return result
