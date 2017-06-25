# FIXME: Need consistent notaiton for orbital rotation
"""Wavefunction with orbitals rotated by Jacobi matrix.

A wavefunction constructed from nonorthonormal orbitals can be written as

..math::
    \ket{\Psi}
    &= \sum_{\mathbf{n}} \sum_{\mathbf{m}}
    f(\mathbf{n}) |C(\mathbf{n}, \mathbf{m})|^- \ket{\mathbf{m}}

where :math:`\ket{\mathbf{m}}` and :math:`\ket{\mathbf{n}}` are Slater determinants constructed from
orthonormal and nonorthonormal orbitals.
The :math:`C(\mathbf{n}, \mathbf{m})` is a submatrix of the transformation matrix :math:`C` where
rows are selected according to :math:`\ket{\mathbf{n}}` and columns to :math:`\ket{\mathbf{m}}`.

If the orbitals are transformed with a Jacobi rotation, then many of the determinants are
simplified.

..math::
    \braket{\mathbf{m} | J^\dagger_{pq} | \Psi}
    &= f(\mathbf{m}) \mbox{ if $p \not\in \mathbf{m}$ and $q \not\in \mathbf{m}$}\\
    &= f(\mathbf{m}) \mbox{ if $p \in \mathbf{m}$ and $q \in \mathbf{m}$}\\
    &= f(\mathbf{m}) (\cos\theta + \sin\theta)
    \mbox{ if $p \in \mathbf{m}$ and $q \not\in \mathbf{m}$}\\
    &= f(\mathbf{m}) (\cos\theta - \sin\theta)
    \mbox{ if $p \not\in \mathbf{m}$ and $q \in \mathbf{m}$}\\

where :math:`J^\dagger_{pq}` is the orbital rotation operator that mixes the orbitals that
corresponds to orbitals :math:`p` and  :math:`q`
"""
from __future__ import absolute_import, division, print_function
import functools
import numpy as np
from ...backend import slater
from ..base_wavefunction import BaseWavefunction
from .nonorth_wavefunction import NonorthWavefunction

__all__ = []


# FIXME: needs refactoring
class JacobiWavefunction(BaseWavefunction):
    """Wavefunction with jacobi rotated orbitals expressed with respect to orthonormal orbitals.

    Rotated orbitals are expressed with respect to orthonormal orbitals.

    Attributes
    ----------
    nelec : int
        Number of electrons
    dtype : {np.float64, np.complex128}
        Data type of the wavefunction
    memory : float
        Memory available for the wavefunction
    nspin : int
        Number of orthonormal spin orbitals (alpha and beta)

    Properties
    ----------
    nparams : int
        Number of parameters
    params_shape : 2-tuple of int
        Shape of the parameters
    nspatial : int
        Number of orthonormal spatial orbitals
    spin : float, None
        Spin of the wavefunction
        :math:`\frac{1}{2}(N_\alpha - N_\beta)` (Note that spin can be negative)
        None means that all spins are allowed
    seniority : int, None
        Seniority (number of unpaired electrons) of the wavefunction
        None means that all seniority is allowed
    template_params : np.ndarray
        Template of the wavefunction parameters
        Depends on the attributes given

    Methods
    -------
    __init__(self, nelec, nspin, dtype=None, memory=None)
        Initializes wavefunction
    assign_nelec(self, nelec)
        Assigns the number of electrons
    assign_nspin(self, nspin)
        Assigns the number of spin orbitals
    assign_dtype(self, dtype)
        Assigns the data type of parameters used to define the wavefunction
    assign_memory(self, memory=None)
        Assigns the memory allocated for the wavefunction
    assign_params(self, params)
        Assigns the parameters of the wavefunction
    get_overlap(self, sd, deriv=None)
        Gets the overlap from cache and compute if not in cache
        Default is no derivatization
    """
    def __init__(self, nelec, nspin, dtype=None, memory=None, wfn=None, orbtype=None,
                 jacobi_indices=None):
        """Initialize the wavefunction.

        Parameters
        ----------
        nelec : int
            Number of electrons
        nspin : int
            Number of spin orbitals
        dtype : {float, complex, np.float64, np.complex128, None}
            Numpy data type
            Default is `np.float64`
        memory : {float, int, str, None}
            Memory available for the wavefunction
            Default does not limit memory usage (i.e. infinite)
        wfn : BaseWavefunction
            Wavefunction that will be built up using nonorthnormal orbitals
        orbtype : str
            Type of orbital used by the wavefunction
            One of 'restricted', 'unrestricted', and 'generalized'
        jacobi_indices : tuple of ints
            2-tuple or list of indices of the orbitals that will be rotated
        """
        super().__init__(nelec, nspin, dtype=dtype, memory=memory)
        self.assign_wfn(wfn)
        self.assign_orbtype(orbtype)
        self.assign_jacobi_indices(jacobi_indices)

    # FIXME: copied from NonorthWavefunction
    @property
    def spin(self):
        """Spin of the wavefunction.

        Since the orbitals may mix regardless of the spin, the spin of the wavefunction is hard to
        determine.

        Returns
        -------
        spin
            Spin of the (composite) wavefunction if the orbitals are restricted or unrestricted
            None if the orbital is generalized
        """
        return NonorthWavefunction.spin.__get__(self)

    # FIXME: copied from NonorthWavefunction
    @property
    def seniority(self):
        """Seniority of the wavefunction."""
        return NonorthWavefunction.seniority.__get__(self)

    @property
    def template_params(self):
        """Return the shape of the wavefunction parameters.

        The orbital transformations are the wavefunction parameters.
        """
        return np.array(0.0)

    def assign_orbtype(self, orbtype=None):
        """Assign the orbital type of the orbital rotation.

        Parameters
        ----------
        orbtype : str
            Type of the orbital that are rotated
            One of 'restricted', 'unrestricted', and 'generalized'
        """
        if orbtype is None:
            orbtype = 'restricted'

        if orbtype not in ['restricted', 'unrestricted', 'generalized']:
            raise ValueError("Orbital type must be one of 'restricted', 'unrestricted', and "
                             "'generalized'.")
        self.orbtype = orbtype

    def assign_jacobi_indices(self, jacobi_indices):
        """Assign the indices of the orbitals associated with the Jacobi rotation.

        Parameters
        ----------
        jacobi_indices : tuple/list of ints
            2-tuple/list of indices of the orbitals that will be rotated

        Raises
        ------
        TypeError
            If `jacobi_indices` is not a tuple or list
            If `jacobi_indices` does not have two elements
            If `jacobi_indices` must only contain integers
        ValueError
            If the indices are negative.
            If the two indices are the same
            If orbitals are generalized and indices are greater than `nspin`
            If orbitals are restricted and indices are greater than `nspatial`
            If orbitals are unrestricted and indices mix alpha and beta orbitals.

        Note
        ----
        `orbtype` is used in this method.
        """
        if not isinstance(jacobi_indices, (tuple, list)):
            raise TypeError('Indices `jacobi_indices` must be a tuple or list.')
        elif len(jacobi_indices) != 2:
            raise TypeError('Indices `jacobi_indices` must be have two elements.')
        elif not all(isinstance(i, int) for i in jacobi_indices):
            raise TypeError('Indices `jacobi_indices` must only contain integers.')

        if not all(0 <= i for i in jacobi_indices):
            raise ValueError('Indices cannot be negative.')
        if jacobi_indices[0] == jacobi_indices[1]:
            raise ValueError('Two indices are the same.')
        if self.orbtype == 'generalized' and not all(i < self.nspin for i in jacobi_indices):
            raise ValueError('If the orbitals are generalized, their indices must be less than the '
                             'number of spin orbitals.')
        elif self.orbtype == 'restricted' and not all(i < self.nspatial for i in jacobi_indices):
            raise ValueError('If the orbitals are restricted, their indices must be less than the '
                             'number of spatial orbitals.')
        elif (self.orbtype == 'unrestricted' and
              (jacobi_indices[0] < self.nspatial) != (jacobi_indices[1] < self.nspatial)):
            raise ValueError('If the orbitals are unrestricted, the alpha and beta orbitals cannot '
                             'be mixed.')

        if jacobi_indices[0] > jacobi_indices[1]:
            jacobi_indices = jacobi_indices[::-1]
        self.jacobi_indices = tuple(jacobi_indices)

    # FIXME: copied from NonorthWavefunction
    def assign_wfn(self, wfn=None):
        """Assign the wavefunction.

        Parameters
        ----------
        wfn : BaseWavefunction
            Wavefunction that will be built up using nonorthnormal orbitals

        Raises
        ------
        ValueError
            If the given wavefunction is not an instance of BaseWavefunction
            If the given wavefunction does not have the same number of electrons as the instantiated
            NonorthWavefunction
            If the given wavefunction does not have the same data type as the instantiated
            NonorthWavefunction.
            If the given wavefunction does not have the same memory as the instantiated
            NonorthWavefunction.
        """
        NonorthWavefunction.assign_wfn(self, wfn=wfn)

    # FIXME: there probably is a more elegant way of handling restricted orbital case.
    def get_overlap(self, sd, deriv=None):
        """Return the overlap of the wavefunction with an orthonormal Slater determinant.

        Parameters
        ----------
        sd : int, mpz
            Slater Determinant against which to project.
        deriv : int
            Index of the parameter to derivatize
            Default does not derivatize

        Returns
        -------
        overlap : float

        Raises
        ------
        TypeError
            If given Slater determinant is not compatible with the format used internally
        """
        if deriv is None:
            # if cached function has not been created yet
            if 'overlap' not in self._cache_fns:
                # assign memory allocated to cache
                if self.memory == np.inf:
                    memory = None
                else:
                    memory = int((self.memory - 5*8*self.nparams) / (self.nparams + 1))

                # create function that will be cached
                @functools.lru_cache(maxsize=memory, typed=False)
                def _olp(sd):
                    p, q = self.jacobi_indices
                    theta = self.params
                    if self.orbtype in ['generalized', 'unrestricted']:
                        if slater.occ(sd, p) == slater.occ(sd, q):
                            return self.wfn.get_overlap(sd)
                        elif slater.occ(sd, p) and not slater.occ(sd, q):
                            return (self.wfn.get_overlap(sd) * np.cos(theta) +
                                    self.wfn.get_overlap(slater.excite(sd, p, q)) * np.sin(theta))
                        elif not slater.occ(sd, p) and slater.occ(sd, q):
                            return (self.wfn.get_overlap(sd) * np.cos(theta) -
                                    self.wfn.get_overlap(slater.excite(sd, q, p)) * np.sin(theta))
                    else:
                        alpha_sd, beta_sd = slater.split_spin(sd, self.nspatial)
                        # alpha block contains both p and q or neither p and q
                        # beta block contains both p and q or neither p and q
                        if (slater.occ(alpha_sd, p) == slater.occ(alpha_sd, q) and
                                slater.occ(beta_sd, p) == slater.occ(beta_sd, q)):
                            return self.wfn.get_overlap(sd)
                        # beta block contains p and not q
                        elif (slater.occ(alpha_sd, p) == slater.occ(alpha_sd, q) and
                              slater.occ(beta_sd, p) and not slater.occ(beta_sd, q)):
                            return (self.wfn.get_overlap(sd) * np.cos(theta) +
                                    self.wfn.get_overlap(slater.excite(sd,
                                                                       p + self.nspatial,
                                                                       q + self.nspatial))
                                    * np.sin(theta))
                        # beta block contains q and not p
                        elif (slater.occ(alpha_sd, p) == slater.occ(alpha_sd, q) and
                              not slater.occ(beta_sd, p) and slater.occ(beta_sd, q)):
                            return (self.wfn.get_overlap(sd) * np.cos(theta) -
                                    self.wfn.get_overlap(slater.excite(sd,
                                                                       q + self.nspatial,
                                                                       p + self.nspatial))
                                    * np.sin(theta))
                        # alpha block contains p and not q
                        # beta block contains both p and q or neither p and q
                        elif (slater.occ(alpha_sd, p) and not slater.occ(alpha_sd, q) and
                              slater.occ(beta_sd, p) == slater.occ(beta_sd, q)):
                            return (self.wfn.get_overlap(sd) * np.cos(theta) -
                                    self.wfn.get_overlap(slater.excite(sd, p, q)) * np.sin(theta))
                        # beta block contains p and not q
                        elif (slater.occ(alpha_sd, p) and not slater.occ(alpha_sd, q) and
                              slater.occ(beta_sd, p) and not slater.occ(beta_sd, q)):
                            return (self.wfn.get_overlap(sd) * (np.cos(theta)**2) +
                                    (self.wfn.get_overlap(slater.excite(sd, p, q)) +
                                     self.wfn.get_overlap(slater.excite(sd,
                                                                        p + self.nspatial,
                                                                        q + self.nspatial)))
                                    * np.cos(theta) * np.sin(theta) +
                                    self.wfn.get_overlap(slater.excite(sd,
                                                                       p, p + self.nspatial,
                                                                       q, q + self.nspatial))
                                    * (np.sin(theta)**2))
                        # beta block contains q and not p
                        elif (slater.occ(alpha_sd, p) and not slater.occ(alpha_sd, q) and
                              not slater.occ(beta_sd, p) and slater.occ(beta_sd, q)):
                            return (self.wfn.get_overlap(sd) * (np.cos(theta)**2) +
                                    (self.wfn.get_overlap(slater.excite(sd, p, q)) -
                                     self.wfn.get_overlap(slater.excite(sd,
                                                                        q + self.nspatial,
                                                                        p + self.nspatial)))
                                    * np.cos(theta) * np.sin(theta) -
                                    self.wfn.get_overlap(slater.excite(sd,
                                                                       p, q + self.nspatial,
                                                                       q, p + self.nspatial))
                                    * (np.sin(theta)**2))
                        # alpha block contains q and not p
                        # beta block contains both p and q or neither p and q
                        elif (not slater.occ(alpha_sd, p) and slater.occ(alpha_sd, q) and
                              slater.occ(beta_sd, p) == slater.occ(beta_sd, q)):
                            return (self.wfn.get_overlap(sd) * np.cos(theta) -
                                    self.wfn.get_overlap(slater.excite(sd, q, p)) * np.sin(theta))
                        # beta block contains p and not q
                        elif (not slater.occ(alpha_sd, p) and slater.occ(alpha_sd, q) and
                              slater.occ(beta_sd, p) and not slater.occ(beta_sd, q)):
                            return (self.wfn.get_overlap(sd) * (np.cos(theta)**2) +
                                    (-self.wfn.get_overlap(slater.excite(sd, q, p)) +
                                     self.wfn.get_overlap(slater.excite(sd,
                                                                        p + self.nspatial,
                                                                        q + self.nspatial)))
                                    * np.cos(theta) * np.sin(theta) -
                                    self.wfn.get_overlap(slater.excite(sd,
                                                                       q, p + self.nspatial,
                                                                       p, q + self.nspatial))
                                    * (np.sin(theta)**2))
                        # beta block contains q and not p
                        elif (not slater.occ(alpha_sd, p) and slater.occ(alpha_sd, q) and
                              not slater.occ(beta_sd, p) and slater.occ(beta_sd, q)):
                            return (self.wfn.get_overlap(sd) * (np.cos(theta)**2) +
                                    (-self.wfn.get_overlap(slater.excite(sd, q, p)) -
                                     self.wfn.get_overlap(slater.excite(sd,
                                                                        q + self.nspatial,
                                                                        p + self.nspatial)))
                                    * np.cos(theta) * np.sin(theta) +
                                    self.wfn.get_overlap(slater.excite(sd,
                                                                       q, q + self.nspatial,
                                                                       p, p + self.nspatial))
                                    * (np.sin(theta)**2))

                # store the cached function
                self._cache_fns['overlap'] = _olp

            # if cached function already exists
            else:
                # reload cached function
                _olp = self._cache_fns['overlap']

            return _olp(sd)
        # if derivatization
        elif isinstance(deriv, int):
            # if cached function has not been created yet
            if 'overlap derivative' not in self._cache_fns:
                # assign memory allocated to cache
                if self.memory == np.inf:
                    memory = None
                else:
                    memory = int((self.memory - 5*8*self.nparams)
                                 / (self.nparams + 1) * self.nparams)

                @functools.lru_cache(maxsize=memory, typed=False)
                def _olp_deriv(sd, deriv):
                    p, q = self.jacobi_indices
                    theta = self.params
                    if self.orbtype in ['generalized', 'unrestricted']:
                        if slater.occ(sd, p) == slater.occ(sd, q):
                            return 0.0
                        elif slater.occ(sd, p) and not slater.occ(sd, q):
                            return (-self.wfn.get_overlap(sd, deriv=None) * np.sin(theta)
                                    + self.wfn.get_overlap(slater.excite(sd, p, q), deriv=None)
                                    * np.cos(theta))
                        elif not slater.occ(sd, p) and slater.occ(sd, q):
                            return (-self.wfn.get_overlap(sd, deriv=None) * np.sin(theta)
                                    - self.wfn.get_overlap(slater.excite(sd, q, p), deriv=None)
                                    * np.cos(theta))
                    else:
                        alpha_sd, beta_sd = slater.split_spin(sd, self.nspatial)
                        # alpha block contains both p and q or neither p and q
                        # beta block contains both p and q or neither p and q
                        if (slater.occ(alpha_sd, p) == slater.occ(alpha_sd, q) and
                                slater.occ(beta_sd, p) == slater.occ(beta_sd, q)):
                            return 0.0
                        # beta block contains p and not q
                        elif (slater.occ(alpha_sd, p) == slater.occ(alpha_sd, q) and
                              slater.occ(beta_sd, p) and not slater.occ(beta_sd, q)):
                            return (self.wfn.get_overlap(sd) * (-np.sin(theta)) +
                                    self.wfn.get_overlap(slater.excite(sd,
                                                                       p + self.nspatial,
                                                                       q + self.nspatial))
                                    * np.cos(theta))
                        # beta block contains q and not p
                        elif (slater.occ(alpha_sd, p) == slater.occ(alpha_sd, q) and
                              not slater.occ(beta_sd, p) and slater.occ(beta_sd, q)):
                            return (self.wfn.get_overlap(sd) * (-np.sin(theta)) -
                                    self.wfn.get_overlap(slater.excite(sd,
                                                                       q + self.nspatial,
                                                                       p + self.nspatial))
                                    * np.cos(theta))
                        # alpha block contains p and not q
                        # beta block contains both p and q or neither p and q
                        elif (slater.occ(alpha_sd, p) and not slater.occ(alpha_sd, q) and
                              slater.occ(beta_sd, p) == slater.occ(beta_sd, q)):
                            return (self.wfn.get_overlap(sd) * (-np.sin(theta)) -
                                    self.wfn.get_overlap(slater.excite(sd, p, q)) * np.cos(theta))
                        # beta block contains p and not q
                        elif (slater.occ(alpha_sd, p) and not slater.occ(alpha_sd, q) and
                              slater.occ(beta_sd, p) and not slater.occ(beta_sd, q)):
                            return (self.wfn.get_overlap(sd) * (-2 * np.cos(theta)*np.sin(theta)) +
                                    (self.wfn.get_overlap(slater.excite(sd, p, q)) +
                                     self.wfn.get_overlap(slater.excite(sd,
                                                                        p + self.nspatial,
                                                                        q + self.nspatial)))
                                    * (-np.sin(theta)*np.sin(theta) + np.cos(theta)*np.cos(theta)) +
                                    self.wfn.get_overlap(slater.excite(sd,
                                                                       p, p + self.nspatial,
                                                                       q, q + self.nspatial))
                                    * (2 * np.sin(theta) * np.cos(theta)))
                        # beta block contains q and not p
                        elif (slater.occ(alpha_sd, p) and not slater.occ(alpha_sd, q) and
                              not slater.occ(beta_sd, p) and slater.occ(beta_sd, q)):
                            return (self.wfn.get_overlap(sd) * (-2 * np.cos(theta)*np.sin(theta)) +
                                    (self.wfn.get_overlap(slater.excite(sd, p, q)) -
                                     self.wfn.get_overlap(slater.excite(sd,
                                                                        q + self.nspatial,
                                                                        p + self.nspatial)))
                                    * (-np.sin(theta)*np.sin(theta) + np.cos(theta)*np.cos(theta)) -
                                    self.wfn.get_overlap(slater.excite(sd,
                                                                       p, q + self.nspatial,
                                                                       q, p + self.nspatial))
                                    * (2 * np.sin(theta) * np.cos(theta)))
                        # alpha block contains q and not p
                        # beta block contains both p and q or neither p and q
                        elif (not slater.occ(alpha_sd, p) and slater.occ(alpha_sd, q) and
                              slater.occ(beta_sd, p) == slater.occ(beta_sd, q)):
                            return (self.wfn.get_overlap(sd) * (-np.sin(theta)) -
                                    self.wfn.get_overlap(slater.excite(sd, q, p)) * np.cos(theta))
                        # beta block contains p and not q
                        elif (not slater.occ(alpha_sd, p) and slater.occ(alpha_sd, q) and
                              slater.occ(beta_sd, p) and not slater.occ(beta_sd, q)):
                            return (self.wfn.get_overlap(sd) * (-2 * np.cos(theta)*np.sin(theta)) +
                                    (-self.wfn.get_overlap(slater.excite(sd, q, p)) +
                                     self.wfn.get_overlap(slater.excite(sd,
                                                                        p + self.nspatial,
                                                                        q + self.nspatial)))
                                    * (-np.sin(theta)*np.sin(theta) + np.cos(theta)*np.cos(theta)) -
                                    self.wfn.get_overlap(slater.excite(sd,
                                                                       q, p + self.nspatial,
                                                                       p, q + self.nspatial))
                                    * (2 * np.sin(theta) * np.cos(theta)))
                        # beta block contains q and not p
                        elif (not slater.occ(alpha_sd, p) and slater.occ(alpha_sd, q) and
                              not slater.occ(beta_sd, p) and slater.occ(beta_sd, q)):
                            return (self.wfn.get_overlap(sd) * (-2 * np.cos(theta)*np.sin(theta)) +
                                    (-self.wfn.get_overlap(slater.excite(sd, q, p)) -
                                     self.wfn.get_overlap(slater.excite(sd,
                                                                        q + self.nspatial,
                                                                        p + self.nspatial)))
                                    * (-np.sin(theta)*np.sin(theta) + np.cos(theta)*np.cos(theta)) +
                                    self.wfn.get_overlap(slater.excite(sd,
                                                                       q, q + self.nspatial,
                                                                       p, p + self.nspatial))
                                    * (2 * np.sin(theta) * np.cos(theta)))

                # store the cached function
                self._cache_fns['overlap derivative'] = _olp_deriv
            # if cached function already exists
            else:
                # reload cached function
                _olp_deriv = self._cache_fns['overlap derivative']

            return _olp_deriv(sd, deriv)
