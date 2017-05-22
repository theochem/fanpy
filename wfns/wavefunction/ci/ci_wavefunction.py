"""Parent class of CI wavefunctions.

This module includes wavefunctions that are expressed as linear combination of Slater determinants.
"""
from __future__ import absolute_import, division, print_function
import itertools
import numpy as np
from ..base_wavefunction import BaseWavefunction
from ...backend import slater
from ...backend.sd_list import sd_list

__all__ = []


class CIWavefunction(BaseWavefunction):
    """Wavefunction that can be expressed as a linear combination of Slater determinants.

    Attributes
    ----------
    nelec : int
        Number of electrons
    nspin : int
        Number of spin orbitals (alpha and beta)
    dtype : {np.float64, np.complex128}
        Data type of the wavefunction
    params : np.ndarray
        Parameters of the wavefunction
    _spin : float
        Total spin of each Slater determinant
        :math:`\frac{1}{2}(N_\alpha - N_\beta)`
        Default is no spin (all spins possible)
    _seniority : int
        Number of unpaired electrons in each Slater determinant
    sd_vec : tuple of int
        List of Slater determinants used to construct the CI wavefunction
    dict_sd_index : dictionary of int to int
        Dictionary from Slater determinant to its index in sd_vec

    Properties
    ----------
    nspatial : int
        Number of spatial orbitals
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
    nparams : int
        Number of parameters
    params_shape : 2-tuple of int
        Shape of the parameters

    Methods
    -------
    __init__(self, nelec, one_int, two_int, dtype=None)
        Initializes wavefunction
    assign_nelec(self, nelec)
        Assigns the number of electrons
    assign_nspin(self, nspin)
        Assigns the number of spin orbitals
    assign_params(self, params)
        Assigns the parameters of the wavefunction
    assign_dtype(self, dtype)
        Assigns the data type of parameters used to define the wavefunction
    assign_spin(self, spin=None)
        Assigns the spin of the wavefunction
    assign_seniority(self, seniority=None)
        Assigns the seniority of the wavefunction
    assign_sd_vec(self, sd_vec=None)
        Assigns the tuple of Slater determinants used in the CI wavefunction
    get_overlap(self, sd, deriv=None)
        Gets the overlap from cache and compute if not in cache
        Default is no derivatization
    """
    def __init__(self, nelec, nspin, dtype=None, params=None, sd_vec=None, spin=None,
                 seniority=None):
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
        params : np.ndarray
            Coefficients of the Slater determinants of a CI wavefunction
        sd_vec : iterable of int
            List of Slater determinants used to construct the CI wavefunction
        spin : float
            Total spin of the wavefunction
            Default is no spin (all spins possible)
            0 is singlet, 0.5 and -0.5 are doublets, 1 and -1 are triplets, etc
            Positive spin means that there are more alpha orbitals than beta orbitals
            Negative spin means that there are more beta orbitals than alpha orbitals
        seniority : int
            Seniority of the wavefunction
            Default is no seniority (all seniority possible)
        """
        super().__init__(nelec, nspin, dtype=dtype)
        self.assign_spin(spin=spin)
        self.assign_seniority(seniority=seniority)
        self.assign_sd_vec(sd_vec=sd_vec)
        # FIXME: atleast doubling memory for faster lookup of sd coefficient
        self.dict_sd_index = {sd: i for i, sd in enumerate(self.sd_vec)}
        self.assign_params(params=params)

    @property
    def template_params(self):
        """Return the template of the parameters of the CI wavefunction.

        First Slater determinant of `sd_vec` is used as the reference.

        Returns
        -------
        np.ndarray

        Note
        ----
        Instance must contain `sd_vec`
        """
        params = np.zeros(len(self.sd_vec), dtype=self.dtype)
        params[0] = 1
        return params

    @property
    def spin(self):
        """Return the spin of the wavefunction.

        ..math::
            \frac{1}{2}(N_\alpha - N_\beta)

        Returns
        -------
        float

        Note
        ----
        `None` means that all possible spins are allowed
        """
        return self._spin

    def assign_spin(self, spin=None):
        """Set the spin of each Slater determinant.

        :math:`\frac{1}{2}(N_\alpha - N_\beta)`

        Parameters
        ----------
        spin : float
            Spin of each Slater determinant
            Default is no spin (all spins possible)

        Raises
        ------
        TypeError
            If the spin is not an integer, float, or None
        ValueError
            If the spin is not an integral multiple of 0.5
        """
        if spin is None:
            self._spin = spin
        elif isinstance(spin, (int, float)):
            if (2*spin) % 1 != 0:
                raise ValueError('Spin should be an integral multiple of 0.5.')
            self._spin = float(spin)
        else:
            raise TypeError('Spin should be provided as an integer, float or `None`.')

    @property
    def seniority(self):
        """Return the seniority (number of unpaired electrons) of the wavefunction.

        Returns
        -------
        int

        Note
        ----
        `None` means that all possible seniority are allowed
        """
        return self._seniority

    def assign_seniority(self, seniority=None):
        """Set the seniority of each Slater determinant.

        :math:`\frac{1}{2}(N_\alpha - N_\beta)`

        Parameters
        ----------
        seniority : float
            Seniority of each Slater determinant
            Default is no seniority (all seniorities possible)

        Raises
        ------
        TypeError
            If the seniority is not an integer, float, or None
        ValueError
            If the seniority is a negative integer
        """
        if not (seniority is None or isinstance(seniority, int)):
            raise TypeError('Invalid seniority of the wavefunction')
        elif isinstance(seniority, int) and seniority < 0:
            raise ValueError('Seniority must be a nonnegative integer.')
        self._seniority = seniority

    def assign_sd_vec(self, sd_vec=None):
        """Set the list of Slater determinants from which the CI wavefunction is constructed.

        Parameters
        ----------
        sd_vec : iterable of int
            List of Slater determinants (in the form of integers that describe the occupation as a
            bitstring)

        Raises
        ------
        TypeError
            If sd_vec is not iterable
            If a Slater determinant cannot be turned into the internal form
        ValueError
            If an empty iterator was provided
            If a Slater determinant does not have the correct number of electrons
            If a Slater determinant does not have the correct spin
            If a Slater determinant does not have the correct seniority

        Note
        ----
        Needs to have `nelec`, `nspin`, `spin`, `seniority`
        """
        # FIXME: terrible memory usage
        # FIXME: no check for repeated entries
        if sd_vec is None:
            sd_vec = sd_list(self.nelec, self.nspatial, num_limit=None, exc_orders=None,
                             spin=self.spin, seniority=self.seniority)

        if not hasattr(sd_vec, '__iter__'):
            raise TypeError("Slater determinants must be given as an iterable")

        sd_vec, temp = itertools.tee(sd_vec, 2)
        sd_vec_is_empty = True
        for sd in temp:
            sd_vec_is_empty = False
            sd = slater.internal_sd(sd)
            if slater.total_occ(sd) != self.nelec:
                raise ValueError('Slater determinant, {0}, does not have the correct number of '
                                 'electrons, {1}'.format(bin(sd), self.nelec))
            elif isinstance(self.spin, float) and slater.get_spin(sd, self.nspatial) != self.spin:
                raise ValueError('Slater determinant, {0}, does not have the correct spin, {1}'
                                 ''.format(bin(sd), self.spin))
            elif (isinstance(self.seniority, int)
                  and slater.get_seniority(sd, self.nspatial) != self.seniority):
                raise ValueError('Slater determinant, {0}, does not have the correct seniority, {1}'
                                 ''.format(bin(sd), self.seniority))
        if sd_vec_is_empty:
            raise ValueError('No Slater determinants were provided.')

        self.sd_vec = tuple(sd_vec)

    def get_overlap(self, sd, deriv=None):
        """Return the overlap of the CI wavefunction with a Slater determinant.

        i.e. the Slater determinnt coefficient
        ..math::
            \big< \Phi_i \big| \Psi \big> = c_i

        where
        ..math::
            \big| \Psi \big> = \sum_i c_i \big| \Phi_i \big>

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
        sd = slater.internal_sd(sd)
        if deriv is None:
            return self.params[self.dict_sd_index[sd]]
        elif deriv == self.dict_sd_index[sd]:
            return 1
        else:
            return 0
