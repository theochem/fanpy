"""Parent class of CI wavefunctions."""
import itertools

import numpy as np
from wfns.backend import slater
from wfns.backend.sd_list import sd_list
from wfns.wfn.base import BaseWavefunction


class CIWavefunction(BaseWavefunction):
    r"""Wavefunction that can be expressed as a linear combination of Slater determinants.

    .. math::

        \left| \Psi \right> &= \sum_i c_i \left| \Phi_i \right>\\
        &= \sum_{\mathbf{m} \in S} c_{\mathbf{m}} \left| \mathbf{m} \right>\\

    where :math:`\Phi_i` is Slater determinants. The :math:`\mathbf{m}` is the occupation vector of
    a Slater determinant (and therefore can be used interchangeably with the Slater determinant) and
    :math:`S` is the set of Slater determinants used to create the wavefunction.

    Attributes
    ----------
    nelec : int
        Number of electrons.
    nspin : int
        Number of spin orbitals (alpha and beta).
    params : np.ndarray
        Parameters of the wavefunction.
    memory : float
        Memory available for the wavefunction.
    _spin : float
        Total spin of each Slater determinant.
        :math:`\frac{1}{2}(N_\alpha - N_\beta)`.
        Default is no spin (all spins possible).
    _seniority : int
        Number of unpaired electrons in each Slater determinant.
    sd_vec : tuple of int
        List of Slater determinants used to construct the CI wavefunction.
    dict_sd_index : dictionary of int to int
        Dictionary from Slater determinant to its index in sd_vec.

    Properties
    ----------
    nparams : int
        Number of parameters.
    nspatial : int
        Number of spatial orbitals
    spin : int
        Spin of the wavefunction
    seniority : int
        Seniority of the wavefunction
    dtype
        Data type of the wavefunction.

    Methods
    -------
    __init__(self, nelec, nspin, memory=None, params=None, sd_vec=None, spin=None,
             seniority=None):
        Initialize the wavefunction.
    assign_nelec(self, nelec)
        Assign the number of electrons.
    assign_nspin(self, nspin)
        Assign the number of spin orbitals.
    assign_memory(self, memory=None):
        Assign memory available for the wavefunction.
    assign_params(self, params)
        Assign parameters of the wavefunction.
    load_cache(self)
        Load the functions whose values will be cached.
    clear_cache(self)
        Clear the cache.
    assign_spin(self, spin=None)
        Assign the spin of the wavefunction.
    assign_seniority(self, seniority=None)
        Assign the seniority of the wavefunction.
    assign_sd_vec(self, sd_vec=None)
        Assign the list of Slater determinants from which the CI wavefunction is constructed.
    get_overlap(self, sd, deriv=None) : float
        Return the overlap of the CI wavefunction with a Slater determinant.

    """

    # pylint:disable=W0223
    def __init__(
        self,
        nelec,
        nspin,
        memory=None,
        params=None,
        sd_vec=None,
        spin=None,
        seniority=None,
    ):
        """Initialize the wavefunction.

        Parameters
        ----------
        nelec : int
            Number of electrons.
        nspin : int
            Number of spin orbitals.
        memory : {float, int, str, None}
            Memory available for the wavefunction.
            Default does not limit memory usage (i.e. infinite).
        params : np.ndarray
            Coefficients of the Slater determinants of a CI wavefunction.
        sd_vec : iterable of int
            List of Slater determinants used to construct the CI wavefunction.
        spin : float
            Total spin of the wavefunction.
            Default is no spin (all spins possible).
            0 is singlet, 0.5 and -0.5 are doublets, 1 and -1 are triplets, etc.
            Positive spin means that there are more alpha orbitals than beta orbitals.
            Negative spin means that there are more beta orbitals than alpha orbitals.
        seniority : int
            Seniority of the wavefunction.
            Default is no seniority (all seniority possible).

        """
        super().__init__(nelec, nspin, memory=memory)
        self.assign_spin(spin=spin)
        self.assign_seniority(seniority=seniority)
        self.assign_sd_vec(sd_vec=sd_vec)
        # FIXME: atleast doubling memory for faster lookup of sd coefficient
        self.dict_sd_index = {sd: i for i, sd in enumerate(self.sd_vec)}
        self.assign_params(params=params)

    @property
    def spin(self):
        r"""Return the spin of the wavefunction.

        .. math::

            \frac{1}{2}(N_\alpha - N_\beta)

        Returns
        -------
        spin : float
            Spin of the wavefunction.

        Notes
        -----
        `None` means that all possible spins are allowed.

        """
        return self._spin

    @property
    def seniority(self):
        """Return the seniority of the wavefunction.

        Seniority of a Slater determinant is its number of unpaired electrons. The seniority of the
        wavefunction is the expected number of unpaired electrons.

        Returns
        -------
        seniority : int
            Seniority of the wavefunction.

        Notes
        -----
        `None` means that all possible seniority are allowed.

        """
        return self._seniority

    @property
    def nsd(self):
        """Return number of Slater determinants.

        Returns
        -------
        nsd : int
            Number of Slater determinants.

        """
        return len(self.sd_vec)

    def assign_spin(self, spin=None):
        r"""Assign the spin of the wavefunction.

        :math:`\frac{1}{2}(N_\alpha - N_\beta)`

        Parameters
        ----------
        spin : float
            Spin of each Slater determinant.
            Default is no spin (all spins possible).

        Raises
        ------
        TypeError
            If the spin is not an integer, float, or None.
        ValueError
            If the spin is not an integral multiple of `0.5`.

        """
        if spin is None:
            self._spin = spin
        elif isinstance(spin, (int, float)):
            if (2 * spin) % 1 != 0:
                raise ValueError("Spin should be an integral multiple of 0.5.")
            self._spin = float(spin)
        else:
            raise TypeError("Spin should be provided as an integer, float or `None`.")

    def assign_seniority(self, seniority=None):
        r"""Assign the seniority of the wavefunction.

        :math:`\frac{1}{2}(N_\alpha - N_\beta)`

        Parameters
        ----------
        seniority : float
            Seniority of each Slater determinant.
            Default is no seniority (all seniorities possible).

        Raises
        ------
        TypeError
            If the seniority is not an integer, float, or None.
        ValueError
            If the seniority is a negative integer.

        """
        if not (seniority is None or isinstance(seniority, int)):
            raise TypeError("Invalid seniority of the wavefunction")
        if isinstance(seniority, int) and seniority < 0:
            raise ValueError("Seniority must be a nonnegative integer.")
        self._seniority = seniority

    def assign_sd_vec(self, sd_vec=None):
        """Assign the list of Slater determinants from which the CI wavefunction is constructed.

        Parameters
        ----------
        sd_vec : iterable of int
            List of Slater determinants.

        Raises
        ------
        TypeError
            If sd_vec is not iterable.
            If a Slater determinant cannot be turned into the internal form.
        ValueError
            If an empty iterator was provided.
            If a Slater determinant does not have the correct number of electrons.
            If a Slater determinant does not have the correct spin.
            If a Slater determinant does not have the correct seniority.

        Notes
        -----
        Needs to have `nelec`, `nspin`, `spin`, `seniority`.

        """
        # pylint: disable=C0103
        # FIXME: terrible memory usage
        # FIXME: no check for repeated entries
        if sd_vec is None:
            sd_vec = sd_list(
                self.nelec,
                self.nspatial,
                num_limit=None,
                exc_orders=None,
                spin=self.spin,
                seniority=self.seniority,
            )

        if not hasattr(sd_vec, "__iter__"):
            raise TypeError("Slater determinants must be given as an iterable")

        sd_vec, temp = itertools.tee(sd_vec, 2)
        sd_vec_is_empty = True
        for sd in temp:
            sd_vec_is_empty = False
            sd = slater.internal_sd(sd)
            if slater.total_occ(sd) != self.nelec:
                raise ValueError(
                    "Slater determinant, {0}, does not have the correct number of "
                    "electrons, {1}".format(bin(sd), self.nelec)
                )
            if isinstance(self.spin, float) and slater.get_spin(sd, self.nspatial) != self.spin:
                raise ValueError(
                    "Slater determinant, {0}, does not have the correct spin, {1}"
                    "".format(bin(sd), self.spin)
                )
            if (
                isinstance(self.seniority, int)
                and slater.get_seniority(sd, self.nspatial) != self.seniority
            ):
                raise ValueError(
                    "Slater determinant, {0}, does not have the correct seniority, {1}"
                    "".format(bin(sd), self.seniority)
                )
        if sd_vec_is_empty:
            raise ValueError("No Slater determinants were provided.")

        self.sd_vec = tuple(sd_vec)

    def assign_params(self, params=None, add_noise=False):
        """Assign the parameters of the wavefunction.

        Parameters
        ----------
        params : {np.ndarray, None}
            Parameters of the wavefunction.
            Default corresponds to the ground state HF wavefunction.
        add_noise : {bool, False}
            Option to add noise to the given parameters.
            Default is False.

        """
        if params is None:
            params = np.zeros(len(self.sd_vec))
            params[0] = 1

        super().assign_params(params=params, add_noise=add_noise)

    def get_overlap(self, sd, deriv=None):
        r"""Return the overlap of the CI wavefunction with a Slater determinant.

        The overlap of the CI wavefunction with a Slater determinant is the coefficient of that
        Slater determinant in the wavefunction.

        .. math::

            \left< \Phi_i \middle| \Psi \right> = c_i

        where

        .. math::

            \left| \Psi \right> = \sum_i c_i \left| \Phi_i \right>

        Returns
        -------
        overlap : float
            Overlap of the CI wavefunction with the Slater determinant.

        Raises
        ------
        TypeError
            If given Slater determinant is not compatible with the format used internally.

        """
        sd = slater.internal_sd(sd)
        try:
            # pylint:disable=R1705
            if deriv is None:
                return self.params[self.dict_sd_index[sd]]
            elif deriv == self.dict_sd_index[sd]:
                return 1.0
            else:
                return 0.0
        except KeyError:
            return 0.0
