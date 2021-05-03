"""Utility function for constructing Wavefunction instances."""
from fanpy.wfn.base import BaseWavefunction
from fanpy.tools import slater

import numpy as np


def wfn_factory(olp, olp_deriv, nelec, nspin, params, memory=None, assign_params=None):
    """Return the instance of the Wavefunction class with the given overlaps.

    Parameters
    ----------
    olp(sd, params) : function
        Function that returns the overlap from the given Slater determinant and the wavefunction
        parameters.
        `sd` is an integer whose bitstring describes the occupation of the Slater determinant. See
        `fanpy.tools.slater` for more details.
        `params` is a numpy array that contains the parameters of the wavefunction. If other types
        of parameters is desired, `assign_params` should be provided.
    olp_deriv(sd, params) : function
        Function that returns the derivatives of the overlap with respect to the wavefunction
        parameters from the given Slater determinant and the wavefunction parameters.
        `sd` is an integer whose bitstring describes the occupation of the Slater determinant. See
        `fanpy.tools.slater` for more details.
        `params` is a numpy array that contains the parameters of the wavefunction. If other types
        of parameters is desired, `assign_params` should be provided.
    nelec : int
        Number of electrons in the wavefunction.
    nspin : int
        Number of spin orbitals in the wavefunction.
    params : np.ndarray
        Parameters of the wavefunction.
        If is not a numpy array, then `assign_params` should be provided.
    memory : {float, int, str, None}
        Memory available (in bytes) for the wavefunction.
        If number is provided, it is the number of bytes.
        If string is provided, it should end iwth either "mb" or "gb" to specify the units.
        Default does not limit memory usage (i.e. infinite).
    assign_params(self, params) : function
        Method for assigning the parameters in the wavefunction class. First argument is `self` and
        second argument is `params`.
        Default uses `BaseWavefunction.assign_params`.

    """

    class GeneratedWavefunction(BaseWavefunction):
        """Generated wavefunction class from the given olp and olp_deriv."""

        def __init__(self, nelec, nspin, memory=None, params=None):
            """Initialize the wavefunction.

            Parameters
            ----------
            nelec : int
                Number of electrons.
            nspin : int
                Number of spin orbitals.
            memory : {float, int, str, None}
                Memory available for the wavefunction.
                If number is provided, it is the number of bytes.
                If string is provided, it should end iwth either "mb" or "gb" to specify the units.
                Default does not limit memory usage (i.e. infinite).
            params : np.ndarray
                Parameters of the wavefunction.

            """
            super().__init__(nelec, nspin, memory=memory)
            self.assign_params(params)

        def assign_params(self, params=None):
            """Assign the parameters of the wavefunction.

            Parameters
            ----------
            params
                Parameters of the wavefunction.

            """
            if assign_params is None:
                super().assign_params(params)
            else:
                assign_params(self, params)

        def _olp(self, sd):  # pylint: disable=E0202
            """Return overlap of the wavefunction with the Slater determinant.

            Parameters
            ----------
            sd : int
                Occupation vector of a Slater determinant given as a bitstring.
                Assumed to have the same number of electrons as the wavefunction.

            Returns
            -------
            olp : {float, complex}
                Overlap of the current instance with the given Slater determinant.

            """
            return olp(sd, self.params)

        def _olp_deriv(self, sd):  # pylint: disable=E0202
            """Return the derivatives of the overlap with the Slater determinant.

            Parameters
            ----------
            sd : int
                Occupation vector of a Slater determinant given as a bitstring.
                Assumed to have the same number of electrons as the wavefunction.

            Returns
            -------
            olp_deriv : np.ndarray
                Derivatives of the overlap with respect to all of the wavefunction's parameters.

            """
            return olp_deriv(sd, self.params)

        def get_overlap(self, sd, deriv=None):
            r"""Return the overlap of the wavefunction with a Slater determinant.

            Parameters
            ----------
            sd : int
                Slater Determinant against which the overlap is taken.
            deriv : {np.ndarray, None}
                Indices of the parameters with respect to which the overlap is derivatized.
                Default returns the overlap without derivatization.

            Returns
            -------
            overlap : {float, np.ndarray}
                Overlap (or derivative of the overlap) of the wavefunction with the given Slater
                determinant.

            Raises
            ------
            TypeError
                If Slater determinant is not an integer.
                If deriv is not a one dimensional numpy array of integers.

            """
            # if no derivatization
            if deriv is None:
                return self._olp(sd)
            # if derivatization
            return self._olp_deriv(sd)[deriv]

    return GeneratedWavefunction(nelec, nspin, params=params, memory=memory)


def convert_to_fanci(wfn, ham, nproj=None, proj_wfn=None, **kwargs):
    """Covert the given wavefunction instance to that of FanCI class.

    https://github.com/QuantumElephant/FanCI

    Parameters
    ----------
    wfn : BaseWavefunction
    ham : pyci.hamiltonian
        PyCI Hamiltonian.
    nproj : int, optional
        Number of determinants in projection ("P") space.
    wfn : pyci.doci_wfn, optional
        If specified, this PyCI wave function defines the projection ("P") space.
    kwargs : Any, optional
        Additional keyword arguments for base FanCI class.

    Returns
    -------
    new_wfn : FanCI

    """
    from typing import Any, Tuple, Union
    import pyci
    from fanci.fanci import FanCI

    class GeneratedFanCI(FanCI):
        """Generated FanCI wavefunction class from the fanpy wavefunction.

        Does not work for composite wavefunctions.

        """
        def __init__(
            self,
            fanpy_wfn: BaseWavefunction,
            ham: pyci.hamiltonian,
            nocc: int,
            nproj: int = None,
            wfn: pyci.doci_wfn = None,
            **kwargs: Any,
        ) -> None:
            r"""
            Initialize the FanCI problem.

            Parameters
            ----------
            fanpy_wfn : BaseWavefunction
                Wavefunction from fanpy.
            ham : pyci.hamiltonian
                PyCI Hamiltonian.
            nocc : int
                Number of occupied orbitals.
            nproj : int, optional
                Number of determinants in projection ("P") space.
            wfn : pyci.doci_wfn, optional
                If specified, this PyCI wave function defines the projection ("P") space.
            kwargs : Any, optional
                Additional keyword arguments for base FanCI class.

            """
            if not isinstance(ham, pyci.hamiltonian):
                raise TypeError(f"Invalid `ham` type `{type(ham)}`; must be `pyci.hamiltonian`")

            # Compute number of parameters
            # FIXME: i'm guessing that energy is always a parameter
            nparam = wfn.nparams + 1

            # Handle default nproj
            nproj = nparam if nproj is None else nproj

            # Handle default wfn (P space == single pair excitations)
            if wfn is None:
                wfn = pyci.doci_wfn(ham.nbasis, nocc, nocc)
                wfn.add_excited_dets(1)
            elif not isinstance(wfn, pyci.doci_wfn):
                raise TypeError(f"Invalid `wfn` type `{type(wfn)}`; must be `pyci.doci_wfn`")
            elif wfn.nocc_up != nocc or wfn.nocc_dn != nocc:
                raise ValueError(f"wfn.nocc_{{up,dn}} does not match `nocc={nocc}` parameter")

            # Initialize base class
            FanCI.__init__(self, ham, wfn, nproj, nparam, **kwargs)

            # Get results of 'searchsorted(i)' from i=0 to i=nbasis for each det. in "S" space
            arange = np.arange(self._wfn.nbasis, dtype=pyci.c_long)
            sspace_data = [occs.searchsorted(arange) for occs in self._sspace]
            pspace_data = sspace_data[: self._nproj]

            # Save sub-class -specific attributes
            self._fanpy_wfn = fanpy_wfn
            self._sspace_data = sspace_data
            self._pspace_data = pspace_data

        def compute_overlap(self, x: np.ndarray, occs_array: Union[np.ndarray, str]) -> np.ndarray:
            r"""
            Compute the FanCI overlap vector.

            Parameters
            ----------
            x : np.ndarray
                Parameter array, [p_0, p_1, ..., p_n].
            occs_array : (np.ndarray | 'P' | 'S')
                Array of determinant occupations for which to compute overlap. A string "P" or "S" can
                be passed instead that indicates whether ``occs_array`` corresponds to the "P" space
                or "S" space, so that a more efficient, specialized computation can be done for these.

            Returns
            -------
            ovlp : np.ndarray
                Overlap array.

            """
            if isinstance(occs_array, np.ndarray):
                pass
            elif occs_array == "P":
                occs_array = self._pspace
            elif occs_array == "S":
                occs_array = self._sspace
            else:
                raise ValueError("invalid `occs_array` argument")

            # FIXME: converting occs_array to slater determinants to be converted back to indices is
            # a waste
            # convert slater determinants
            sds = []
            for i, occs in enumerate(occs_array):
                # FIXME: CHECK IF occs IS BOOLEAN OR INTEGERS
                # convert occupation vector to sd
                if occs.dtype == bool:
                    sd = slater.create(0, *np.where(occs)[0])
                else:
                    sd = slater.create(0, *occs)
                sds.append(sd)

            # Feed in parameters into fanpy wavefunction
            self._fanpy_wfn.assign_params(x)

            # initialize
            y = np.empty(occs_array.shape[0], dtype=pyci.c_double)

            # Compute overlaps of occupation vectors
            if hasattr(self._fanpy_wfn, "get_overlaps"):
                y += self._fanpy_wfn.get_overlaps(sds)
            else:
                for i, sd in enumerate(sds):
                    y[i] = self._fanpy_wfn.get_overlap(sd)
            return y

        def compute_overlap_deriv(
            self, x: np.ndarray, occs_array: Union[np.ndarray, str]
        ) -> np.ndarray:
            r"""
            Compute the FanCI overlap derivative matrix.

            Parameters
            ----------
            x : np.ndarray
                Parameter array, [p_0, p_1, ..., p_n].
            occs_array : (np.ndarray | 'P' | 'S')
                Array of determinant occupations for which to compute overlap. A string "P" or "S" can
                be passed instead that indicates whether ``occs_array`` corresponds to the "P" space
                or "S" space, so that a more efficient, specialized computation can be done for these.

            Returns
            -------
            ovlp : np.ndarray
                Overlap derivative array.

            """
            if isinstance(occs_array, np.ndarray):
                pass
            elif occs_array == "P":
                occs_array = self._pspace
            elif occs_array == "S":
                occs_array = self._sspace
            else:
                raise ValueError("invalid `occs_array` argument")

            # FIXME: converting occs_array to slater determinants to be converted back to indices is
            # a waste
            # convert slater determinants
            sds = []
            for i, occs in enumerate(occs_array):
                # FIXME: CHECK IF occs IS BOOLEAN OR INTEGERS
                # convert occupation vector to sd
                if occs.dtype == bool:
                    sd = slater.create(0, *np.where(occs)[0])
                else:
                    sd = slater.create(0, *occs)
                sds.append(sd)

            # Feed in parameters into fanpy wavefunction
            self._fanpy_wfn.assign_params(x)

            # Shape of y is (no. determinants, no. active parameters excluding energy)
            y = np.zeros((occs_array.shape[0], self._nactive - self._mask[-1]), dtype=pyci.c_double)

            # Compute derivatives of overlaps
            if hasattr(self._fanpy_wfn, "get_overlaps"):
                y += self._fanpy_wfn.get_overlaps(sds, deriv=True)
            else:
                for i, sd in enumerate(sds):
                    y[i] = self._fanpy_wfn.get_overlap(
                        sd, deriv=np.arange(self.nparam)[self._mask[:-1]]
                    )
            return y

    return GeneratedFanCI(wfn, ham, wfn.nelec, nproj=nproj, wfn=proj_wfn, **kwargs)
