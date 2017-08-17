"""CI Pairs Wavefunction.
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from .doci import DOCI
from ..geminals.ap1rog import AP1roG
from ...backend.sd_list import sd_list
from ...backend import slater

__all__ = []


class CIPairs(DOCI):
    r"""CI Pairs Wavefunction.

    DOCI wavefunction with only first order pairwise excitations

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
        Spin is zero (singlet)
    seniority : int, None
        Seniority (number of unpaired electrons) of the wavefunction
        Seniority is zero
    template_params : np.ndarray
        Template of the wavefunction parameters
        Depends on the attributes given
    nparams : int
        Number of parameters
    params_shape : 2-tuple of int
        Shape of the parameters

    Methods
    -------
    __init__(self, nelec, nspin, dtype=None, params=None, sd_vec=None, spin=None, seniority=None)
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
    to_ap1rog(self, exc_lvl=0)
        Returns the CIPairs wavefunction as a AP1roG wavefunction
    """
    def assign_sd_vec(self, sd_vec=None):
        """Set the list of Slater determinants for the CI Pairs wavefunction.

        Ignores user input and uses the Slater determinants for the FCI wavefunction (within the
        given spin)

        Parameters
        ----------
        sd_vec : iterable of int
            List of Slater determinants (in the form of integers that describe the occupation as a
            bitstring)

        Raises
        ------
        ValueError
            If the sd_vec is not `None` (default value)

        Note
        ----
        Needs to have `nelec`, `nspin`, `spin`, `seniority`
        """
        if sd_vec is None:
            super().assign_sd_vec(sd_list(self.nelec, self.nspatial, num_limit=None, exc_orders=[2],
                                          spin=self.spin, seniority=self.seniority))
        else:
            raise ValueError('Only the default list of Slater determinants is allowed. i.e. sd_vec '
                             'is `None`. If you would like to customize your CI wavefunction, use '
                             'CIWavefunction instead.')

    def to_ap1rog(self):
        """Return the AP1roG wavefunction that corresponds to the CIPairs wavefunction.

        Returns
        -------
        ap1rog : wfns.wavefunction.geminals.ap1rog.AP1roG
            AP1roG wavefunction
        """
        # select Slater determinant with largest contributor as the reference Slater determinant
        ref_sd_ind = np.argsort(np.abs(self.params))[-1]
        ref_sd = self.sd_vec[ref_sd_ind]
        spatial_ref_sd, _ = slater.split_spin(ref_sd, self.nspatial)
        # use ap1rog normalization scheme
        ci_params = self.params / self.params[ref_sd_ind]

        # create AP1roG
        ap1rog = AP1roG(self.nelec, self.nspin, dtype=self.dtype, ngem=None, orbpairs=None,
                        ref_sd=ref_sd, params=None)
        # fill empty geminal coefficient
        gem_coeffs = np.zeros(ap1rog.params.shape, dtype=self.dtype)
        for occ_ind in slater.occ_indices(spatial_ref_sd):
            for vir_ind in slater.vir_indices(spatial_ref_sd, self.nspatial):
                # excite slater determinant
                sd_exc = slater.excite(ref_sd, occ_ind, vir_ind)
                sd_exc = slater.excite(sd_exc, occ_ind+self.nspatial, vir_ind+self.nspatial)
                # set geminal coefficient (`a` decremented b/c first npair columns are removed)
                row_ind = ap1rog.dict_reforbpair_ind[(occ_ind, occ_ind+self.nspatial)]
                col_ind = ap1rog.dict_orbpair_ind[(vir_ind, vir_ind+self.nspatial)]
                try:
                    gem_coeffs[row_ind, col_ind] = ci_params[self.dict_sd_index[sd_exc]]
                except KeyError:
                    gem_coeffs[row_ind, col_ind] = 0.0
        ap1rog.assign_params(gem_coeffs)

        return ap1rog
