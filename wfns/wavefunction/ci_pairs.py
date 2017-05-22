"""CI Pairs Wavefunction.
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from .doci import DOCI
from ..backend.sd_list import sd_list
from ..backend import slater

__all__ = []


class CIPairs(DOCI):
    """CI Pairs Wavefunction.

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

    # FIXME: implement after ap1rog
    def to_ap1rog(self):
        """Return the AP1roG wavefunction that corresponds to the CIPairs wavefunction.

        Returns
        -------
        ap1rog : wfns.wavefunction.ap1rog.AP1roG
            AP1roG wavefunction
        """
        # import AP1roG (because nasty cyclic import business)
        from .ap1rog import AP1roG

        npair = self.nelec//2
        # dictionary of slater determinant to coefficient
        zip_sd_coeff = zip(self.civec, self.params)
        zip_sd_coeff.sort(reverse=True, key=lambda x: abs(x[1]))
        dict_sd_coeff = {sd: coeff for sd, coeff in zip_sd_coeff}
        # reference SD
        ref_sd = zip_sd_coeff[0][0]
        # normalize
        dict_sd_coeff = {sd: coeff/dict_sd_coeff[ref_sd] for sd, coeff in
                         dict_sd_coeff.iteritems()}
        # make ap1rog object
        ap1rog = AP1roG(self.nelec, self.one_int, self.two_int, dtype=self.dtype,
                        nuc_nuc=self.nuc_nuc, orbtype=self.orbtype, ref_sds=(ref_sd, ))
        # fill empty geminal coefficient
        gem_coeffs = np.zeros((npair, self.nspatial - npair))
        for occ_ind in (i for i in slater.occ_indices(ref_sd) if i < self.nspatial):
            for vir_ind in (a for a in slater.vir_indices(ref_sd, self.nspin) if a < self.nspatial):
                # excite slater determinant
                sd_exc = slater.excite(ref_sd, occ_ind, vir_ind)
                sd_exc = slater.excite(sd_exc, occ_ind+self.nspatial, vir_ind+self.nspatial)
                # set geminal coefficient (`a` decremented b/c first npair columns are removed)
                row_ind = ap1rog.dict_orbpair_ind[(occ_ind, occ_ind+self.nspatial)]
                col_ind = ap1rog.dict_orbpair_ind[(vir_ind, vir_ind+self.nspatial)]
                try:
                    gem_coeffs[row_ind, col_ind] = dict_sd_coeff[sd_exc]
                except KeyError:
                    gem_coeffs[row_ind, col_ind] = 0
        ap1rog.assign_params(np.hstack((gem_coeffs.flat, self.get_energy(include_nuc=False,
                                                                         exc_lvl=exc_lvl))))
        return ap1rog
