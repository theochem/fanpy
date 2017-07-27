""" Child of DOCI wavefunction (seniority zero) that uses only first order pair excitations
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from .doci import DOCI
from ..sd_list import sd_list
from .. import slater

__all__ = []


class CIPairs(DOCI):
    """ DOCI wavefunction with only first order pairwise excitations

    Attributes
    ----------
    nelec : int
        Number of electrons
    one_int : 1- or 2-tuple np.ndarray(K,K)
        One electron integrals for restricted, unrestricted, or generalized orbitals
        1-tuple for spatial (restricted) and generalized orbitals
        2-tuple for unrestricted orbitals (alpha-alpha and beta-beta components)
    two_int : 1- or 3-tuple np.ndarray(K,K)
        Two electron integrals for restricted, unrestricted, or generalized orbitals
        In physicist's notation
        1-tuple for spatial (restricted) and generalized orbitals
        3-tuple for unrestricted orbitals (alpha-alpha-alpha-alpha, alpha-beta-alpha-beta, and
        beta-beta-beta-beta components)
    dtype : {np.float64, np.complex128}
        Numpy data type
    nuc_nuc : float
        Nuclear-nuclear repulsion energy
    orbtype : {'restricted', 'unrestricted', 'generalized'}
        Type of the orbital used in obtaining the one-electron and two-electron integrals
    dict_exc_index : dict from int to int
        Dictionary from the excitation order to the column index of the coefficient matrix
    spin : float
        Total spin of the wavefunction
        Default is no spin (all spins possible)
        0 is singlet, 0.5 and -0.5 are doublets, 1 and -1 are triplets, etc
        Positive spin means that there are more alpha orbitals than beta orbitals
        Negative spin means that there are more beta orbitals than alpha orbitals
    civec : tuple of int
        List of Slater determinants used to construct the CI wavefunction

    Properties
    ----------
    nspin : int
        Number of spin orbitals (alpha and beta)
    nspatial : int
        Number of spatial orbitals

    Method
    ------
    __init__(self, nelec, one_int, two_int, dtype=None, nuc_nuc=None, orbtype=None)
        Initializes wavefunction
    assign_nelec(self, nelec)
        Assigns the number of electrons
    assign_dtype(self, dtype)
        Assigns the data type of parameters used to define the wavefunction
    assign_nuc_nuc(self, nuc_nuc=None)
        Assigns the nuclear nuclear repulsion
    assign_integrals(self, one_int, two_int, orbtype=None)
        Assigns integrals of the one electron basis set used to describe the Slater determinants
    assign_excs(self, excs=None)
        Assigns excitations to include in the calculation
    assign_spin(self, spin=None)
        Assigns the spin of the wavefunction
    assign_civec(self, civec=None)
        Assigns the tuple of Slater determinants used in the CI wavefunction
    get_energy(self, include_nuc=True, exc_lvl=0)
        Gets the energy of the CI wavefunction
    compute_density_matrix(self, exc_lvl=0, is_chemist_notation=False, val_threshold=0)
        Constructs the one and two electron density matrices for the given excitation level
    to_proj(self, Other, exc_lvl=0)
        Try to convert the CI wavefunction into the appropriate Projected Wavefunction
    compute_ci_matrix(self)
        Returns CI Hamiltonian matrix in the Slater determinant basis
    generate_civec(self)
        Returns a list of seniority 0 Slater determinants that are one pair excitation away from the
        ground state Slater determinant
    to_ap1rog(self, exc_lvl=0)
        Returns the CIPairs wavefunction as a AP1roG wavefunction
    """
    def generate_civec(self):
        """ Returns a list of Slater determinants for CI-Pairs wavefunction

        All seniority zero Slater determinants with only electron pair excitations from ground state

        Returns
        -------
        civec : list of ints
            Integer that describes the occupation of a Slater determinant as a bitstring
        """
        return sd_list(self.nelec, self.nspatial, exc_orders=[2], seniority=self.seniority)


    def to_ap1rog(self, exc_lvl=0):
        """ Returns the AP1roG wavefunction corresponding to the CIPairs wavefunction

        Parameters
        ----------
        exc_lvl : int
            Excitation level of the wavefunction
            0 is the ground state wavefunction
            i is the ith order excitation

        Returns
        -------
        ap1rog : wfns.proj.ap1rog.AP1roG
            AP1roG wavefunction

        Raises
        ------
        TypeError
            If excitation level is not an integer
        ValueError
            If excitation level is negative
        """
        # import AP1roG (because nasty cyclic import business)
        from ..proj.geminals.ap1rog import AP1roG

        if not isinstance(exc_lvl, int):
            raise TypeError('Excitation level must be an integer')
        if exc_lvl < 0:
            raise ValueError('Excitation level cannot be negative')

        npair = self.nelec//2
        # dictionary of slater determinant to coefficient
        zip_sd_coeff = zip(self.civec, self.sd_coeffs[:, self.dict_exc_index[exc_lvl]].flat)
        zip_sd_coeff.sort(reverse=True, key=lambda x: abs(x[1]))
        dict_sd_coeff = {sd : coeff for sd, coeff in zip_sd_coeff}
        # reference SD
        ref_sd = zip_sd_coeff[0][0]
        # normalize
        dict_sd_coeff = {sd : coeff/dict_sd_coeff[ref_sd] for sd, coeff in
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
