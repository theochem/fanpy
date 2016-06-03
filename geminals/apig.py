from __future__ import absolute_import, division, print_function

import numpy as np
from gmpy2 import mpz

from .proj_wavefunction import ProjectionWavefunction
from . import slater
from .sd_list import doci_sd_list
from .proj_hamiltonian import doci_hamiltonian
from .math_tools import permanent_ryser

class APIG(ProjectionWavefunction):
    """ Antisymmetric Product of Interacting Geminals

    Attributes
    ----------
    dtype : {np.float64, np.complex128}
        Numpy data type
    H : tuple of np.ndarray(K,K)
        One electron integrals for the spatial orbitals
    G : tuple of np.ndarray(K,K,K,K)
        Two electron integrals for the spatial orbitals
    nuc_nuc : float
        Nuclear nuclear repulsion value
    nspatial : int
        Number of spatial orbitals
    nspin : int
        Number of spin orbitals (alpha and beta)
    nelec : int
        Number of electrons
    npair : int
        Number of electron pairs
        Assumes that the number of electrons is even
    nparticle : int
        Number of quasiparticles (electrons)
    ngeminal : int
        Number of geminals

    Private
    -------
    _methods : dict
        Default dimension of projection space
    _energy : float
        Electronic energy
    _nci : int
        Number of Slater determinants

    Methods
    -------
    """
    @property
    def template_params(self):
        gem_coeffs = np.eye(self.npair, self.nspatial, dtype=self.dtype)
        params = gem_coeffs.flatten()
        return params

    def compute_pspace(self, num_sd):
        """ Generates Slater determinants on which to project against

        Number of Slater determinants generated is determined strictly by the size of the
        projection space (self.nproj). First row corresponds to the ground state SD, then
        the next few rows are the first excited, then the second excited, etc

        Returns
        -------
        civec : list of ints
            Integer that describes the occupation of a Slater determinant as a bitstring
        """
        return doci_sd_list(self, num_sd)

    def compute_overlap(self, sd, deriv=None):
        # get indices of the occupied orbitals
        alpha_sd, beta_sd = slater.split_spin(sd, self.nspatial)
        occ_alpha_indices = slater.occ_indices(alpha_sd)
        occ_beta_indices = slater.occ_indices(beta_sd)
        if occ_alpha_indices != occ_beta_indices:
            raise ValueError('Given Slater determinant, {0}, does not belong'
                             ' to the DOCI Slater determinants'.format(bin(sd)))

        # build geminal coefficient
        if self.energy_is_param:
            gem_coeffs = self.params[:-1].reshape(self.npair, self.nspatial)
        else:
            gem_coeffs = self.params.reshape(self.npair, self.nspatial)

        val = 0.0
        if deriv is None:
            val = permanent_ryser(gem_coeffs[:, occ_alpha_indices])
            self.cache[sd] = val
        elif isinstance(deriv, int) and deriv < self.energy_index:
            row_to_remove = deriv // self.nspatial
            col_to_remove = deriv %  self.nspatial
            if col_to_remove in occ_alpha_indices:
                row_inds = [i for i in range(self.npair) if i != row_to_remove]
                col_inds = [i for i in occ_alpha_indices if i != col_to_remove]
                if len(row_inds) == 0 and len(col_inds) == 0:
                    val = 1
                else:
                    val = permanent_ryser(gem_coeffs[row_inds][:, col_inds])
                # construct new gmpy2.mpz to describe the slater determinant and
                # derivation index
                self.d_cache[(sd, deriv)] = val
        return val

    def compute_hamiltonian(self, sd, deriv=None):
        return sum(doci_hamiltonian(self, sd, self.orb_type, deriv=deriv))

    def normalize(self):
        # build geminal coefficient
        gem_coeffs = self.params[:self.energy_index].reshape(self.npair, self.nspatial)
        # normalize the geminals
        norm = np.sum(gem_coeffs**2, axis=1)
        gem_coeffs *= np.abs(norm[:, np.newaxis])**(-0.5)
        gem_coeffs[norm<0, :] *= -1
        # normalize the wavefunction
        norm = self.compute_norm()
        gem_coeffs *= norm**(-0.5/self.npair)
        # set attributes
        if self.energy_is_param:
            self.params = np.hstack((gem_coeffs.flatten(), self.params[-1]))
        else:
            self.params = gem_coeffs.flatten()
        # FIXME: need smarter caching (just delete the ones affected)
        self.cache = {}
        self.d_cache = {}

