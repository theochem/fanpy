from __future__ import absolute_import, division, print_function

import numpy as np
from gmpy2 import mpz

from .proj_wavefunction import ProjectionWavefunction
from . import slater
from .sd_list import ci_sd_list
from .math_tools import permanent_ryser


class APFSG(ProjectionWavefunction):
    """ Antisymmetric Product of Factorized Set Geminals
    """

    @property
    def template_params(self):
        init_pair = np.eye(self.npair, self.nspatial, dtype=self.type)
        init_unpair = np.zeros((self.npair, self.nspatial), dtype=self.type)
        init_spin = np.eye(2 * self.nspatial, 2 *
                           self.nspatial, dtype=self.type).flatten()
        init_spatial = np.hstack((init_pair, init_unpair)).flatten()
        return np.hstack((init_spatial, init_spin))

    def compute_pspace(self, num_sd):
        number_sd = 2 * self.npair * self.nspatial + 4 * self.nspatial ** 2
        return ci_sd_list(number_sd)

    def compute_overlap(self, sd, deriv=None):
        sd = mpz(sd)
        # get occupied orbital indices
        alpha_sd, beta_sd = slater.split_spin(sd, self.nspatial)
        occ_alpha_indices = slater.occ_indices(alpha_sd)
        occ_beta_indices = slater.occ_indices(beta_sd)
        # build geminal coefficient
        boson_part = self.params[
            :self.npair * self.nspatial * 2].reshape(self.npair, self.nspatial)
        c_cs, a_os = boson_part[
            :, :self.nspatial], boson_part[:, self.nspatial:]
        fermion = self.params[self.npair * self.nspatial *
                              2:].reshape(2 * self.nspatial, 2 * self.nspatial)
        new_c_cs = np.zeros(c_cs.shape)
        new_a_os = np.zeros(a_os.shape)
        new_b_os = np.zeros(fermion.shape)
        for i in occ_alpha_indices:
            new_c_cs[:, i] = c_cs[:, i]
            new_b_os[:, 2 * i] = b_os[:, 2 * i]
            new_b_os[2 * i, :] = b_os[2 * i, :]
        for j in occ_beta_indices:
            new_a_os[:, j] = a_os[:, j]
            new_b_os[:, 2 * i + 1] = b_os[:, 2 * i + 1]
            new_b_os[2 * i + 1, :] = b_os[2 * i + 1, :]

    def compute_hamiltonian(self, sd, deriv=None):
        pass

    def normlize(self):
        pass
