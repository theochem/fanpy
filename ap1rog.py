from __future__ import absolute_import, division, print_function

import numpy as np

from apig import Apig


class Ap1rog(Apig):
    """
    Antisymmetrized Product of One Reference Orbital Geminals wavefunction.

    """

    _normalize = False

    _solver_options = Apig._solver_options.copy()
    _solver_options.update({
        "ftol": 1.0e-15,
        "gtol": 1.0e-15,
        "xtol": 1.0e-15,
    })

    def _make_npspace(self):
        """
        Number of Slater determinants in the projection space.

        """

        return self.npair * (self.nbasis - self.npair)

    def _make_nhspace(self):
        """
        Number of excited Slater determinants generated by one application of the Hamiltonian to
        another Slater determinant.

        """

        return self.npair * (self.nbasis - self.npair)

    def _make_dim_deriv(self):
        """
        Partitioning of parameters wrt which partial derivatives are taken.

        """

        return (self.npair, self.nbasis - self.npair)

    def _make_index_gen(self):
        """
        General indices for each unique Slater determinant.

        """

        index_gen = -np.ones((self.nsd, 2 * self.npair + 1), dtype=int)
        for key, value in self.dict.iteritems():
            occ = [i for i in range(self.npair) if not key[i]]
            vir = [i - self.npair for i in range(self.npair, self.nbasis) if key[i]]
            nexc = len(occ)
            tmp = -np.ones(2 * self.npair + 1, dtype=int)
            tmp[0] = nexc
            tmp[1:(len(occ) + 1)] = occ
            tmp[(self.npair + 1):(len(vir) + self.npair + 1)] = vir
            index_gen[value, :] = tmp
        return index_gen

    def _make_x(self):
        """
        Generate an initial guess at the geminal coefficient vector.

        """

        # Generate empty matrix of proper shape and dtype
        dim = (self.npair, self.nbasis - self.npair)
        x = np.zeros(dim, dtype=self.dtype)

        # Add random noise
        x[:, :] += (0.2 / x.size) * (np.random.rand(*dim) - 0.5)

        # Return the raveled matrix as a vector
        x = x.ravel()
        return x

    def _make_C(self):
        """
        Generate a representation of the geminal coefficient matrix.

        """

        return self.x.reshape(self.npair, self.nbasis - self.npair)

    def _compute_overlap(self, index, deriv=None):
        """
        Compute the overlap of the indexth Slater determinant in the cache.

        """


        if deriv:
            nexc = self.index_gen[index, 0]
            rows = self.index_gen[index, 1:(1 + nexc)].tolist()
            cols = self.index_gen[index, (1 + self.npair):(1 + self.npair + nexc)].tolist()
            if deriv[0] not in rows or deriv[1] not in cols:
                dolp = 0.
            else:
                matrix = self.C[rows][:, cols].copy()
                matrix[rows.index(deriv[0]), cols.index(deriv[1])] = 1.
                dolp = self.permanent(matrix)
            return dolp
        else:
            nexc = self.index_gen[index, 0]
            rows = self.index_gen[index, 1:(1 + nexc)].tolist()
            cols = self.index_gen[index, (1 + self.npair):(1 + self.npair + nexc)].tolist()
            olp = self.permanent(self.C[rows][:, cols])
            return olp

    def to_apig(self, dtype=None):
        """
        Initialize an APIG wavefunction instance using this AP1roG instance's coefficient vector as
        the initial guess for the APIG coefficient vector.

        """

        dtype = self.dtype if dtype is None else dtype
        extra = self.npspace - self._make_npspace()
        x = np.zeros((self.npair, self.nbasis), dtype=dtype)
        x[:, :self.npair] += np.eye(self.npair)
        x[:, self.npair:] += self.C
        x = x.ravel()
        return Apig(self.nelec, self.H, self.G, dtype=dtype, extra=extra, x=x)


# vim: set nowrap textwidth=100 cc=101 :
