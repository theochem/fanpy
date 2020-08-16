import numpy as np

from fanpy.tools.graphs import generate_general_pmatch
from fanpy.wfn.geminal.apg import APG


class APG2(APG):
    def __init__(
        self, nelec, nspin, dtype=None, memory=None, ngem=None, orbpairs=None, params=None, tol=1e-4
    ):
        super().__init__(
            nelec, nspin, dtype=dtype, memory=memory, ngem=ngem, orbpairs=orbpairs, params=params
        )
        self.tol = tol
        self.connectivity = None

    def generate_possible_orbpairs(self, occ_indices):
        if self.connectivity is None:
            dead_indices = np.where(np.sum(np.abs(self.params), axis=0) < self.tol)[0]
            dead_orbpairs = np.array([self.get_orbpair(ind) for ind in dead_indices])
            connectivity = np.ones((self.nspin, self.nspin))
            if dead_orbpairs.size > 0:
                connectivity[dead_orbpairs[:, 0], dead_orbpairs[:, 1]] = 0
                connectivity[dead_orbpairs[:, 1], dead_orbpairs[:, 0]] = 0
            self.connectivity = connectivity
        else:
            connectivity = self.connectivity
        occ_indices = np.array(occ_indices)
        for pmatch, sign in generate_general_pmatch(
            occ_indices, connectivity[occ_indices[:, None], occ_indices[None, :]]
        ):
            yield pmatch, sign

    def clear_cache(self, key=None):
        super().clear_cache(key=key)
        self.connectivity = None


def test_something():
    test = APG2(2, 6, tol=1)
    test.params = np.ones(test.params.shape)
    test.params[:, test.get_col_ind((0, 1))] = 0
    test.params[:, test.get_col_ind((2, 3))] = 0
    # test.params[:, test.get_col_ind((0, 3))] = 0
    print()
    for i in test.generate_possible_orbpairs([0, 1, 2, 3]):
        print(i)
