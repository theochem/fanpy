import networkx
import numpy as np

from wfns.backend.slater import sign_perm
from wfns.wfn.geminal.apg import APG


class APG4(APG):
    def __init__(
            self, nelec, nspin, dtype=None, memory=None, ngem=None, orbpairs=None, params=None,
            tol=1e-4, num_matchings=2
    ):
        super().__init__(nelec, nspin, dtype=dtype, memory=memory, ngem=ngem, orbpairs=orbpairs,
                         params=params)
        self.tol = tol
        self.num_matchings = num_matchings
        self.adjacency = None
        self.weights = None

    def generate_possible_orbpairs(self, occ_indices):
        if self.adjacency is None:
            weights = np.sum(np.abs(self.params), axis=0)
            full_adjacency = np.zeros((self.nspin, self.nspin))
            # FIXME: make adjacency matrix as big the the number of occupied orbitals, not spin orbitals
            for i in range(self.nspin):
                for j in range(i + 1, self.nspin):
                    col_ind = self.get_col_ind((i, j))
                    if weights[col_ind] < self.tol:
                        continue
                    full_adjacency[i, j] = np.log10(weights[col_ind]) - np.log10(self.tol)
            full_adjacency += full_adjacency.T
            self.adjacency = full_adjacency.copy()
            self.weights = weights
        else:
            full_adjacency = self.adjacency.copy()
            weights = self.weights
        occ_indices = np.array(occ_indices)

        for num in range(self.num_matchings):
            adjacency = full_adjacency[occ_indices[:, None], occ_indices[None, :]]
            graph = networkx.from_numpy_matrix(adjacency)
            pmatch = sorted(
                networkx.max_weight_matching(graph, maxcardinality=True), key=lambda i: i[0]
            )
            if len(pmatch) != len(occ_indices) // 2:
                # print(
                #     f"WARNING: There are no matchings left. Only {num} pairing schemes will be used."
                # )
                break
            pmatch = [(occ_indices[i[0]], occ_indices[i[1]]) for i in pmatch]
            sign = sign_perm([j for i in pmatch for j in i], is_decreasing=False)
            yield pmatch, sign

            edge_weights = [weights[self.get_col_ind(orbpair)] for orbpair in pmatch]
            min_edge = pmatch[np.argmin(edge_weights)]
            full_adjacency[min_edge[0], min_edge[1]] = 0
            full_adjacency[min_edge[1], min_edge[0]] = 0


def test_something():
    test = APG4(2, 6, tol=0.1, num_matchings=3)
    test.params = np.ones(test.params.shape) * 0.9
    test.params[:, test.get_col_ind((0, 1))] = 0
    test.params[:, test.get_col_ind((2, 3))] = 0
    # test.params[:, test.get_col_ind((0, 3))] = 0
    print()
    for i in test.generate_possible_orbpairs([0, 1, 2, 3]):
        print(i)
