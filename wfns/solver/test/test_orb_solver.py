"""Test wfns.solver.orb_solver."""
from __future__ import absolute_import, division, print_function
import numpy as np
from wfns.tools import find_datafile
from wfns.wavefunction.ci.doci import DOCI
from wfns.hamiltonian.chemical_hamiltonian import ChemicalHamiltonian
from wfns.solver.orb_solver import optimize_ham_orbitals_jacobi
from wfns.solver import ci_solver


def test_optimize_ham_orbitals_jacobi_h4_doci():
    """Test orbital optimization of DOCI wavefunction for H4 (square) (STO-6G).

    Uses Peter's Davidson code as a reference.

    HF energy: -1.13126983927
    OO DOCI energy: -1.884948574812363

    NOTE
    ----
    Optimized orbitals are read in from Peter's code
    """
    nelec = 4
    nspin = 8
    doci = DOCI(nelec, nspin)

    one_int = np.load(find_datafile('test/h4_square_hf_sto6g_oneint.npy'))
    two_int = np.load(find_datafile('test/h4_square_hf_sto6g_twoint.npy'))
    nuc_nuc = 2.70710678119
    ham = ChemicalHamiltonian(one_int, two_int, orbtype='restricted', energy_nuc_nuc=nuc_nuc)

    # FIXME: magic
    ham.orb_rotate_jacobi((2, 3), 0.4)

    # import matplotlib.pyplot as plt
    # import itertools as it

    # def _objective(theta, p=0, q=1):
    #     sds = sd_list(4, 4, num_limit=None, exc_orders=None)
    #     rotated_ham = copy.deepcopy(ham)
    #     rotated_ham.orb_rotate_jacobi((p, q), theta)
    #     ci_solver.eigen_solve(doci, rotated_ham, exc_lvl=0)

    #     norm = sum(doci.get_overlap(sd)**2 for sd in sds)
    #     energy = sum(doci.get_overlap(sd) * sum(rotated_ham.integrate_wfn_sd(doci, sd))
    #                  for sd in sds)
    #     return energy / norm

    # num_steps = 50
    # thetas = np.linspace(-np.pi, np.pi, num_steps)
    # for orbpair in it.combinations(range(4), 2):
    #     plt.plot(thetas, [_objective(i, *orbpair) for i in thetas], label=orbpair)

    # plt.legend()
    # plt.show()

    optimize_ham_orbitals_jacobi(doci, ham, wfn_solver=ci_solver.eigen_solve)
    energy = ci_solver.eigen_solve(doci, ham, exc_lvl=0)

    assert np.isclose(energy, -4.59205536957)
