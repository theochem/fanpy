"""Tests wfn.ci.solver
"""
import numpy as np
from nose.tools import assert_raises
from wfns.ci.ci_wavefunction import CIWavefunction
from wfns.ci.solver import solve


class TestCIWavefunction(CIWavefunction):
    """ Child of CIWavefunction used to test CIWavefunction

    Because CIWavefunction is an abstract class
    """
    def generate_civec(self):
        pass

    def compute_ci_matrix(self):
        return np.array([[1, 3], [4, 8]])

def test_solve_eigh():
    """ Tests wfn.ci.solver.solve with the eigh algorithm
    """
    test_wfn = TestCIWavefunction(2, np.ones((2, 2)), np.ones((2, 2, 2, 2)), excs=[0, 1],
                                  civec=[0b0011, 0b1100])
    # check type
    assert_raises(TypeError, lambda: solve(test_wfn, solver_type='random'))

    # check results
    results = solve(test_wfn, solver_type='eigh')
    assert results['success']
    assert np.allclose(test_wfn.sd_coeffs, np.array([[-0.91063291, 0.41321628],
                                                     [0.41321628, 0.91063291]]))
    assert np.allclose(test_wfn.energies, np.array([-0.81507291, 9.81507291]))
    # check exc order
    test_wfn = TestCIWavefunction(2, np.ones((2, 2)), np.ones((2, 2, 2, 2)), excs=[1, 0],
                                  civec=[0b0011, 0b1100])
    results = solve(test_wfn, solver_type='eigh')
    assert results['success']
    assert np.allclose(test_wfn.sd_coeffs, np.array([[0.41321628, -0.91063291],
                                                     [0.91063291, 0.41321628]]))
    assert np.allclose(test_wfn.energies, np.array([9.81507291, -0.81507291]))
