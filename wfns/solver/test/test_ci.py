"""Test wfn.solver.ci."""
import numpy as np
from nose.tools import assert_raises
from wfns.wfn.ci.base import CIWavefunction
from wfns.ham.restricted_chemical import RestrictedChemicalHamiltonian
from wfns.solver import ci


class TestChemicalHamiltonian(RestrictedChemicalHamiltonian):
    """Class that overwrite integrate_sd_sd for simplicity."""
    def integrate_sd_sd(self, sd1, sd2, deriv=None):
        if sd1 > sd2:
            sd1, sd2 = sd2, sd1
        if [sd1, sd2] == [0b0011, 0b0011]:
            return [1]
        elif [sd1, sd2] == [0b0011, 0b1100]:
            return [3]
        elif [sd1, sd2] == [0b1100, 0b1100]:
            return [8]


def test_brute():
    """Test wfn.solver.ci.brute."""
    test_wfn = CIWavefunction(2, 4, sd_vec=[0b0011, 0b1100])
    test_ham = TestChemicalHamiltonian(np.ones((2, 2), dtype=float),
                                       np.ones((2, 2, 2, 2), dtype=float))
    # check type
    assert_raises(TypeError, ci.brute, None, test_ham, 0)
    assert_raises(TypeError, ci.brute, test_wfn, None, 0)

    test_ham = TestChemicalHamiltonian(np.ones((2, 2), dtype=complex),
                                       np.ones((2, 2, 2, 2), dtype=complex))
    assert_raises(ValueError, ci.brute, test_wfn, test_ham, 0)
    test_ham = TestChemicalHamiltonian(np.ones((3, 3), dtype=complex),
                                       np.ones((3, 3, 3, 3), dtype=complex))
    assert_raises(ValueError, ci.brute, test_wfn, test_ham, 0)

    test_wfn = CIWavefunction(2, 4, sd_vec=[0b0011, 0b1100])
    test_ham = TestChemicalHamiltonian(np.ones((2, 2), dtype=float),
                                       np.ones((2, 2, 2, 2), dtype=float))
    assert_raises(TypeError, ci.brute, test_wfn, test_ham, None)
    assert_raises(TypeError, ci.brute, test_wfn, test_ham, -1)

    # 0 = det [[1, 3]
    #          [3, 8]]
    # 0 = (1-lambda)(8-lambda) - 3*3
    # 0 = lambda^2 - 9*lambda - 1
    # lambda = (9 \pm \sqrt{9^2 + 4}) / 2
    #        = (9 \pm \sqrt{85}) / 2
    # [[1-lambda,        3], [[v1],   [[0],
    #  [3       , 8-lambda]]  [v2]] =  [0]]
    results = ci.brute(test_wfn, test_ham)
    energies = results['eigval']
    coeffs = results['eigvec']
    assert np.allclose(energies[0], (9 - 85**0.5)/2)
    matrix = np.array([[1-energies[0], 3], [3, 8-energies[0]]])
    assert np.allclose(matrix.dot(coeffs[:, 0]), np.zeros(2))
    assert np.allclose(energies[1], (9 + 85**0.5)/2)
    matrix = np.array([[1-energies[1], 3], [3, 8-energies[1]]])
    assert np.allclose(matrix.dot(coeffs[:, 1]), np.zeros(2))
