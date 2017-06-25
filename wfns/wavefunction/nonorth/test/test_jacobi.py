"""Test wfns.wavefunction.nonorth.jacobi"""
from __future__ import absolute_import, division, print_function
from nose.tools import assert_raises
import numpy as np
from wfns.wavefunction.nonorth.jacobi import JacobiWavefunction


class TestJacobiWavefunction(JacobiWavefunction):
    """Class to test JacobiWavefunction."""
    def __init__(self):
        pass


def test_jacobi_template_params():
    """Test JacobiWavefunction.template_params."""
    test = TestJacobiWavefunction()
    assert test.template_params.size == 1
    assert test.template_params.shape == ()
    assert np.allclose(test.template_params, [0])


def test_jacobi_assign_orbtype():
    """Test JacobiWavefunction.assign_orbtype."""
    test = TestJacobiWavefunction()
    test.nelec = 4
    test.nspin = 10
    test.dtype = np.float64
    test.memory = 10

    test.assign_orbtype(None)
    assert test.orbtype == 'restricted'
    test.assign_orbtype('restricted')
    assert test.orbtype == 'restricted'
    test.assign_orbtype('unrestricted')
    assert test.orbtype == 'unrestricted'
    test.assign_orbtype('generalized')
    assert test.orbtype == 'generalized'

    assert_raises(ValueError, test.assign_orbtype, 'lkjsadf')


def test_jacobi_assign_jacobi_indices():
    """Test JacobiWavefunction.assign_jacobi_indices."""
    test = TestJacobiWavefunction()
    test.nelec = 4
    test.nspin = 10
    test.dtype = np.float64
    test.memory = 10

    # not tuple or list
    assert_raises(TypeError, test.assign_jacobi_indices, 1)
    assert_raises(TypeError, test.assign_jacobi_indices, {1: 0, 2: 3})
    assert_raises(TypeError, test.assign_jacobi_indices, set([0, 1]))
    # bad number of entries
    assert_raises(TypeError, test.assign_jacobi_indices, [0])
    assert_raises(TypeError, test.assign_jacobi_indices, (1, 2, 3))
    # bad type of entries
    assert_raises(TypeError, test.assign_jacobi_indices, [0, 1.0])
    assert_raises(TypeError, test.assign_jacobi_indices, ('0', 1))

    # negative entry
    assert_raises(ValueError, test.assign_jacobi_indices, [0, -1])
    # same entries
    assert_raises(ValueError, test.assign_jacobi_indices, [0, 0])
    # bad input for given orbital type
    test.orbtype = 'generalized'
    assert_raises(ValueError, test.assign_jacobi_indices, [0, 10])
    test.assign_jacobi_indices([0, 9])
    assert test.jacobi_indices == (0, 9)
    test.orbtype = 'restricted'
    assert_raises(ValueError, test.assign_jacobi_indices, [0, 5])
    test.assign_jacobi_indices([0, 4])
    assert test.jacobi_indices == (0, 4)
    test.orbtype = 'unrestricted'
    assert_raises(ValueError, test.assign_jacobi_indices, [0, 6])
    assert_raises(ValueError, test.assign_jacobi_indices, [1, 9])
    test.assign_jacobi_indices([0, 1])
    assert test.jacobi_indices == (0, 1)
    test.assign_jacobi_indices([5, 6])
    assert test.jacobi_indices == (5, 6)

    # swap entries
    test.assign_jacobi_indices([4, 1])
    assert test.jacobi_indices == (1, 4)


def test_jacobi_get_overlap():
    """Test JacobiWavefunction.get_overlap"""
    test = TestJacobiWavefunction()
    test.nelec = 2
    test.nspin = 4
    test.dtype = np.float64
    test.memory = 10
    test.assign_wfn(None)
    test._cache_fns = {}
    test.wfn.params = np.arange(1, 7)
    wfn_sd_coeff = {0b0101: 1, 0b0110: 2, 0b1100: 3, 0b0011: 4, 0b1001: 5, 0b1010: 6}

    # generalized
    test.assign_orbtype('generalized')
    test.assign_jacobi_indices((0, 3))
    # jacobi matrix [[cos, 0, 0, sin],
    #                [0, 1, 0, 0],
    #                [0, 0, 1, 0],
    #                [-sin, 0, 0, cos]]
    test.assign_params(np.array(1.57079632679))
    # 0b1001 uses [[cos, 0, 0, sin],
    #              [-sin, 0, 0, cos]]
    assert test.get_overlap(0b1001) == 1 * wfn_sd_coeff[0b1001]
    # 0b1001 uses [[0, 1, 0, 0],
    #              [0, 0, 1, 0]]
    assert test.get_overlap(0b0110) == 1 * wfn_sd_coeff[0b0110]
    # 0b0101 uses [[cos, 0, 0, sin],
    #              [0, 0, 1, 0]]
    assert test.get_overlap(0b0101) == (np.cos(test.params) * wfn_sd_coeff[0b0101] +
                                        np.sin(test.params) * wfn_sd_coeff[0b1100])
