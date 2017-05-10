""" Tests wfns.wavefunction.ci_wavefunction
"""
from __future__ import absolute_import, division, print_function
from nose.tools import assert_raises
import numpy as np
from wfns.wavefunction.ci_wavefunction import CIWavefunction
from wfns.hamiltonian.ci_matrix import ci_matrix


def test_assign_spin():
    """ Tests CIWavefunction.assign_spin
    """
    test = CIWavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)))
    # check error
    assert_raises(TypeError, lambda: test.assign_spin('1'))
    assert_raises(TypeError, lambda: test.assign_spin([1]))
    assert_raises(TypeError, lambda: test.assign_spin(long(1)))
    # None
    test.assign_spin(None)
    assert test.spin is None
    # Int assigned
    test.assign_spin(10)
    assert test.spin == 10
    # float assigned
    test.assign_spin(0.8)
    assert test.spin == 0.8


def test_assign_seniority():
    """ Tests CIWavefunction.assign_seniority
    """
    test = CIWavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)))
    # check error
    assert_raises(TypeError, lambda: test.assign_seniority('1'))
    assert_raises(TypeError, lambda: test.assign_seniority(1.0))
    assert_raises(TypeError, lambda: test.assign_seniority(long(1)))
    # None
    test.assign_seniority(None)
    assert test.seniority is None
    # Int assigned
    test.assign_seniority(10)
    assert test.seniority == 10


def test_assign_civec():
    """
    Tests CIWavefunction.assign_civec
    """
    test = CIWavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)))
    # check error
    #  not iterable
    assert_raises(TypeError, lambda: test.assign_civec(2))
    #  iterable of not ints
    assert_raises(TypeError, lambda: test.assign_civec((str(i) for i in range(2))))
    assert_raises(TypeError, lambda: test.assign_civec([float(i) for i in range(2)]))
    #  bad electron number
    assert_raises(ValueError, lambda: test.assign_civec([0b1, 0b111]))
    #  bad spin
    test.assign_spin(0.5)
    test.assign_seniority(None)
    assert_raises(ValueError, lambda: test.assign_civec([0b000011, 0b000110]))
    #  bad seniority
    test.assign_spin(None)
    test.assign_seniority(0)
    assert_raises(ValueError, test.assign_civec, [0b000011, 0b000110])
    #  bad spin and seniority
    test.assign_spin(1)
    test.assign_seniority(0)
    assert_raises(ValueError, test.assign_civec, [0b000011, 0b000110, 0b110000, 0b001001, 0b000101])

    test = CIWavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)))
    # None assigned
    del test.civec
    test.assign_civec()
    print([bin(i) for i in test.civec])

    assert test.civec == (0b001001, 0b001010, 0b011000, 0b001100, 0b101000, 0b000011, 0b010001,
                          0b000101, 0b100001, 0b010010, 0b000110, 0b100010, 0b010100, 0b110000,
                          0b100100)
    del test.civec
    test.assign_civec(None)
    assert test.civec == (0b001001, 0b001010, 0b011000, 0b001100, 0b101000, 0b000011, 0b010001,
                          0b000101, 0b100001, 0b010010, 0b000110, 0b100010, 0b010100, 0b110000,
                          0b100100)
    # tuple assigned
    del test.civec
    test.assign_civec((0b0011,))
    assert test.civec == (0b0011,)
    # list assigned
    del test.civec
    test.assign_civec([0b1100, ])
    assert test.civec == (0b1100,)
    # generator assigned
    del test.civec
    test.assign_civec((i for i in [0b1001]))
    assert test.civec == (0b1001,)
    # repeated elements
    # NOTE: no check for repeated elements
    del test.civec
    test.assign_civec([0b0101, ]*20)
    assert test.civec == (0b0101, )*20
    # spin
    test = CIWavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)), spin=1)
    test.assign_civec([0b000011, 0b000110, 0b110000, 0b001001, 0b000101])
    assert test.civec == (0b000011, 0b000110, 0b000101)
    # seniority
    test = CIWavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)), seniority=0)
    test.assign_civec([0b000011, 0b000110, 0b110000, 0b001001, 0b000101])
    assert test.civec == (0b001001, )


def test_assign_excs():
    """ Tests CIWavefunction.assign_excs
    """
    test = CIWavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)))
    # check error
    #  string
    assert_raises(TypeError, lambda: test.assign_excs(excs='0'))
    #  dictionary
    assert_raises(TypeError, lambda: test.assign_excs(excs={0:0}))
    #  generators
    assert_raises(TypeError, lambda: test.assign_excs(excs=(0 for i in range(4))))
    #  list of floats
    assert_raises(TypeError, lambda: test.assign_excs(excs=[0.0, 1.0]))
    #  bad excitation levels
    assert_raises(ValueError, lambda: test.assign_excs(excs=[-1]))
    assert_raises(ValueError, lambda: test.assign_excs(excs=[15]))
    # assign
    test.assign_excs(excs=None)
    assert test.dict_exc_index == {0:0}
    test.assign_excs(excs=[1])
    assert test.dict_exc_index == {1:0}
    test.assign_excs(excs=(1, 3))
    assert test.dict_exc_index == {1:0, 3:1}
    test.assign_excs(excs=(3, 1))
    assert test.dict_exc_index == {3:0, 1:1}


def test_get_energy():
    """
    Tests CIWavefunction.get_energy
    """
    # check
    test = CIWavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)), excs=[0])
    assert_raises(ValueError, lambda: test.get_energy(exc_lvl=-1))
    assert_raises(ValueError, lambda: test.get_energy(exc_lvl=2))

    # without nuclear repulsion
    test = CIWavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)), excs=[0, 2, 1])
    test.energies = np.arange(3)
    assert test.get_energy(exc_lvl=0) == 0
    assert test.get_energy(exc_lvl=2) == 1
    assert test.get_energy(exc_lvl=1) == 2

    # wit nuclear repulsion
    test = CIWavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)), nuc_nuc=2.4,
                              excs=[0, 2, 1])
    test.energies = np.arange(3)
    assert test.get_energy(include_nuc=True, exc_lvl=0) == 2.4
    assert test.get_energy(include_nuc=True, exc_lvl=2) == 3.4
    assert test.get_energy(include_nuc=True, exc_lvl=1) == 4.4
    assert test.get_energy(include_nuc=False, exc_lvl=0) == 0
    assert test.get_energy(include_nuc=False, exc_lvl=2) == 1
    assert test.get_energy(include_nuc=False, exc_lvl=1) == 2

def test_compute_density_matrix():
    """ Tests wfns.wavefunction.ci_wavefunction.compute_density_matrix
    """
    test = CIWavefunction(2, np.ones((2, 2)), np.ones((2, 2, 2, 2)), excs=[0],
                              orbtype='restricted', civec=[0b0101, 0b1001, 0b0110, 0b1010])
    test.sd_coeffs[:, 0] = np.array([0.993594152, 0.0, 0.0, -0.113007352])
    one_density, two_density = test.compute_density_matrix(exc_lvl=0, is_chemist_notation=False,
                                                           val_threshold=0)
    # reference
    ref_one_density = np.array([[0.197446E+01, -0.163909E-14],
                                [-0.163909E-14, 0.255413E-01]])
    ref_two_density = np.array([[[[1.97445868, 0], [0, -0.22456689]],
                                 [[0, 0], [0, 0]]],
                                [[[0, 0], [0, 0]],
                                 [[-0.22456689, 0], [0, 0.02554132]]]])
    # compare
    assert np.allclose(one_density[0], ref_one_density)
    assert np.allclose(two_density[0], ref_two_density)
# TODO: add test for to_proj (once proj_wavefunction is finished)

def test_compute_ci_matrix():
    """ Tests wfns.wavefunction.ci_wavefunction.compute_ci_matrix
    """
    test = CIWavefunction(2, np.ones((2, 2)), np.ones((2, 2, 2, 2)), excs=[0],
                              orbtype='restricted', civec=[0b0101, 0b1001, 0b0110, 0b1010])
    assert np.allclose(test.compute_ci_matrix(), ci_matrix(test.one_int, test.two_int, test.civec,
                                                           test.dtype, test.orbtype))
