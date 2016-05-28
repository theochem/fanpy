from __future__ import absolute_import, division, print_function
from nose.tools import assert_raises

import sys
sys.path.append('../')
from ci_wavefunction import CIWavefunction
import numpy as np

class TestCIWavefunction(CIWavefunction):
    # overwrite to stop initialization
    def __init__(self):
        pass

    @property
    def _nci(self):
        return 4

    def compute_civec(self):
        return [0b1111, 0b10111, 0b11011, 0b11101]

    def compute_ci_matrix(self):
        pass

def test_assign_nci():
    """
    Tests CIWavefunction.assign_nci
    """
    test = TestCIWavefunction()
    # None assigned
    test.nci = None
    test.assign_nci()
    assert test.nci == 4
    test.nci = None
    test.assign_nci(None)
    assert test.nci == 4
    # Int assigned
    test.nci = None
    test.assign_nci(10)
    assert test.nci == 10
    # Other assigned
    assert_raises(TypeError, lambda:test.assign_nci('123'))

def test_assign_civec():
    """
    Tests CIWavefunction.assign_civec
    """
    test = TestCIWavefunction()
    # None assigned
    test.civec = None
    test.assign_civec()
    assert test.civec == (0b1111, 0b10111, 0b11011, 0b11101)
    test.civec = None
    test.assign_civec(None)
    assert test.civec == (0b1111, 0b10111, 0b11011, 0b11101)
    # tuple assigned
    test.civec = None
    test.assign_civec((0b1111,))
    assert test.civec == (0b1111,)
    # list assigned
    test.civec = None
    test.assign_civec([0b1111,])
    assert test.civec == (0b1111,)
    # Other assigned
    assert_raises(TypeError, lambda:test.assign_civec(0b1111))
    assert_raises(TypeError, lambda:test.assign_civec((0b1111)))
    assert_raises(TypeError, lambda:test.assign_civec('123'))

def test_dict_sd_coeff():
    """
    Tests CIWavefunction.dict_sd_coeff
    """
    test = TestCIWavefunction()
    test.civec = [0b1111, 0b110011]
    test.sd_coeffs = np.arange(6).reshape(2,3)
    # ground state
    sd_coeff = test.dict_sd_coeff()
    assert sd_coeff[0b1111] == 0
    assert sd_coeff[0b110011] == 3
    sd_coeff = test.dict_sd_coeff(exc_lvl=0)
    assert sd_coeff[0b1111] == 0
    assert sd_coeff[0b110011] == 3
    # 1st excited
    sd_coeff = test.dict_sd_coeff(exc_lvl=1)
    assert sd_coeff[0b1111] == 1
    assert sd_coeff[0b110011] == 4
    # 2nd excited
    sd_coeff = test.dict_sd_coeff(exc_lvl=2)
    assert sd_coeff[0b1111] == 2
    assert sd_coeff[0b110011] == 5
    # bad excitation
    assert_raises(TypeError, lambda:test.dict_sd_coeff(exc_lvl='2'))
    assert_raises(TypeError, lambda:test.dict_sd_coeff(exc_lvl=2.0))
    assert_raises(ValueError, lambda:test.dict_sd_coeff(exc_lvl=-2))
