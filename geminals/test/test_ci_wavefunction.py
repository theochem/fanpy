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
