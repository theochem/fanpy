"""Test wfns.objective.twosided_energy."""
from nose.tools import assert_raises
from wfns.objective.twosided_energy import TwoSidedEnergy
from wfns.wavefunction.ci.ci_wavefunction import CIWavefunction


class TestTwoSidedEnergy(TwoSidedEnergy):
    def __init__(self):
        pass


def test_twosided_energy_assign_pspacess():
    """Test TwoSidedEnergy.assign_pspacess."""
    test = TestTwoSidedEnergy()
    test.wfn = CIWavefunction(2, 4)
    # default pspace_l
    test.assign_pspaces(pspace_l=None, pspace_r=(0b0101, ), pspace_n=[0b0101])
    assert test._pspace_l is None
    assert test._pspace_r == (0b0101, )
    assert test._pspace_n == [0b0101]
    # default pspace_r
    test.assign_pspaces(pspace_l=[0b0101], pspace_r=None, pspace_n=(0b0101, ))
    assert test._pspace_l == [0b0101]
    assert test._pspace_r is None
    assert test._pspace_n == (0b0101, )
    # default pspace_n
    test.assign_pspaces(pspace_l=[0b0101], pspace_r=(0b0101, ), pspace_n=None)
    assert test._pspace_l == [0b0101]
    assert test._pspace_r == (0b0101, )
    assert test._pspace_n is None
    # error checking
    assert_raises(TypeError, test.assign_pspaces, pspace_l=set([0b0101, 0b1010]))
    assert_raises(ValueError, test.assign_pspaces, pspace_r=[0b1101])
    assert_raises(ValueError, test.assign_pspaces, pspace_r=[0b10001])
    assert_raises(ValueError, test.assign_pspaces, pspace_n=[CIWavefunction(3, 4)])
    assert_raises(ValueError, test.assign_pspaces, pspace_n=[CIWavefunction(2, 6)])
    assert_raises(TypeError, test.assign_pspaces, pspace_n=['0101'])


def test_twosided_energy_pspaces():
    """Test TwoSidedEnergy.pspace_l,  TwoSidedEnergy.pspace_r, and  TwoSidedEnergy.pspace_n."""
    test = TestTwoSidedEnergy()
    test.wfn = CIWavefunction(2, 4)
    test.assign_pspaces(pspace_l=[0b1001], pspace_r=[0b1010], pspace_n=[0b0101])
    assert list(test.pspace_l) == [0b1001]
    assert list(test.pspace_r) == [0b1010]
    assert list(test.pspace_n) == [0b0101]
    test.assign_pspaces(pspace_l=None, pspace_r=[0b1010], pspace_n=[0b0101])
    assert list(test.pspace_l) == [0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010]
    assert list(test.pspace_r) == [0b1010]
    assert list(test.pspace_n) == [0b0101]
    test.assign_pspaces(pspace_l=[0b1001], pspace_r=None, pspace_n=[0b0101])
    assert list(test.pspace_l) == [0b1001]
    assert list(test.pspace_r) == [0b1001]
    assert list(test.pspace_n) == [0b0101]
    test.assign_pspaces(pspace_l=[0b1001], pspace_r=[0b1010], pspace_n=None)
    assert list(test.pspace_l) == [0b1001]
    assert list(test.pspace_r) == [0b1010]
    assert list(test.pspace_n) == [0b1001]
