"""Test wfns.objective.onesided_energy."""
from nose.tools import assert_raises
from wfns.objective.onesided_energy import OneSidedEnergy
from wfns.wavefunction.ci.ci_wavefunction import CIWavefunction


class TestOneSidedEnergy(OneSidedEnergy):
    def __init__(self):
        pass


def test_onesided_energy_assign_pspaces():
    """Test OneSidedEnergy.assign_pspaces."""
    test = TestOneSidedEnergy()
    test.wfn = CIWavefunction(2, 4)

    test.assign_pspace(pspace=None)
    assert test._pspace is None

    test.assign_pspace(pspace=[0b0101])
    assert test._pspace == (0b0101, )
    test.assign_pspace(pspace=(0b0101, ))
    assert test._pspace == (0b0101, )

    assert_raises(TypeError, test.assign_pspace, pspace=set([0b0101, 0b1010]))
    assert_raises(ValueError, test.assign_pspace, pspace=[0b1101])
    assert_raises(ValueError, test.assign_pspace, pspace=[0b10001])
    assert_raises(ValueError, test.assign_pspace, pspace=[CIWavefunction(3, 4)])
    assert_raises(ValueError, test.assign_pspace, pspace=[CIWavefunction(2, 6)])
    assert_raises(TypeError, test.assign_pspace, pspace=['0101'])


def test_onesided_energy_pspace():
    """Test OneSidedEnergy.pspace."""
    test = TestOneSidedEnergy()
    test.wfn = CIWavefunction(2, 4)
    test.assign_pspace(pspace=[0b1001])
    assert list(test.pspace) == [0b1001]
    test.assign_pspace(pspace=None)
    assert list(test.pspace) == [0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010]
