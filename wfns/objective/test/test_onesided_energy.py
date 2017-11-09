"""Test wfns.objective.onesided_energy."""
from nose.tools import assert_raises
from wfns.objective.onesided_energy import OneSidedEnergy
from wfns.wavefunction.ci.ci_wavefunction import CIWavefunction


class TestOneSidedEnergy(OneSidedEnergy):
    def __init__(self):
        pass


def test_onesided_energy_assign_refwfn():
    """Test OneSidedEnergy.assign_refwfn."""
    test = TestOneSidedEnergy()
    test.wfn = CIWavefunction(2, 4)

    test.assign_refwfn(refwfn=None)
    assert list(test.refwfn) == [0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010]

    test.assign_refwfn(refwfn=[0b0101])
    assert test.refwfn == (0b0101, )
    test.assign_refwfn(refwfn=(0b0101, ))
    assert test.refwfn == (0b0101, )
    ciwfn = CIWavefunction(2, 4)
    test.assign_refwfn(refwfn=ciwfn)
    assert test.refwfn == ciwfn

    assert_raises(TypeError, test.assign_refwfn, refwfn=set([0b0101, 0b1010]))
    assert_raises(ValueError, test.assign_refwfn, refwfn=[0b1101])
    assert_raises(ValueError, test.assign_refwfn, refwfn=[0b10001])
    assert_raises(ValueError, test.assign_refwfn, refwfn=CIWavefunction(3, 4))
    assert_raises(ValueError, test.assign_refwfn, refwfn=CIWavefunction(2, 6))
    assert_raises(TypeError, test.assign_refwfn, refwfn=['0101'])
