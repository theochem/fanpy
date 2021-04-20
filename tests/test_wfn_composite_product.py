"""Test fanpy.wavefunction.composite.product."""


from fanpy.wfn.ci.base import CIWavefunction
from fanpy.wfn.composite.product import ProductWavefunction

import numpy as np

import pytest


def test_init():
    """Test ProductWavefunction.assign_wfns."""
    wfn1 = CIWavefunction(4, 10)
    wfn2 = CIWavefunction(4, 10)
    wfn = ProductWavefunction([wfn1, wfn2])

    assert wfn.nelec == 4
    assert wfn.nspin == 10
    assert wfn.wfns == [wfn1, wfn2]

    with pytest.raises(TypeError):
        wfn = ProductWavefunction([wfn1, 2])

    wfn1.nelec = 3
    with pytest.raises(ValueError):
        wfn = ProductWavefunction([wfn1, wfn2])

    with pytest.raises(ValueError):
        wfn = ProductWavefunction([wfn1, wfn1])


def test_get_overlap():
    """Test ProductWavefunction.get_overlap."""
    wfn1 = CIWavefunction(4, 10)
    wfn1.assign_params(np.random.rand(wfn1.nparams))
    wfn2 = CIWavefunction(4, 10)
    wfn2.assign_params(np.random.rand(wfn2.nparams))

    test = ProductWavefunction((wfn1, wfn2))
    for sd in wfn1.sds:
        assert test.get_overlap(sd) == (wfn1.params * wfn2.params)[wfn1.dict_sd_index[sd]]

    with pytest.raises(TypeError):
        test.get_overlap(0b0011001010, deriv=np.array([0]))
    with pytest.raises(TypeError):
        test.get_overlap(0b0011001010, deriv=(test.wfns[0], np.array([0.0])))
    with pytest.raises(TypeError):
        test.get_overlap(0b0011001010, deriv=(test.wfns[0], np.array([[0]])))
    with pytest.raises(TypeError):
        test.get_overlap(0b0011001010, deriv=(test.wfns[0], np.array([[0]])))
    with pytest.raises(ValueError):
        test.get_overlap(0b0011001010, deriv=(CIWavefunction(2, 4), np.array([0])))

    assert np.allclose(
        test.get_overlap(0b0001100011, deriv=(test.wfns[0], np.array([0, 1]))),
        np.array([wfn2.get_overlap(0b0001100011), 0]) * np.array([1]),
    )
    assert np.allclose(
        test.get_overlap(0b0001100011, deriv=(test.wfns[1], np.array([0, 1]))),
        np.array([wfn1.get_overlap(0b0001100011), 0]) * np.array([1]),
    )
