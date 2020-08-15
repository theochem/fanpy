"""Test fanpy.wavefunction.composite.lincomb."""
import types

import numpy as np
import pytest
from utils import skip_init
from fanpy.wfn.base import BaseWavefunction
from fanpy.wfn.composite.lincomb import LinearCombinationWavefunction
from fanpy.wfn.ci.base import CIWavefunction


class TempWavefunction(BaseWavefunction):
    """Base wavefunction that bypasses abstract class structure."""

    _spin = None
    _seniority = None

    def get_overlap(self):
        """Do nothing."""
        pass

    @property
    def spin(self):
        """Return the spin of the wavefunction."""
        return self._spin

    @property
    def seniority(self):
        """Return the seniority of the wavefunction."""
        return self._seniority

    def assign_params(self, params=None, add_noise=False):
        """Assign the parameters of the wavefunction."""
        if params is None:
            params = np.identity(10)
        super().assign_params(params=params, add_noise=add_noise)


def test_assign_wfns():
    """Test LinearCombinationWavefunction.assign_wfns."""
    test_wfn = TempWavefunction(4, 10)
    test = skip_init(LinearCombinationWavefunction)
    with pytest.raises(TypeError):
        LinearCombinationWavefunction.assign_wfns(test, (1, test_wfn))
    with pytest.raises(TypeError):
        LinearCombinationWavefunction.assign_wfns(test, (test_wfn, 2))
    test.nelec = 4
    with pytest.raises(ValueError):
        LinearCombinationWavefunction.assign_wfns(test, (test_wfn, TempWavefunction(5, 10)))
    test.memory = np.inf
    with pytest.raises(ValueError):
        LinearCombinationWavefunction.assign_wfns(
            test, (test_wfn, TempWavefunction(4, 10, memory="2gb"))
        )
    with pytest.raises(ValueError):
        LinearCombinationWavefunction.assign_wfns(test, (test_wfn,))

    with pytest.raises(ValueError):
        LinearCombinationWavefunction.assign_wfns(test, (test_wfn, test_wfn))
    # NOTE: wavefunctions with different numbers of spin orbitals are allowed
    LinearCombinationWavefunction.assign_wfns(test, (test_wfn, TempWavefunction(4, 12)))
    assert test.wfns[0].nelec == 4
    assert test.wfns[0].nspin == 10
    assert test.wfns[1].nelec == 4
    assert test.wfns[1].nspin == 12


def test_spin():
    """Test LinearCombinationWavefunction.spin."""
    test_wfn = TempWavefunction(4, 10)
    test_wfn._spin = 3
    test_wfn2 = TempWavefunction(4, 10)
    test_wfn2._spin = 3
    test = LinearCombinationWavefunction(4, 10, (test_wfn, test_wfn2))
    assert test.spin == 3
    test = LinearCombinationWavefunction(4, 10, (test_wfn, TempWavefunction(4, 10)))
    assert test.spin is None


def test_seniority():
    """Test LinearCombinationWavefunction.seniority."""
    test_wfn = TempWavefunction(4, 10)
    test_wfn._seniority = 3
    test_wfn2 = TempWavefunction(4, 10)
    test_wfn2._seniority = 3
    test = LinearCombinationWavefunction(4, 10, (test_wfn, test_wfn2))
    assert test.seniority == 3
    test = LinearCombinationWavefunction(4, 10, (test_wfn, TempWavefunction(4, 10)))
    assert test.seniority is None


def test_default_params():
    """Test LinearCombinationWavefunction.params default."""
    test_wfn = TempWavefunction(4, 10)
    test = LinearCombinationWavefunction(4, 10, (test_wfn, TempWavefunction(4, 10)))
    assert np.allclose(test.params, np.array([1, 0]))
    test = LinearCombinationWavefunction(
        4, 10, (test_wfn, TempWavefunction(4, 10), TempWavefunction(4, 10))
    )
    assert np.allclose(test.params, np.array([1, 0, 0]))


# TODO: test deriv functionality
def test_get_overlap():
    """Test LinearCombinationWavefunction.get_overlap.

    Make simple CI wavefunction as a demonstration.

    """
    test_wfn_1 = TempWavefunction(4, 10)
    test_wfn_1.params = np.array([0.9])
    test_wfn_2 = TempWavefunction(4, 10)
    test_wfn_2.params = np.array([0.1])

    def olp_one(self, sd, deriv=None):
        """Overlap of wavefunction 1."""
        if sd == 0b0101:
            if deriv:
                return np.array([1])
            return self.params[0]
        else:
            if deriv:
                return np.array([2])
            return 0.0

    def olp_two(self, sd, deriv=None):
        """Overlap of wavefunction 2."""
        if sd == 0b1010:
            return self.params[0]
        else:
            return 0.0

    test_wfn_1.get_overlap = types.MethodType(olp_one, test_wfn_1)
    test_wfn_2.get_overlap = types.MethodType(olp_two, test_wfn_2)

    test = LinearCombinationWavefunction(
        4, 10, (test_wfn_1, test_wfn_2), params=np.array([0.7, 0.3])
    )
    assert test.get_overlap(0b0101) == 0.7 * test_wfn_1.params[0] + 0.3 * 0
    assert test.get_overlap(0b1010) == 0.7 * 0 + 0.3 * test_wfn_2.params[0]

    with pytest.raises(TypeError):
        test.get_overlap(0b0101, deriv=np.array([0]))
    with pytest.raises(TypeError):
        test.get_overlap(0b0101, deriv=(test.wfns[0], np.array([0.0])))
    with pytest.raises(TypeError):
        test.get_overlap(0b0101, deriv=(test.wfns[0], np.array([[0]])))
    with pytest.raises(ValueError):
        test.get_overlap(0b0101, deriv=(TempWavefunction(2, 4), np.array([0])))
    with pytest.raises(ValueError):
        test.get_overlap(0b0101, deriv=(test, np.array([-1])))
    with pytest.raises(ValueError):
        test.get_overlap(0b0101, deriv=(test, np.array([test.nparams])))

    assert np.allclose(
        test.get_overlap(0b0101, deriv=(test.wfns[0], np.array([0]))),
        test.get_overlap(0b0101) * np.array([1]),
    )
    assert np.allclose(
        test.get_overlap(0b0101, deriv=(test, np.array([0, 1]))),
        np.array([test.wfns[0].get_overlap(0b0101), test.wfns[1].get_overlap(0b0101)])
    )


def test_save_params(tmp_path):
    """Test LinearCombinationWavefunction.save_params."""
    test_wfn_1 = CIWavefunction(3, 6)
    test_wfn_1.assign_params(np.random.rand(test_wfn_1.nparams))
    test_wfn_2 = CIWavefunction(3, 6)
    test_wfn_2.assign_params(np.random.rand(test_wfn_2.nparams))
    wfn = LinearCombinationWavefunction(3, 6, [test_wfn_1, test_wfn_2], params=np.random.rand(2))
    wfn.save_params(str(tmp_path / "temp.npy"))
    assert np.allclose(np.load(str(tmp_path / "temp.npy")), wfn.params)
    assert np.allclose(np.load(str(tmp_path / "temp_CIWavefunction1.npy")), wfn.wfns[0].params)
    assert np.allclose(np.load(str(tmp_path / "temp_CIWavefunction2.npy")), wfn.wfns[1].params)

    test_wfn_1 = CIWavefunction(3, 6)
    test_wfn_1.assign_params(np.random.rand(test_wfn_1.nparams))
    test_wfn_2 = TempWavefunction(3, 6)
    test_wfn_2.assign_params(np.random.rand(10))
    wfn = LinearCombinationWavefunction(3, 6, [test_wfn_1, test_wfn_2], params=np.random.rand(2))
    wfn.save_params(str(tmp_path / "temp.npy"))
    assert np.allclose(np.load(str(tmp_path / "temp.npy")), wfn.params)
    assert np.allclose(np.load(str(tmp_path / "temp_CIWavefunction.npy")), wfn.wfns[0].params)
    assert np.allclose(np.load(str(tmp_path / "temp_TempWavefunction.npy")), wfn.wfns[1].params)
