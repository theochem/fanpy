"""Test wfns.wavefunction.geminals.rank2_geminal.RankTwoGeminal."""
import numpy as np
import pytest
from wfns.wfn.geminal.apig import APIG
from wfns.wfn.geminal.base import BaseGeminal
from wfns.wfn.geminal.rank2_approx import full_to_rank2, RankTwoApprox


class RankTwoGeminal(RankTwoApprox, BaseGeminal):
    """RankTwoGeminal that skips initialization."""

    def __init__(self):
        """Do nothing."""
        self._cache_fns = {}

    def generate_possible_orbpairs(self, occ_indices):
        """Generate orbital pairing scheme."""
        yield from APIG.generate_possible_orbpairs(self, occ_indices)


def test_rank2_geminal_params_from_full():
    """Test full_to_rank2."""
    # noise generated using the following code
    # noise = 0.001*np.random.rand(4, 10)
    # but is specified explicitly b/c the test often fails
    noise = np.array(
        [
            [
                4.51870434e-04,
                9.74649328e-04,
                6.35243079e-04,
                2.34541705e-04,
                8.38506005e-05,
                7.42212732e-05,
                7.78475326e-04,
                7.90466146e-05,
                2.41744423e-04,
                5.38114863e-05,
            ],
            [
                8.03578883e-04,
                4.46145764e-04,
                5.06811058e-06,
                3.53330804e-04,
                3.11343409e-04,
                4.88073808e-04,
                7.13388210e-04,
                1.68058131e-04,
                6.01961235e-04,
                2.54699230e-04,
            ],
            [
                2.17332952e-04,
                3.32899445e-04,
                8.68823215e-04,
                9.61565590e-04,
                2.83463946e-04,
                9.62409947e-05,
                2.15738937e-04,
                8.40910378e-04,
                8.26679340e-05,
                7.85761618e-04,
            ],
            [
                3.12310560e-04,
                7.53233950e-04,
                1.29859605e-05,
                8.96259460e-04,
                2.06408799e-04,
                8.05341015e-04,
                5.39334471e-04,
                4.67358563e-04,
                2.76985551e-04,
                9.44197774e-04,
            ],
        ]
    )
    fullrank_params = np.eye(4, 10) + noise
    test = full_to_rank2(fullrank_params)
    assert np.allclose(
        fullrank_params, test[14:24] / (test[:4, np.newaxis] - test[4:14]), atol=0.1, rtol=0
    )

    fullrank_params = np.array(
        [
            [
                1.033593181822e00,
                3.130903350751e-04,
                -4.321247538977e-03,
                -1.767251395337e-03,
                -1.769214953534e-03,
                -1.169729179981e-03,
            ],
            [
                -5.327889357199e-01,
                9.602580629349e-01,
                -1.139839360648e-02,
                -2.858698370621e-02,
                -2.878270043699e-02,
                -1.129324573431e-01,
            ],
        ]
    )
    test = full_to_rank2(fullrank_params)
    assert np.allclose(
        fullrank_params, test[8:14] / (test[:2, np.newaxis] - test[2:8]), atol=0.1, rtol=0
    )


@pytest.mark.skip(reason="Test does not always pass (depends on random noise).")
def test_rank2_geminal_default_params():
    """Test RankTwoGeminal.default_params."""
    np.random.seed(424242)

    test = RankTwoGeminal()
    test.assign_nspin(10)
    test.assign_orbpairs()
    test.assign_nelec(2)
    # ngem 1
    test.assign_ngem(1)
    test.assign_params()
    template = test.params
    lambdas = template[:1, np.newaxis]
    epsilons = template[1:46]
    zetas = template[46:]
    answer = np.zeros(45)
    answer[test.dict_orbpair_ind[(0, 5)]] = 1
    assert np.allclose(zetas / (lambdas - epsilons), answer, atol=0.001, rtol=0)
    # ngem 2
    test.assign_ngem(2)
    test.assign_params()
    template = test.params
    lambdas = template[:2, np.newaxis]
    epsilons = template[2:47]
    zetas = template[47:]
    answer = np.zeros((2, 45))
    answer[0, test.dict_orbpair_ind[(0, 5)]] = 1
    answer[1, test.dict_orbpair_ind[(1, 6)]] = 1
    assert np.allclose(zetas / (lambdas - epsilons), answer, atol=0.001, rtol=0)
    # ngem 3
    test.assign_ngem(3)
    test.assign_params()
    template = test.params
    lambdas = template[:3, np.newaxis]
    epsilons = template[3:48]
    zetas = template[48:]
    answer = np.zeros((3, 45))
    answer[0, test.dict_orbpair_ind[(0, 5)]] = 1
    answer[1, test.dict_orbpair_ind[(1, 6)]] = 1
    answer[2, test.dict_orbpair_ind[(2, 7)]] = 1
    assert np.allclose(zetas / (lambdas - epsilons), answer, atol=0.01, rtol=0)


def test_rank2_geminal_assign_params():
    """Test RankTwoGeminal.assign_params."""
    test = RankTwoGeminal()
    test.assign_nspin(8)
    test.assign_nelec(4)
    test.assign_ngem(2)
    test.assign_orbpairs([(0, 4), (1, 5), (2, 6), (3, 7)])
    # check assignment
    test.assign_params(np.array([1, 2, 0, 0, 0, 0, 1, 1, 1, 1.0]))
    assert np.allclose(test.params, np.array([1, 2, 0, 0, 0, 0, 1, 1, 1, 1.0]))
    # check default
    test.assign_params(None)
    lambdas = test.params[:2, np.newaxis]
    epsilons = test.params[2:6]
    zetas = test.params[6:]
    assert np.allclose(zetas / (lambdas - epsilons), np.eye(2, 4), atol=0.001, rtol=0)
    # check error
    with pytest.raises(ValueError):
        test.assign_params(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0]))
    with pytest.raises(ValueError):
        test.assign_params(np.array([1, 2, 1, 9, 9, 9, 0, 0, 0, 0.0]))
    with pytest.raises(ValueError):
        test.assign_params(np.array([1, 2, 9, 1, 9, 9, 0, 0, 0, 0.0]))
    with pytest.raises(ValueError):
        test.assign_params(np.array([1, 2, 9, 9, 1, 9, 0, 0, 0, 0.0]))
    with pytest.raises(ValueError):
        test.assign_params(np.array([1, 2, 9, 9, 9, 1, 0, 0, 0, 0.0]))
    with pytest.raises(ValueError):
        test.assign_params(np.array([1, 2, 2, 9, 9, 9, 0, 0, 0, 0.0]))
    with pytest.raises(ValueError):
        test.assign_params(np.array([1, 2, 9, 2, 9, 9, 0, 0, 0, 0.0]))
    with pytest.raises(ValueError):
        test.assign_params(np.array([1, 2, 9, 9, 2, 9, 0, 0, 0, 0.0]))
    with pytest.raises(ValueError):
        test.assign_params(np.array([1, 2, 9, 9, 9, 2, 0, 0, 0, 0.0]))
    with pytest.raises(NotImplementedError):
        test.assign_params(test)


def test_rank2_geminal_lambdas():
    """Test RankTwoGeminal.lambdas."""
    test = RankTwoGeminal()
    test.assign_nspin(8)
    test.assign_nelec(4)
    test.assign_ngem(2)
    test.assign_orbpairs()
    test.assign_params(np.arange(58, dtype=float))
    assert np.allclose(test.lambdas.flatten(), test.params[:2])


def test_rank2_geminal_epsilons():
    """Test RankTwoGeminal.epsilons."""
    test = RankTwoGeminal()
    test.assign_nspin(8)
    test.assign_nelec(4)
    test.assign_ngem(2)
    test.assign_orbpairs()
    test.assign_params(np.arange(58, dtype=float))
    assert np.allclose(test.epsilons.flatten(), test.params[2:30])


def test_rank2_geminal_zetas():
    """Test RankTwoGeminal.zetas."""
    test = RankTwoGeminal()
    test.assign_nspin(8)
    test.assign_nelec(4)
    test.assign_ngem(2)
    test.assign_orbpairs()
    test.assign_params(np.arange(58, dtype=float))
    assert np.allclose(test.zetas.flatten(), test.params[30:])


def test_rank2_geminal_fullrank_params():
    """Test RankTwoGeminal.fullrank_params."""
    test = RankTwoGeminal()
    test.assign_nspin(8)
    test.assign_nelec(4)
    test.assign_ngem(2)
    test.assign_orbpairs()
    test.assign_params(np.arange(58, dtype=float))
    assert np.allclose(
        test.fullrank_params, test.params[30:] / (test.params[:2, np.newaxis] - test.params[2:30])
    )


def test_rank2_geminal_compute_permanent():
    """Test RankTwoGeminal.compute_permanent."""
    test = RankTwoGeminal()
    test.assign_nspin(8)
    test.assign_orbpairs([(0, 4), (1, 5), (2, 6), (3, 7)])
    # two electrons
    test.assign_nelec(2)
    test.assign_ngem(1)
    test.assign_params(np.arange(1, 10, dtype=float))
    # overlap
    assert np.allclose(test.compute_permanent([0], deriv=None), test.fullrank_params[0, 0])
    assert np.allclose(test.compute_permanent([1], deriv=None), test.fullrank_params[0, 1])
    assert np.allclose(test.compute_permanent([2], deriv=None), test.fullrank_params[0, 2])
    assert np.allclose(test.compute_permanent([3], deriv=None), test.fullrank_params[0, 3])
    # differentiate
    assert np.equal(
        test.compute_permanent([0], deriv=0),
        -test.zetas[0] / (test.lambdas[0] - test.epsilons[0]) ** 2,
    )
    assert np.equal(
        test.compute_permanent([0], deriv=1),
        test.zetas[0] / (test.lambdas[0] - test.epsilons[0]) ** 2,
    )
    assert test.compute_permanent([0], deriv=2) == 0
    assert test.compute_permanent([0], deriv=3) == 0
    assert test.compute_permanent([0], deriv=4) == 0
    assert test.compute_permanent([0], deriv=5) == 1.0 / (test.lambdas[0] - test.epsilons[0])
    assert test.compute_permanent([0], deriv=6) == 0
    assert test.compute_permanent([0], deriv=7) == 0
    assert test.compute_permanent([0], deriv=8) == 0
    with pytest.raises(ValueError):
        test.compute_permanent([0], deriv=99)
    with pytest.raises(ValueError):
        test.compute_permanent([0], deriv=-1)

    # four electrons
    test.assign_nelec(4)
    test.assign_ngem(2)
    test.assign_params(np.arange(1, 11, dtype=float))
    # overlap
    assert np.allclose(
        test.compute_permanent([0, 1], deriv=None),
        (
            test.fullrank_params[0, 0] * test.fullrank_params[1, 1]
            + test.fullrank_params[0, 1] * test.fullrank_params[1, 0]
        ),
    )
    assert np.allclose(
        test.compute_permanent([0, 2], deriv=None),
        (
            test.fullrank_params[0, 0] * test.fullrank_params[1, 2]
            + test.fullrank_params[1, 0] * test.fullrank_params[0, 2]
        ),
    )
    assert np.allclose(
        test.compute_permanent([0, 3], deriv=None),
        (
            test.fullrank_params[0, 0] * test.fullrank_params[1, 3]
            + test.fullrank_params[1, 0] * test.fullrank_params[0, 3]
        ),
    )
    assert np.allclose(
        test.compute_permanent([1, 2], deriv=None),
        (
            test.fullrank_params[0, 1] * test.fullrank_params[1, 2]
            + test.fullrank_params[1, 1] * test.fullrank_params[0, 2]
        ),
    )
    assert np.allclose(
        test.compute_permanent([1, 3], deriv=None),
        (
            test.fullrank_params[0, 1] * test.fullrank_params[1, 3]
            + test.fullrank_params[1, 1] * test.fullrank_params[0, 3]
        ),
    )
    assert np.allclose(
        test.compute_permanent([2, 3], deriv=None),
        (
            test.fullrank_params[0, 2] * test.fullrank_params[1, 3]
            + test.fullrank_params[1, 2] * test.fullrank_params[0, 3]
        ),
    )
    # differentiate
    assert np.allclose(
        test.compute_permanent([0, 1], deriv=0),
        (
            test.fullrank_params[1, 1]
            * (-test.zetas[0] / (test.lambdas[0] - test.epsilons[0]) ** 2)
            + test.fullrank_params[1, 0]
            * (-test.zetas[1] / (test.lambdas[0] - test.epsilons[1]) ** 2)
        ),
    )
    assert np.allclose(
        test.compute_permanent([0, 1], deriv=1),
        (
            test.fullrank_params[0, 1]
            * (-test.zetas[0] / (test.lambdas[1] - test.epsilons[0]) ** 2)
            + test.fullrank_params[0, 0]
            * (-test.zetas[1] / (test.lambdas[1] - test.epsilons[1]) ** 2)
        ),
    )
    assert np.allclose(
        test.compute_permanent([0, 1], deriv=2),
        (
            test.fullrank_params[1, 1] * (test.zetas[0] / (test.lambdas[0] - test.epsilons[0]) ** 2)
            + test.fullrank_params[0, 1]
            * (test.zetas[0] / (test.lambdas[1] - test.epsilons[0]) ** 2)
        ),
    )
    assert np.allclose(
        test.compute_permanent([0, 1], deriv=3),
        (
            test.fullrank_params[1, 0] * (test.zetas[1] / (test.lambdas[0] - test.epsilons[1]) ** 2)
            + test.fullrank_params[0, 0]
            * (test.zetas[1] / (test.lambdas[1] - test.epsilons[1]) ** 2)
        ),
    )
    assert np.allclose(test.compute_permanent([0, 1], deriv=4), 0)
    assert np.allclose(test.compute_permanent([0, 1], deriv=5), 0)
    assert np.allclose(
        test.compute_permanent([0, 1], deriv=6),
        (
            test.fullrank_params[1, 1] * (1.0 / (test.lambdas[0] - test.epsilons[0]))
            + test.fullrank_params[0, 1] * (1.0 / (test.lambdas[1] - test.epsilons[0]))
        ),
    )
    assert np.allclose(
        test.compute_permanent([0, 1], deriv=7),
        (
            test.fullrank_params[1, 0] * (1.0 / (test.lambdas[0] - test.epsilons[1]))
            + test.fullrank_params[0, 0] * (1.0 / (test.lambdas[1] - test.epsilons[1]))
        ),
    )
    assert test.compute_permanent([0, 1], deriv=8) == 0
    with pytest.raises(ValueError):
        test.compute_permanent([0, 1], deriv=99)


def test_rank2_geminal_get_overlap():
    """Test RankTwoGeminal.get_overlap."""
    test = RankTwoGeminal()
    test.assign_nspin(8)
    test.assign_nelec(4)
    test.assign_memory()
    test.assign_ngem(2)
    test.assign_orbpairs([(0, 4), (1, 5), (2, 6), (3, 7)])
    test.assign_params(np.arange(1, 11, dtype=float))
    test.load_cache()
    # check overlap
    assert np.allclose(
        -test.get_overlap(0b00110011),
        (
            test.fullrank_params[0, 0] * test.fullrank_params[1, 1]
            + test.fullrank_params[0, 1] * test.fullrank_params[1, 0]
        ),
    )
    assert np.allclose(
        -test.get_overlap(0b01010101),
        (
            test.fullrank_params[0, 0] * test.fullrank_params[1, 2]
            + test.fullrank_params[1, 0] * test.fullrank_params[0, 2]
        ),
    )
    assert np.allclose(
        -test.get_overlap(0b10011001),
        (
            test.fullrank_params[0, 0] * test.fullrank_params[1, 3]
            + test.fullrank_params[1, 0] * test.fullrank_params[0, 3]
        ),
    )
    assert np.allclose(
        -test.get_overlap(0b01100110),
        (
            test.fullrank_params[0, 1] * test.fullrank_params[1, 2]
            + test.fullrank_params[1, 1] * test.fullrank_params[0, 2]
        ),
    )
    assert np.allclose(
        -test.get_overlap(0b10101010),
        (
            test.fullrank_params[0, 1] * test.fullrank_params[1, 3]
            + test.fullrank_params[1, 1] * test.fullrank_params[0, 3]
        ),
    )
    assert np.allclose(
        -test.get_overlap(0b11001100),
        (
            test.fullrank_params[0, 2] * test.fullrank_params[1, 3]
            + test.fullrank_params[1, 2] * test.fullrank_params[0, 3]
        ),
    )
    # check derivative
    assert np.allclose(
        -test.get_overlap(0b00110011, deriv=np.array([0])),
        (
            test.fullrank_params[1, 1]
            * (-test.zetas[0] / (test.lambdas[0] - test.epsilons[0]) ** 2)
            + test.fullrank_params[1, 0]
            * (-test.zetas[1] / (test.lambdas[0] - test.epsilons[1]) ** 2)
        ),
    )
    assert np.allclose(
        -test.get_overlap(0b00110011, deriv=np.array([1])),
        (
            test.fullrank_params[0, 1]
            * (-test.zetas[0] / (test.lambdas[1] - test.epsilons[0]) ** 2)
            + test.fullrank_params[0, 0]
            * (-test.zetas[1] / (test.lambdas[1] - test.epsilons[1]) ** 2)
        ),
    )
    assert np.allclose(
        -test.get_overlap(0b00110011, deriv=np.array([2])),
        (
            test.fullrank_params[1, 1] * (test.zetas[0] / (test.lambdas[0] - test.epsilons[0]) ** 2)
            + test.fullrank_params[0, 1]
            * (test.zetas[0] / (test.lambdas[1] - test.epsilons[0]) ** 2)
        ),
    )
    assert np.allclose(
        -test.get_overlap(0b00110011, deriv=np.array([3])),
        (
            test.fullrank_params[1, 0] * (test.zetas[1] / (test.lambdas[0] - test.epsilons[1]) ** 2)
            + test.fullrank_params[0, 0]
            * (test.zetas[1] / (test.lambdas[1] - test.epsilons[1]) ** 2)
        ),
    )
    assert np.allclose(-test.get_overlap(0b00110011, deriv=np.array([4])), 0)
    assert np.allclose(-test.get_overlap(0b00110011, deriv=np.array([5])), 0)
    assert np.allclose(
        -test.get_overlap(0b00110011, deriv=np.array([6])),
        (
            test.fullrank_params[1, 1] * (1.0 / (test.lambdas[0] - test.epsilons[0]))
            + test.fullrank_params[0, 1] * (1.0 / (test.lambdas[1] - test.epsilons[0]))
        ),
    )
    assert np.allclose(
        -test.get_overlap(0b00110011, deriv=np.array([7])),
        (
            test.fullrank_params[1, 0] * (1.0 / (test.lambdas[0] - test.epsilons[1]))
            + test.fullrank_params[0, 0] * (1.0 / (test.lambdas[1] - test.epsilons[1]))
        ),
    )
    assert test.get_overlap(0b00110011, deriv=np.array([8])) == 0
