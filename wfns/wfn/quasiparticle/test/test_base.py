"""Test BaseQuasiparticle.base."""
import numpy as np
from numpy.testing import assert_raises
from wfns.wfn.quasiparticle.base import BaseQuasiparticle


class TestQuasiparticle(BaseQuasiparticle):
    """Class for testing BaseQuasiparticle."""
    def __init__(self):
        """Dummy initializer."""

    def generate_possible_orbsubsets(self, occ_indices):
        """Generate an orbpair.

        Notes
        -----
        First 8 orbitals must be occupied.

        """
        yield ((0, 1), (2, 3), (5, 6, 7), (4, ))


def test_assign_nquasiparticle():
    """Test BaseQuasiparticle.assign_nquasiparticle."""
    test = TestQuasiparticle()
    assert_raises(NotImplementedError, test.assign_nquasiparticle)
    assert_raises(NotImplementedError, test.assign_nquasiparticle, None)
    assert_raises(TypeError, test.assign_nquasiparticle, '2')
    assert_raises(TypeError, test.assign_nquasiparticle, 2.0)
    assert_raises(ValueError, test.assign_nquasiparticle, -1)
    assert_raises(ValueError, test.assign_nquasiparticle, 0)

    test.assign_nquasiparticle(2)
    assert test.nquasiparticle == 2


def test_assign_orbsubsets():
    """Test BaseQuasiparticle.assign_orbsubsets."""
    test = TestQuasiparticle()
    assert_raises(NotImplementedError, test.assign_orbsubsets)
    assert_raises(NotImplementedError, test.assign_orbsubsets, None)
    assert_raises(TypeError, test.assign_orbsubsets, 12)
    assert_raises(TypeError, test.assign_orbsubsets, [2])
    assert_raises(TypeError, test.assign_orbsubsets, [(2.0,)])
    assert_raises(ValueError, test.assign_orbsubsets, [(2, 2)])
    assert_raises(ValueError, test.assign_orbsubsets, [(2,), (2,)])

    test.assign_orbsubsets([(0,), (1,)])
    assert test.dict_ind_orbsubset == {0: (0,), 1: (1,)}
    assert test.dict_orbsubset_ind == {(0,): 0, (1,): 1}
    assert test.orbsubset_sizes == (1, )

    test.assign_orbsubsets([(0,), (1, 2)])
    assert test.dict_ind_orbsubset == {0: (0,), 1: (1, 2)}
    assert test.dict_orbsubset_ind == {(0,): 0, (1, 2): 1}
    assert test.orbsubset_sizes == (1, 2)


def test_norbsubsets():
    """Test BaseQuasiparticle.norbsubsets."""
    test = TestQuasiparticle()
    test.assign_orbsubsets([(0, 1), (1, 2), (0, 1, 2)])
    assert test.norbsubsets == 3


def test_params_shape():
    """Test BaseQuasiparticle.params_shape."""
    test = TestQuasiparticle()
    test.assign_nquasiparticle(4)
    test.assign_orbsubsets([(0,), (0, 1), (0, 1, 2), (0, 1, 2, 3), (1,)])
    assert test.params_shape == (4, 5)


def test_get_col_ind():
    """Test BaseQuasiparticle.get_col_ind."""
    test = TestQuasiparticle()
    test.assign_orbsubsets([(0,), (0, 1), (0, 1, 2), (0, 1, 2, 3), (1,)])
    assert test.get_col_ind((0, )) == 0
    assert test.get_col_ind((0, 1)) == 1
    assert test.get_col_ind((0, 1, 2)) == 2
    assert test.get_col_ind((0, 1, 2, 3)) == 3
    assert test.get_col_ind((1, )) == 4
    assert_raises(ValueError, test.get_col_ind, (2, ))
    assert_raises(ValueError, test.get_col_ind, [0, ])


def test_get_orbsubset():
    """Test BaseQuasiparticle.get_orbsubset."""
    test = TestQuasiparticle()
    test.assign_orbsubsets([(0,), (0, 1), (0, 1, 2), (0, 1, 2, 3), (1,)])
    assert test.get_orbsubset(0) == (0, )
    assert test.get_orbsubset(1) == (0, 1)
    assert test.get_orbsubset(2) == (0, 1, 2)
    assert test.get_orbsubset(3) == (0, 1, 2, 3)
    assert test.get_orbsubset(4) == (1, )
    assert_raises(ValueError, test.get_orbsubset, 5)
    assert_raises(ValueError, test.get_orbsubset, [0])


def test_template_params():
    """Test BaseQuasiparticle.template_params."""
    test = TestQuasiparticle()
    test.assign_nelec(3)
    test.assign_nspin(10)
    test.assign_dtype(float)
    test.assign_nquasiparticle(4)
    test.assign_orbsubsets([(0, 1), (2, 3), (5, 6, 7), (4,), (1, 3, 5)])
    assert np.allclose(test.template_params, np.eye(4, 5))


def test_assign_params():
    """Test BaseQuasiparticle.assign_params."""
    test = TestQuasiparticle()
    test.assign_nelec(8)
    test.assign_nspin(10)
    test.assign_dtype(float)
    test.assign_nquasiparticle(4)
    test.assign_orbsubsets([(0, 1), (2, 3), (5, 6, 7), (4,), (1, 3, 5)])
    test.assign_params()
    assert np.allclose(test.params, test.template_params)

    test2 = TestQuasiparticle()
    test2.assign_nelec(2)
    assert_raises(ValueError, test.assign_params, test2)
    test2.assign_nelec(8)
    test2.assign_nspin(12)
    assert_raises(ValueError, test.assign_params, test2)
    test2.assign_nspin(10)
    test2.assign_nquasiparticle(10)
    assert_raises(ValueError, test.assign_params, test2)

    test2.assign_nquasiparticle(4)
    test2.assign_orbsubsets([(0, 1), (2, 3), (5, 6, 7), (4,), (1, 3, 5)])
    test2.params = np.arange(20).reshape(4, 5)
    test.assign_params(test2)
    assert np.allclose(test.params, np.arange(20).reshape(4, 5))

    test2.assign_orbsubsets([(0, 1), (2, 3), (5, 6, 8), (4,), (1, 3, 5)])
    test2.params = np.arange(20).reshape(4, 5)
    test.assign_params(test2)
    answer = np.arange(20).reshape(4, 5)
    answer[:, 2] = 0
    assert np.allclose(test.params, answer)


def test_compute_permsum():
    """Test BaseQuasiparticle.compute_permsum."""
    test = TestQuasiparticle()
    test.assign_nquasiparticle(4)
    test.assign_orbsubsets([(0,), (1,), (2,), (3,), (4,), (5,)])
    test.params = np.arange(24).reshape(4, 6)
    # [[ 0  1  2  3]
    #  [ 6  7  8  9]
    #  [12 13 14 15]
    #  [18 19 20 21]]
    answer = ((0*7+1*6)*(14*21-15*20) + (0*8+2*6)*(13*21-15*19) + (0*9+3*6)*(13*20-14*19) +
              (1*8+2*7)*(12*21-15*18) + (1*9+3*7)*(12*20-14*18) + (2*9+3*8)*(12*19-13*18))
    assert np.allclose(test.compute_permsum(2, [0, 1, 2, 3], row_inds=None, deriv=None), answer)
    # NOTE: permutational sum is not invariant with respect to interchange of columns
    answer = ((0*7+1*6)*(15*20-14*21) + (0*8+2*6)*(13*21-15*19) + (0*9+3*6)*(13*20-14*19) +
              (1*8+2*7)*(12*21-15*18) + (1*9+3*7)*(12*20-14*18) + (2*9+3*8)*(12*19-13*18))
    assert np.allclose(test.compute_permsum(2, [0, 1, 3, 2], row_inds=None, deriv=None), answer)

    answer = (0 * (7*14*21 - 7*15*20 - 8*13*21 + 8*15*19 + 9*13*20 - 9*14*19) +
              1 * (6*14*21 - 6*15*20 - 8*12*21 + 8*15*18 + 9*12*20 - 9*14*18) +
              2 * (7*12*21 - 7*15*18 - 6*13*21 + 6*15*19 + 9*13*18 - 9*12*19) +
              3 * (7*14*18 - 7*12*20 - 8*13*18 + 8*12*19 + 6*13*20 - 6*14*19))
    assert np.allclose(test.compute_permsum(1, [0, 1, 2, 3], row_inds=None, deriv=None), answer)

    answer = ((0*7*14 + 0*8*13 + 1*6*14 + 1*8*12 + 2*6*13 + 2*7*12) * 21 +
              (0*7*15 + 0*9*13 + 1*6*15 + 1*9*12 + 3*6*13 + 3*7*12) * 20 +
              (0*9*14 + 0*8*15 + 3*6*14 + 3*8*12 + 2*6*15 + 2*9*12) * 19 +
              (3*7*14 + 3*8*13 + 1*9*14 + 1*8*15 + 2*9*13 + 2*7*15) * 18)
    assert np.allclose(test.compute_permsum(3, [0, 1, 2, 3], row_inds=None, deriv=None), answer)

    answer = (0*7*14*21 - 0*7*15*20 - 0*8*13*21 + 0*8*15*19 +
              0*9*13*20 - 0*9*14*19 - 1*6*14*21 + 1*6*15*20 +
              1*8*12*21 - 1*8*15*18 - 1*9*12*20 + 1*9*14*18 +
              2*6*13*21 - 2*6*15*19 - 2*7*12*21 + 2*7*15*18 +
              2*9*12*19 - 2*9*13*18 - 3*6*13*20 + 3*6*14*19 +
              3*7*12*20 - 3*7*14*18 - 3*8*12*19 + 3*8*13*18)
    assert np.allclose(test.compute_permsum(0, [0, 1, 2, 3], row_inds=None, deriv=None), answer)

    answer = (0*7*14*21 + 0*7*15*20 + 0*8*13*21 + 0*8*15*19 +
              0*9*13*20 + 0*9*14*19 + 1*6*14*21 + 1*6*15*20 +
              1*8*12*21 + 1*8*15*18 + 1*9*12*20 + 1*9*14*18 +
              2*6*13*21 + 2*6*15*19 + 2*7*12*21 + 2*7*15*18 +
              2*9*12*19 + 2*9*13*18 + 3*6*13*20 + 3*6*14*19 +
              3*7*12*20 + 3*7*14*18 + 3*8*12*19 + 3*8*13*18)
    assert np.allclose(test.compute_permsum(4, [0, 1, 2, 3], row_inds=None, deriv=None), answer)

    assert np.allclose(test.compute_permsum(2, [0, 1, 2, 3], row_inds=None, deriv=4), 0)
    answer = 7*(14*21-15*20) + 8*(13*21-15*19) + 9*(13*20-14*19)

    assert np.allclose(test.compute_permsum(2, [0, 1, 2, 3], row_inds=None, deriv=0), answer)
    answer = 7*14*21 - 7*15*20 - 8*13*21 + 8*15*19 + 9*13*20 - 9*14*19

    assert np.allclose(test.compute_permsum(1, [0, 1, 2, 3], row_inds=None, deriv=0), answer)

    assert np.allclose(test.compute_permsum(1, [0], row_inds=[0], deriv=0), 1)
    assert np.allclose(test.compute_permsum(0, [0], row_inds=[0], deriv=0), 1)

    # [[12 13]
    #  [18 19]]
    answer = 12*19 + 13*18
    assert np.allclose(test.compute_permsum(2, [0, 1], row_inds=[2, 3], deriv=None), answer)


def test_process_subsets():
    """Test BaseQuasiparticle._process_subsets."""
    test = TestQuasiparticle()
    test.dict_orbsubset_ind = {(0, 1): 0, (2, 3): 1, (4, 5, 6): 2, (7, ): 3}
    assert np.allclose(test._process_subsets([(0, 1), (2, 3), (7, )])[0], [0, 1, 3])
    assert test._process_subsets([(0, 1), (2, 3), (7, )])[1] == [(0, 1), (2, 3)]
    assert test._process_subsets([(0, 1), (2, 3), (7, )])[2] == [(7, )]

    assert np.allclose(test._process_subsets([(2, 3), (0, 1)])[0], [1, 0])
    assert test._process_subsets([(2, 3), (0, 1)])[1] == [(2, 3), (0, 1)]
    assert test._process_subsets([(2, 3), (0, 1)])[2] == []

    assert np.allclose(test._process_subsets([(0, 1), (7, ), (2, 3)])[0], [0, 1, 3])
    assert np.allclose(test._process_subsets([(7, ), (0, 1), (2, 3)])[0], [0, 1, 3])
    assert np.allclose(test._process_subsets([(7, ), (2, 3), (0, 1)])[0], [1, 0, 3])


def test_olp_deriv():
    """Test BaseQuasiparticle._olp_deriv."""
    test = TestQuasiparticle()
    test.assign_nquasiparticle(4)
    test.params = np.arange(24).reshape(4, 6)

    test.assign_orbsubsets([(0, 1), (2, 3), (5, 6, 7), (4,), (1, 3, 5), (2, 4, 6, 8)])
    assert np.allclose(-test._olp_deriv(0b11111111, None),
                       test.compute_permsum(2, [0, 1, 2, 3], deriv=None))
    assert np.allclose(-test._olp_deriv(0b11111111, 0),
                       test.compute_permsum(2, [0, 1, 2, 3], deriv=0))

    # Note the ordering of the orbital subsets and column indices
    test.assign_orbsubsets([(0, 1), (2, 3), (4,), (5, 6, 7), (1, 3, 5), (2, 4, 6, 8)])
    assert np.allclose(test._olp_deriv(0b11111111, None),
                       -test.compute_permsum(2, [0, 1, 3, 2], deriv=None))
    assert np.allclose(-test._olp_deriv(0b11111111, 0),
                       test.compute_permsum(2, [0, 1, 3, 2], deriv=0))


def test_olp():
    """Test BaseQuasiparticle._olp."""
    test = TestQuasiparticle()
    test.assign_nquasiparticle(4)
    test.assign_orbsubsets([(0, 1), (2, 3), (4,), (5, 6, 7), (1, 3, 5), (2, 4, 6, 8)])
    test.params = np.arange(24).reshape(4, 6)
    assert np.allclose(test._olp_deriv(0b11111111, None),
                       -test.compute_permsum(2, [0, 1, 3, 2], deriv=None))


def test_get_overlap():
    """Test BaseQuasiparticle.get_overlap."""
    test = TestQuasiparticle()
    test.assign_nquasiparticle(4)
    test.assign_orbsubsets([(0, 1), (2, 3), (4,), (5, 6, 7), (1, 3, 5), (2, 4, 6, 8)])
    test.params = np.arange(24).reshape(4, 6)
    test.assign_memory()
    test._cache_fns = {}
    test.load_cache()
    assert test.get_overlap(0b11111111) == -test.compute_permsum(2, [0, 1, 3, 2], deriv=None)
    assert test.get_overlap(0b11111111, deriv=0) == -test.compute_permsum(2, [0, 1, 3, 2], deriv=0)
    assert test.get_overlap(0b11111111, deriv=5) == 0
    assert test.get_overlap(0b11111111, deriv=24) == 0

    assert_raises(ValueError, test.get_overlap, 0b11111111, deriv='1')
    assert_raises(ValueError, test.get_overlap, 0b11111111, deriv=1.0)
