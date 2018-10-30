"""Test wfns.wfn.quasiparticle.det_ratio."""
from nose.tools import assert_raises
import numpy as np
from wfns.wfn.quasiparticle.det_ratio import DeterminantRatio


class TestDeterminantRatio(DeterminantRatio):
    """DeterminantRatio that skips initialization."""
    def __init__(self):
        pass


def test_assign_numerator_mask():
    """Test DeterminantRatio.assign_numerator_mask."""
    test = TestDeterminantRatio()
    # check default
    test.assign_numerator_mask()
    assert np.allclose(test.numerator_mask, [True, False])
    # check custom
    test.assign_numerator_mask(np.array([True, False, True]))
    assert np.allclose(test.numerator_mask, [True, False, True])
    # check errors
    assert_raises(TypeError, test.assign_numerator_mask, [True, False])
    assert_raises(TypeError, test.assign_numerator_mask, np.array([0, 1]))
    assert_raises(TypeError, test.assign_numerator_mask, np.array([], dtype=bool))


def test_matrix_shape():
    """Test DeterminantRatio.matrix_shape."""
    test = TestDeterminantRatio()
    test.nelec = 2
    test.nspin = 4
    assert test.matrix_shape == (2, 4)


def test_matrix_size():
    """Test DeterminantRatio.matrix_size."""
    test = TestDeterminantRatio()
    test.nelec = 2
    test.nspin = 4
    assert test.matrix_size == 8


def test_num_matrices():
    """Test DeterminantRatio.num_matrices."""
    test = TestDeterminantRatio()
    test.numerator_mask = np.array([False, False, False, True, False])
    assert test.num_matrices == 5


def test_get_columns():
    """Test DeterminantRatio.get_columns."""
    test = TestDeterminantRatio()
    assert np.allclose(test.get_columns(0b010101, 0), np.array([0, 2, 4]))
    assert np.allclose(test.get_columns(0b000111, 0), np.array([0, 1, 2]))


def test_get_matrix():
    """Test DeterminantRatio.get_matrix."""
    test = TestDeterminantRatio()
    test.nelec = 2
    test.nspin = 4
    test.numerator_mask = np.array([True, False])
    test.params = np.random.rand(16)
    assert np.allclose(test.get_matrix(0), test.params[:8].reshape(2, 4))
    assert np.allclose(test.get_matrix(1), test.params[8:].reshape(2, 4))
    assert_raises(TypeError, test.get_matrix, '0')
    assert_raises(TypeError, test.get_matrix, 0.0)
    assert_raises(ValueError, test.get_matrix, -1)
    assert_raises(ValueError, test.get_matrix, 16)


def test_decompose_index():
    """Test DeterminantRatio.decompose_index."""
    test = TestDeterminantRatio()
    test.nelec = 2
    test.nspin = 4
    test.numerator_mask = np.array([True, False])
    test.params = np.random.rand(16)
    for ind_matrix in range(2):
        for ind_row_matrix in range(test.nelec):
            for ind_col_matrix in range(test.nspin):
                assert (test.decompose_index(ind_matrix*8 + ind_row_matrix*4 + ind_col_matrix) ==
                        (ind_matrix, ind_row_matrix, ind_col_matrix))


def test_template_params():
    """Test DeterminantRatio.template_params."""
    test = TestDeterminantRatio()
    test.dtype = float
    test.nelec = 2
    test.nspin = 4
    test.numerator_mask = np.array([True, False, False])

    matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    indices = np.array([True, False, True, False])
    assert np.allclose(test.template_params[:8], matrix.flat)

    template_params = test.template_params
    template_matrix = template_params[8: 16].reshape(2, 4)
    assert np.allclose(template_matrix[:, indices], matrix[:, indices])
    assert np.allclose(np.linalg.norm(template_matrix[:, np.logical_not(indices)], axis=0), 1)

    assert np.allclose(template_params[8: 16], template_params[16:])


def test_assign_params():
    """Test DeterminantRatio.assign_params."""
    test = TestDeterminantRatio()
    test.dtype = float
    test.nelec = 2
    test.nspin = 4
    test.numerator_mask = np.array([True, False, False])
    test.assign_params(params=None, add_noise=False)

    # check if template is correct (cannot check directly because template changes with every call)
    matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    indices = np.array([True, False, True, False])
    assert np.allclose(test.params[:8], matrix.flat)
    template_matrix = test.params[8: 16].reshape(2, 4)
    assert np.allclose(template_matrix[:, indices], matrix[:, indices])
    assert np.allclose(np.linalg.norm(template_matrix[:, np.logical_not(indices)], axis=0), 1)
    assert np.allclose(test.params[16:], test.params[8: 16])

    params = np.array([1, 2, 3, 4, 5, 6, 7, 8] * 3, dtype=float)
    test.assign_params(params=params, add_noise=False)
    assert np.allclose(test.params, params)

    test2 = TestDeterminantRatio()
    test2.dtype = float
    test2.nelec = 2
    test2.nspin = 4
    test2.numerator_mask = np.array([True, False, False])
    test2.assign_params(params=test, add_noise=False)
    assert np.allclose(test2.params, params)


def test_olp():
    """Test DeterminantRatio._olp."""
    matrix1 = np.random.rand(8).reshape(2, 4)
    matrix2 = np.random.rand(8).reshape(2, 4)

    funcs = [lambda x, y: x / y,
             lambda x, y: y / x,
             lambda x, y: x * y,
             lambda x, y: 1 / x / y]
    tests = [DeterminantRatio(2, 4, numerator_mask=np.array([True, False]),
                              params=np.hstack([matrix1.flat, matrix2.flat])),
             DeterminantRatio(2, 4, numerator_mask=np.array([False, True]),
                              params=np.hstack([matrix1.flat, matrix2.flat])),
             DeterminantRatio(2, 4, numerator_mask=np.array([True, True]),
                              params=np.hstack([matrix1.flat, matrix2.flat])),
             DeterminantRatio(2, 4, numerator_mask=np.array([False, False]),
                              params=np.hstack([matrix1.flat, matrix2.flat]))]

    for func, test in zip(funcs, tests):
        assert np.allclose(test._olp(0b0011),
                           func(np.linalg.det(matrix1[:, [0, 1]]),
                                np.linalg.det(matrix2[:, [0, 1]])))
        assert np.allclose(test._olp(0b0101),
                           func(np.linalg.det(matrix1[:, [0, 2]]),
                                np.linalg.det(matrix2[:, [0, 2]])))
        assert np.allclose(test._olp(0b1001),
                           func(np.linalg.det(matrix1[:, [0, 3]]),
                                np.linalg.det(matrix2[:, [0, 3]])))
        assert np.allclose(test._olp(0b0110),
                           func(np.linalg.det(matrix1[:, [1, 2]]),
                                np.linalg.det(matrix2[:, [1, 2]])))
        assert np.allclose(test._olp(0b1010),
                           func(np.linalg.det(matrix1[:, [1, 3]]),
                                np.linalg.det(matrix2[:, [1, 3]])))
        assert np.allclose(test._olp(0b1100),
                           func(np.linalg.det(matrix1[:, [2, 3]]),
                                np.linalg.det(matrix2[:, [2, 3]])))


def test_olp_deriv():
    """Test DeterminantRatio._olp_deriv."""
    matrix1 = np.random.rand(8).reshape(2, 4)
    matrix2 = np.random.rand(8).reshape(2, 4)

    test = DeterminantRatio(2, 4, numerator_mask=np.array([True, False]),
                            params=np.hstack([matrix1.flat, matrix2.flat]))
    assert np.allclose(test._olp_deriv(0b0011, 0),
                       matrix1[1, 1] / np.linalg.det(matrix2[:, [0, 1]]))
    assert np.allclose(test._olp_deriv(0b0011, 1),
                       - matrix1[1, 0] / np.linalg.det(matrix2[:, [0, 1]]))
    assert np.allclose(test._olp_deriv(0b0011, 4),
                       - matrix1[0, 1] / np.linalg.det(matrix2[:, [0, 1]]))
    assert np.allclose(test._olp_deriv(0b0011, 5),
                       matrix1[0, 0] / np.linalg.det(matrix2[:, [0, 1]]))
    assert np.allclose(test._olp_deriv(0b0011, 8),
                       np.linalg.det(matrix1[:, [0, 1]])
                       * (-1) / np.linalg.det(matrix2[:, [0, 1]]) ** 2 * matrix2[1, 1])
    assert np.allclose(test._olp_deriv(0b0011, 9),
                       np.linalg.det(matrix1[:, [0, 1]])
                       * (-1) / np.linalg.det(matrix2[:, [0, 1]]) ** 2 * (-matrix2[1, 0]))
    assert np.allclose(test._olp_deriv(0b0011, 12),
                       np.linalg.det(matrix1[:, [0, 1]])
                       * (-1) / np.linalg.det(matrix2[:, [0, 1]]) ** 2 * (-matrix2[0, 1]))
    assert np.allclose(test._olp_deriv(0b0011, 13),
                       np.linalg.det(matrix1[:, [0, 1]])
                       * (-1) / np.linalg.det(matrix2[:, [0, 1]]) ** 2 * matrix2[0, 0])

    assert np.allclose(test._olp_deriv(0b1010, 1),
                       matrix1[1, 3] / np.linalg.det(matrix2[:, [1, 3]]))
    assert np.allclose(test._olp_deriv(0b1010, 3),
                       - matrix1[1, 1] / np.linalg.det(matrix2[:, [1, 3]]))
    assert np.allclose(test._olp_deriv(0b1010, 5),
                       - matrix1[0, 3] / np.linalg.det(matrix2[:, [1, 3]]))
    assert np.allclose(test._olp_deriv(0b1010, 7),
                       matrix1[0, 1] / np.linalg.det(matrix2[:, [1, 3]]), matrix2[1, 3])
    assert np.allclose(test._olp_deriv(0b1010, 9),
                       np.linalg.det(matrix1[:, [1, 3]])
                       * (-1) / np.linalg.det(matrix2[:, [1, 3]])**2 * matrix2[1, 3])
    assert np.allclose(test._olp_deriv(0b1010, 11),
                       np.linalg.det(matrix1[:, [1, 3]])
                       * (-1) / np.linalg.det(matrix2[:, [1, 3]])**2 * (-matrix2[1, 1]))
    assert np.allclose(test._olp_deriv(0b1010, 13),
                       np.linalg.det(matrix1[:, [1, 3]])
                       * (-1) / np.linalg.det(matrix2[:, [1, 3]])**2 * (-matrix2[0, 3]))
    assert np.allclose(test._olp_deriv(0b1010, 15),
                       np.linalg.det(matrix1[:, [1, 3]])
                       * (-1) / np.linalg.det(matrix2[:, [1, 3]])**2 * matrix2[0, 1])

    test = DeterminantRatio(2, 4, numerator_mask=np.array([False, True]),
                            params=np.hstack([matrix1.flat, matrix2.flat]))
    assert np.allclose(test._olp_deriv(0b0101, 0),
                       np.linalg.det(matrix2[:, [0, 2]])
                       * (-1) / np.linalg.det(matrix1[:, [0, 2]]) ** 2 * matrix1[1, 2])
    assert np.allclose(test._olp_deriv(0b0101, 2),
                       np.linalg.det(matrix2[:, [0, 2]])
                       * (-1) / np.linalg.det(matrix1[:, [0, 2]]) ** 2 * (-matrix1[1, 0]))
    assert np.allclose(test._olp_deriv(0b0101, 4),
                       np.linalg.det(matrix2[:, [0, 2]])
                       * (-1) / np.linalg.det(matrix1[:, [0, 2]]) ** 2 * (-matrix1[0, 2]))
    assert np.allclose(test._olp_deriv(0b0101, 6),
                       np.linalg.det(matrix2[:, [0, 2]])
                       * (-1) / np.linalg.det(matrix1[:, [0, 2]]) ** 2 * matrix1[0, 0])
    assert np.allclose(test._olp_deriv(0b0101, 8),
                       matrix2[1, 2] / np.linalg.det(matrix1[:, [0, 2]]))
    assert np.allclose(test._olp_deriv(0b0101, 10),
                       - matrix2[1, 0] / np.linalg.det(matrix1[:, [0, 2]]))
    assert np.allclose(test._olp_deriv(0b0101, 12),
                       - matrix2[0, 2] / np.linalg.det(matrix1[:, [0, 2]]))
    assert np.allclose(test._olp_deriv(0b0101, 14),
                       matrix2[0, 0] / np.linalg.det(matrix1[:, [0, 2]]))

    test = DeterminantRatio(2, 4, numerator_mask=np.array([True, True]),
                            params=np.hstack([matrix1.flat, matrix2.flat]))
    assert np.allclose(test._olp_deriv(0b1001, 0),
                       matrix1[1, 3] * np.linalg.det(matrix2[:, [0, 3]]))
    assert np.allclose(test._olp_deriv(0b1001, 3),
                       -matrix1[1, 0] * np.linalg.det(matrix2[:, [0, 3]]))
    assert np.allclose(test._olp_deriv(0b1001, 4),
                       -matrix1[0, 3] * np.linalg.det(matrix2[:, [0, 3]]))
    assert np.allclose(test._olp_deriv(0b1001, 7),
                       matrix1[0, 0] * np.linalg.det(matrix2[:, [0, 3]]))
    assert np.allclose(test._olp_deriv(0b1001, 8),
                       np.linalg.det(matrix1[:, [0, 3]]) * matrix2[1, 3])
    assert np.allclose(test._olp_deriv(0b1001, 11),
                       np.linalg.det(matrix1[:, [0, 3]]) * (-matrix2[1, 0]))
    assert np.allclose(test._olp_deriv(0b1001, 12),
                       np.linalg.det(matrix1[:, [0, 3]]) * (-matrix2[0, 3]))
    assert np.allclose(test._olp_deriv(0b1001, 15),
                       np.linalg.det(matrix1[:, [0, 3]]) * matrix2[0, 0])

    test = DeterminantRatio(2, 4, numerator_mask=np.array([False, False]),
                            params=np.hstack([matrix1.flat, matrix2.flat]))
    assert np.allclose(test._olp_deriv(0b0110, 1),
                       (-1) / np.linalg.det(matrix1[:, [1, 2]]) ** 2 * matrix1[1, 2]
                       / np.linalg.det(matrix2[:, [1, 2]]))
    assert np.allclose(test._olp_deriv(0b0110, 2),
                       (-1) / np.linalg.det(matrix1[:, [1, 2]]) ** 2 * (-matrix1[1, 1])
                       / np.linalg.det(matrix2[:, [1, 2]]))
    assert np.allclose(test._olp_deriv(0b0110, 5),
                       (-1) / np.linalg.det(matrix1[:, [1, 2]]) ** 2 * (-matrix1[0, 2])
                       / np.linalg.det(matrix2[:, [1, 2]]))
    assert np.allclose(test._olp_deriv(0b0110, 6),
                       (-1) / np.linalg.det(matrix1[:, [1, 2]]) ** 2 * matrix1[0, 1]
                       / np.linalg.det(matrix2[:, [1, 2]]))
    assert np.allclose(test._olp_deriv(0b0110, 9),
                       1 / np.linalg.det(matrix1[:, [1, 2]])
                       * (-1) / np.linalg.det(matrix2[:, [1, 2]]) ** 2 * matrix2[1, 2])
    assert np.allclose(test._olp_deriv(0b0110, 10),
                       1 / np.linalg.det(matrix1[:, [1, 2]])
                       * (-1) / np.linalg.det(matrix2[:, [1, 2]]) ** 2 * (-matrix2[1, 1]))
    assert np.allclose(test._olp_deriv(0b0110, 13),
                       1 / np.linalg.det(matrix1[:, [1, 2]])
                       * (-1) / np.linalg.det(matrix2[:, [1, 2]]) ** 2 * (-matrix2[0, 2]))
    assert np.allclose(test._olp_deriv(0b0110, 14),
                       1 / np.linalg.det(matrix1[:, [1, 2]])
                       * (-1) / np.linalg.det(matrix2[:, [1, 2]]) ** 2 * matrix2[0, 1])


def test_get_overlap():
    """Test DeterminantRatio.get_overlap."""
    matrix1 = np.random.rand(8).reshape(2, 4)
    matrix2 = np.random.rand(8).reshape(2, 4)
    test = DeterminantRatio(2, 4, numerator_mask=np.array([True, False]),
                            params=np.hstack([matrix1.flat, matrix2.flat]))
    assert test.get_overlap(0b0001) == 0
    assert test.get_overlap(0b0111) == 0

    assert test.get_overlap(0b0101, -1) == 0
    assert test.get_overlap(0b0101, 16) == 0
    for i in range(16):
        if i in [0, 2, 4, 6, 8, 10, 12, 14]:
            continue
        assert test.get_overlap(0b0101, i) == 0
    assert_raises(TypeError, test.get_overlap, 0b0101, 0.0)

    assert np.allclose(test.get_overlap(0b0011),
                       np.linalg.det(matrix1[:, [0, 1]]) / np.linalg.det(matrix2[:, [0, 1]]))
    assert np.allclose(test.get_overlap(0b0011, 0),
                       matrix1[1, 1] / np.linalg.det(matrix2[:, [0, 1]]))
