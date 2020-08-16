"""Test fanpy.wfn.quasiparticle.det_ratio."""
import pytest
import numpy as np
from fanpy.wfn.quasiparticle.det_ratio import DeterminantRatio


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
    with pytest.raises(TypeError):
        test.assign_numerator_mask([True, False])
    with pytest.raises(TypeError):
        test.assign_numerator_mask(np.array([0, 1]))
    with pytest.raises(TypeError):
        test.assign_numerator_mask(np.array([], dtype=bool))


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
    with pytest.raises(TypeError):
        test.get_matrix("0")
    with pytest.raises(TypeError):
        test.get_matrix(0.0)
    with pytest.raises(ValueError):
        test.get_matrix(-1)
    with pytest.raises(ValueError):
        test.get_matrix(16)


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
                assert test.decompose_index(
                    ind_matrix * 8 + ind_row_matrix * 4 + ind_col_matrix
                ) == (ind_matrix, ind_row_matrix, ind_col_matrix)


def test_assign_params():
    """Test DeterminantRatio.assign_params."""
    test = TestDeterminantRatio()
    test.nelec = 2
    test.nspin = 4
    test.numerator_mask = np.array([True, False, False])
    test.assign_params(params=None, add_noise=False)

    # check if template is correct (cannot check directly because template changes with every call)
    matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    indices = np.array([True, False, True, False])
    assert np.allclose(test.params[:8], matrix.flat)
    template_matrix = test.params[8:16].reshape(2, 4)
    assert np.allclose(template_matrix[:, indices], matrix[:, indices])
    assert np.allclose(np.linalg.norm(template_matrix[:, np.logical_not(indices)], axis=0), 1)
    assert np.allclose(test.params[16:], test.params[8:16])

    params = np.array([1, 2, 3, 4, 5, 6, 7, 8] * 3, dtype=float)
    test.assign_params(params=params, add_noise=False)
    assert np.allclose(test.params, params)

    test2 = TestDeterminantRatio()
    test2.nelec = 2
    test2.nspin = 4
    test2.numerator_mask = np.array([True, False, False])
    test2.assign_params(params=test, add_noise=False)
    assert np.allclose(test2.params, params)


def test_olp():
    """Test DeterminantRatio._olp."""
    matrix1 = np.random.rand(8).reshape(2, 4)
    matrix2 = np.random.rand(8).reshape(2, 4)

    funcs = [lambda x, y: x / y, lambda x, y: y / x, lambda x, y: x * y, lambda x, y: 1 / x / y]
    tests = [
        DeterminantRatio(
            2,
            4,
            numerator_mask=np.array([True, False]),
            params=np.hstack([matrix1.flat, matrix2.flat]),
        ),
        DeterminantRatio(
            2,
            4,
            numerator_mask=np.array([False, True]),
            params=np.hstack([matrix1.flat, matrix2.flat]),
        ),
        DeterminantRatio(
            2,
            4,
            numerator_mask=np.array([True, True]),
            params=np.hstack([matrix1.flat, matrix2.flat]),
        ),
        DeterminantRatio(
            2,
            4,
            numerator_mask=np.array([False, False]),
            params=np.hstack([matrix1.flat, matrix2.flat]),
        ),
    ]

    for func, test in zip(funcs, tests):
        assert np.allclose(
            test._olp(0b0011),
            func(np.linalg.det(matrix1[:, [0, 1]]), np.linalg.det(matrix2[:, [0, 1]])),
        )
        assert np.allclose(
            test._olp(0b0101),
            func(np.linalg.det(matrix1[:, [0, 2]]), np.linalg.det(matrix2[:, [0, 2]])),
        )
        assert np.allclose(
            test._olp(0b1001),
            func(np.linalg.det(matrix1[:, [0, 3]]), np.linalg.det(matrix2[:, [0, 3]])),
        )
        assert np.allclose(
            test._olp(0b0110),
            func(np.linalg.det(matrix1[:, [1, 2]]), np.linalg.det(matrix2[:, [1, 2]])),
        )
        assert np.allclose(
            test._olp(0b1010),
            func(np.linalg.det(matrix1[:, [1, 3]]), np.linalg.det(matrix2[:, [1, 3]])),
        )
        assert np.allclose(
            test._olp(0b1100),
            func(np.linalg.det(matrix1[:, [2, 3]]), np.linalg.det(matrix2[:, [2, 3]])),
        )


def test_olp_deriv():
    """Test DeterminantRatio._olp_deriv."""
    matrix1 = np.random.rand(8).reshape(2, 4)
    matrix2 = np.random.rand(8).reshape(2, 4)

    test = DeterminantRatio(
        2, 4, numerator_mask=np.array([True, False]), params=np.hstack([matrix1.flat, matrix2.flat])
    )
    deriv = test._olp_deriv(0b0011)
    assert np.allclose(deriv[0], matrix1[1, 1] / np.linalg.det(matrix2[:, [0, 1]]))
    assert np.allclose(deriv[1], -matrix1[1, 0] / np.linalg.det(matrix2[:, [0, 1]]))
    assert np.allclose(deriv[4], -matrix1[0, 1] / np.linalg.det(matrix2[:, [0, 1]]))
    assert np.allclose(deriv[5], matrix1[0, 0] / np.linalg.det(matrix2[:, [0, 1]]))
    assert np.allclose(
        deriv[8],
        np.linalg.det(matrix1[:, [0, 1]])
        * (-1)
        / np.linalg.det(matrix2[:, [0, 1]]) ** 2
        * matrix2[1, 1],
    )
    assert np.allclose(
        deriv[9],
        np.linalg.det(matrix1[:, [0, 1]])
        * (-1)
        / np.linalg.det(matrix2[:, [0, 1]]) ** 2
        * (-matrix2[1, 0]),
    )
    assert np.allclose(
        deriv[12],
        np.linalg.det(matrix1[:, [0, 1]])
        * (-1)
        / np.linalg.det(matrix2[:, [0, 1]]) ** 2
        * (-matrix2[0, 1]),
    )
    assert np.allclose(
        deriv[13],
        np.linalg.det(matrix1[:, [0, 1]])
        * (-1)
        / np.linalg.det(matrix2[:, [0, 1]]) ** 2
        * matrix2[0, 0],
    )

    deriv = test._olp_deriv(0b1010)
    assert np.allclose(deriv[1], matrix1[1, 3] / np.linalg.det(matrix2[:, [1, 3]]))
    assert np.allclose(deriv[3], -matrix1[1, 1] / np.linalg.det(matrix2[:, [1, 3]]))
    assert np.allclose(deriv[5], -matrix1[0, 3] / np.linalg.det(matrix2[:, [1, 3]]))
    assert np.allclose(deriv[7], matrix1[0, 1] / np.linalg.det(matrix2[:, [1, 3]]), matrix2[1, 3])
    assert np.allclose(
        deriv[9],
        np.linalg.det(matrix1[:, [1, 3]])
        * (-1)
        / np.linalg.det(matrix2[:, [1, 3]]) ** 2
        * matrix2[1, 3],
    )
    assert np.allclose(
        deriv[11],
        np.linalg.det(matrix1[:, [1, 3]])
        * (-1)
        / np.linalg.det(matrix2[:, [1, 3]]) ** 2
        * (-matrix2[1, 1]),
    )
    assert np.allclose(
        deriv[13],
        np.linalg.det(matrix1[:, [1, 3]])
        * (-1)
        / np.linalg.det(matrix2[:, [1, 3]]) ** 2
        * (-matrix2[0, 3]),
    )
    assert np.allclose(
        deriv[15],
        np.linalg.det(matrix1[:, [1, 3]])
        * (-1)
        / np.linalg.det(matrix2[:, [1, 3]]) ** 2
        * matrix2[0, 1],
    )

    test = DeterminantRatio(
        2, 4, numerator_mask=np.array([False, True]), params=np.hstack([matrix1.flat, matrix2.flat])
    )
    deriv = test._olp_deriv(0b0101)
    assert np.allclose(
        deriv[0],
        np.linalg.det(matrix2[:, [0, 2]])
        * (-1)
        / np.linalg.det(matrix1[:, [0, 2]]) ** 2
        * matrix1[1, 2],
    )
    assert np.allclose(
        deriv[2],
        np.linalg.det(matrix2[:, [0, 2]])
        * (-1)
        / np.linalg.det(matrix1[:, [0, 2]]) ** 2
        * (-matrix1[1, 0]),
    )
    assert np.allclose(
        deriv[4],
        np.linalg.det(matrix2[:, [0, 2]])
        * (-1)
        / np.linalg.det(matrix1[:, [0, 2]]) ** 2
        * (-matrix1[0, 2]),
    )
    assert np.allclose(
        deriv[6],
        np.linalg.det(matrix2[:, [0, 2]])
        * (-1)
        / np.linalg.det(matrix1[:, [0, 2]]) ** 2
        * matrix1[0, 0],
    )
    assert np.allclose(deriv[8], matrix2[1, 2] / np.linalg.det(matrix1[:, [0, 2]]))
    assert np.allclose(deriv[10], -matrix2[1, 0] / np.linalg.det(matrix1[:, [0, 2]]))
    assert np.allclose(deriv[12], -matrix2[0, 2] / np.linalg.det(matrix1[:, [0, 2]]))
    assert np.allclose(deriv[14], matrix2[0, 0] / np.linalg.det(matrix1[:, [0, 2]]))

    test = DeterminantRatio(
        2, 4, numerator_mask=np.array([True, True]), params=np.hstack([matrix1.flat, matrix2.flat])
    )
    deriv = test._olp_deriv(0b1001)
    assert np.allclose(deriv[0], matrix1[1, 3] * np.linalg.det(matrix2[:, [0, 3]]))
    assert np.allclose(deriv[3], -matrix1[1, 0] * np.linalg.det(matrix2[:, [0, 3]]))
    assert np.allclose(deriv[4], -matrix1[0, 3] * np.linalg.det(matrix2[:, [0, 3]]))
    assert np.allclose(deriv[7], matrix1[0, 0] * np.linalg.det(matrix2[:, [0, 3]]))
    assert np.allclose(deriv[8], np.linalg.det(matrix1[:, [0, 3]]) * matrix2[1, 3])
    assert np.allclose(deriv[11], np.linalg.det(matrix1[:, [0, 3]]) * (-matrix2[1, 0]))
    assert np.allclose(deriv[12], np.linalg.det(matrix1[:, [0, 3]]) * (-matrix2[0, 3]))
    assert np.allclose(deriv[15], np.linalg.det(matrix1[:, [0, 3]]) * matrix2[0, 0])

    test = DeterminantRatio(
        2,
        4,
        numerator_mask=np.array([False, False]),
        params=np.hstack([matrix1.flat, matrix2.flat]),
    )
    deriv = test._olp_deriv(0b0110)
    assert np.allclose(
        deriv[1],
        (-1)
        / np.linalg.det(matrix1[:, [1, 2]]) ** 2
        * matrix1[1, 2]
        / np.linalg.det(matrix2[:, [1, 2]]),
    )
    assert np.allclose(
        deriv[2],
        (-1)
        / np.linalg.det(matrix1[:, [1, 2]]) ** 2
        * (-matrix1[1, 1])
        / np.linalg.det(matrix2[:, [1, 2]]),
    )
    assert np.allclose(
        deriv[5],
        (-1)
        / np.linalg.det(matrix1[:, [1, 2]]) ** 2
        * (-matrix1[0, 2])
        / np.linalg.det(matrix2[:, [1, 2]]),
    )
    assert np.allclose(
        deriv[6],
        (-1)
        / np.linalg.det(matrix1[:, [1, 2]]) ** 2
        * matrix1[0, 1]
        / np.linalg.det(matrix2[:, [1, 2]]),
    )
    assert np.allclose(
        deriv[9],
        1
        / np.linalg.det(matrix1[:, [1, 2]])
        * (-1)
        / np.linalg.det(matrix2[:, [1, 2]]) ** 2
        * matrix2[1, 2],
    )
    assert np.allclose(
        deriv[10],
        1
        / np.linalg.det(matrix1[:, [1, 2]])
        * (-1)
        / np.linalg.det(matrix2[:, [1, 2]]) ** 2
        * (-matrix2[1, 1]),
    )
    assert np.allclose(
        deriv[13],
        1
        / np.linalg.det(matrix1[:, [1, 2]])
        * (-1)
        / np.linalg.det(matrix2[:, [1, 2]]) ** 2
        * (-matrix2[0, 2]),
    )
    assert np.allclose(
        deriv[14],
        1
        / np.linalg.det(matrix1[:, [1, 2]])
        * (-1)
        / np.linalg.det(matrix2[:, [1, 2]]) ** 2
        * matrix2[0, 1],
    )


def test_get_overlap():
    """Test DeterminantRatio.get_overlap."""
    matrix1 = np.random.rand(8).reshape(2, 4)
    matrix2 = np.random.rand(8).reshape(2, 4)
    test = DeterminantRatio(
        2, 4, numerator_mask=np.array([True, False]), params=np.hstack([matrix1.flat, matrix2.flat])
    )
    assert test.get_overlap(0b0001) == 0
    assert test.get_overlap(0b0111) == 0

    for i in range(16):
        if i in [0, 2, 4, 6, 8, 10, 12, 14]:
            continue
        assert test.get_overlap(0b0101, np.array([i])) == 0
    with pytest.raises(TypeError):
        test.get_overlap(0b0101, 1)
    with pytest.raises(TypeError):
        test.get_overlap("1")

    assert np.allclose(
        test.get_overlap(0b0011),
        np.linalg.det(matrix1[:, [0, 1]]) / np.linalg.det(matrix2[:, [0, 1]]),
    )
    assert np.allclose(
        test.get_overlap(0b0011, np.array([0])), matrix1[1, 1] / np.linalg.det(matrix2[:, [0, 1]])
    )


def test_determinantratio_init():
    """Test DeterminantRatio.__init__."""
    wfn = DeterminantRatio(4, 10, enable_cache=True)
    assert wfn.nelec == 4
    assert wfn.nspin == 10
    assert wfn._cache_fns["overlap"]
    assert wfn._cache_fns["overlap derivative"]

    wfn = DeterminantRatio(4, 10, enable_cache=False)
    with pytest.raises(AttributeError):
        wfn._cache_fns["overlap"]
    with pytest.raises(AttributeError):
        wfn._cache_fns["overlap derivative"]
