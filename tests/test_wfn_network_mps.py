"""Tests for the wfns.wfn.network.mps."""
import numpy as np
import numdifftools as nd
import pytest
from utils import skip_init
from wfns.wfn.network.mps import MatrixProductState


def test_assign_dimension():
    """Test MatrixProductState.assign_dimension."""
    test = skip_init(MatrixProductState)
    test.nspin = 4

    test.assign_dimension()
    assert test.dimension == 4
    test.assign_dimension(2)
    assert test.dimension == 2

    with pytest.raises(TypeError):
        test.assign_dimension(2.0)
    with pytest.raises(ValueError):
        test.assign_dimension(-5)


def test_get_occupation_indices():
    """Test MatrixProductState.get_occupation_indices."""
    test = skip_init(MatrixProductState)
    test.nspin = 6
    assert np.allclose(test.get_occupation_indices(0b000000), [0, 0, 0])
    assert np.allclose(test.get_occupation_indices(0b000001), [1, 0, 0])
    assert np.allclose(test.get_occupation_indices(0b001000), [2, 0, 0])
    assert np.allclose(test.get_occupation_indices(0b001001), [3, 0, 0])
    assert np.allclose(test.get_occupation_indices(0b011101), [3, 2, 1])


def test_get_matrix_shape():
    """Test MatrixProductState.get_matrix_shape."""
    test = skip_init(MatrixProductState)
    test.nspin = 8
    test.dimension = 10
    assert test.get_matrix_shape(0) == (4, 1, 10)
    assert test.get_matrix_shape(1) == (4, 10, 10)
    assert test.get_matrix_shape(2) == (4, 10, 10)
    assert test.get_matrix_shape(3) == (4, 10, 1)

    with pytest.raises(TypeError):
        test.get_matrix_shape(0.0)
    with pytest.raises(ValueError):
        test.get_matrix_shape(-1)
    with pytest.raises(ValueError):
        test.get_matrix_shape(4)


def test_get_matrix_indices():
    """Test MatrixProductState.get_matrix_indices."""
    test = skip_init(MatrixProductState)
    test.nspin = 8
    test.dimension = 10
    assert test.get_matrix_indices(0) == (0, 40)
    assert test.get_matrix_indices(1) == (40, 440)
    assert test.get_matrix_indices(2) == (440, 840)
    assert test.get_matrix_indices(3) == (840, 880)


def test_get_matrix():
    """Test MatrixProductState.get_matrix."""
    test = skip_init(MatrixProductState)
    test.nspin = 8
    test.dimension = 10
    test.params = np.arange(880)
    assert np.allclose(test.get_matrix(0), np.arange(40).reshape(4, 1, 10))
    assert np.allclose(test.get_matrix(1), np.arange(40, 440).reshape(4, 10, 10))
    assert np.allclose(test.get_matrix(2), np.arange(440, 840).reshape(4, 10, 10))
    assert np.allclose(test.get_matrix(3), np.arange(840, 880).reshape(4, 10, 1))


def test_decompose_index():
    """Test MatrixProductState.decompose_index."""
    test = skip_init(MatrixProductState)
    test.nspin = 8
    test.dimension = 10
    for i in range(40):
        ind_occ = i // 10
        ind_row = 0
        ind_col = (i % 10) % 10
        assert test.decompose_index(i) == (0, ind_occ, ind_row, ind_col)

    for i in range(40, 840):
        ind_spatial = (i - 40) // 400
        ind_spatial += 1
        ind_occ = ((i - 40) % 400) // 100
        ind_row = (((i - 40) % 400) % 100) // 10
        ind_col = (((i - 40) % 400) % 100) % 10
        assert test.decompose_index(i) == (ind_spatial, ind_occ, ind_row, ind_col)

    for i in range(840, 880):
        ind_occ = (i - 840) // 10
        ind_row = (i - 840) % 10
        ind_col = 0
        assert test.decompose_index(i) == (3, ind_occ, ind_row, ind_col)


def test_default_params():
    """Test MatrixProductState.default_params."""
    test = skip_init(MatrixProductState)
    test.nelec = 4
    test.nspin = 8
    test.dimension = 10

    answer1 = np.ones(10) * 10 ** (-1 / 4)
    answer2 = np.identity(10) * 10 ** (-1 / 4)

    test.assign_params()
    default = test.params

    tensor = default[0:40].reshape(4, 1, 10)
    assert np.allclose(tensor[0], 0)
    assert np.allclose(tensor[1], 0)
    assert np.allclose(tensor[2], 0)
    assert np.allclose(tensor[3], answer1)

    tensor = default[40:440].reshape(4, 10, 10)
    assert np.allclose(tensor[0], 0)
    assert np.allclose(tensor[1], 0)
    assert np.allclose(tensor[2], 0)
    assert np.allclose(tensor[3], answer2)

    tensor = default[440:840].reshape(4, 10, 10)
    assert np.allclose(tensor[0], answer2)
    assert np.allclose(tensor[1], 0)
    assert np.allclose(tensor[2], 0)
    assert np.allclose(tensor[3], 0)

    tensor = default[840:880].reshape(4, 10, 1)
    assert np.allclose(tensor[0], answer1)
    assert np.allclose(tensor[1], 0)
    assert np.allclose(tensor[2], 0)
    assert np.allclose(tensor[3], 0)


# FIXME: this may move to the parent class
def test_assign_params():
    """Test MatrixProductState.assign_params."""
    test = skip_init(MatrixProductState)
    test.nelec = 2
    test.nspin = 6
    test.dimension = 2
    params = np.array(
        [0, 0, 0, 0, 0, 0, 1, 1] + [1, 0, 0, 1] + 3 * [0, 0, 0, 0] + [1, 1, 0, 0, 0, 0, 0, 0]
    ) * 2 ** (-1 / 3)
    test.assign_params(params=None, add_noise=False)
    assert np.allclose(test.params, params)

    test2 = skip_init(MatrixProductState)
    test2.nelec = 2
    test2.nspin = 6
    test2.dimension = 2
    test2.assign_params(params=test, add_noise=False)
    assert np.allclose(test2.params, params)


def test_olp():
    """Test MatrixProductState._olp."""
    matrix1 = np.random.rand(4, 1, 2)
    matrix2 = np.random.rand(4, 2, 2)
    matrix3 = np.random.rand(4, 2, 1)

    test = MatrixProductState(
        2, 6, dimension=2, params=np.hstack([matrix1.flat, matrix2.flat, matrix3.flat])
    )

    assert np.allclose(test._olp(0b000011), matrix1[1].dot(matrix2[1]).dot(matrix3[0]))
    assert np.allclose(test._olp(0b000101), matrix1[1].dot(matrix2[0]).dot(matrix3[1]))
    assert np.allclose(test._olp(0b001001), matrix1[3].dot(matrix2[0]).dot(matrix3[0]))
    assert np.allclose(test._olp(0b010001), matrix1[1].dot(matrix2[2]).dot(matrix3[0]))
    assert np.allclose(test._olp(0b100001), matrix1[1].dot(matrix2[0]).dot(matrix3[2]))

    assert np.allclose(test._olp(0b000110), matrix1[0].dot(matrix2[1]).dot(matrix3[1]))
    assert np.allclose(test._olp(0b001010), matrix1[2].dot(matrix2[1]).dot(matrix3[0]))
    assert np.allclose(test._olp(0b010010), matrix1[0].dot(matrix2[3]).dot(matrix3[0]))
    assert np.allclose(test._olp(0b100010), matrix1[0].dot(matrix2[1]).dot(matrix3[2]))

    assert np.allclose(test._olp(0b001100), matrix1[2].dot(matrix2[0]).dot(matrix3[1]))
    assert np.allclose(test._olp(0b010100), matrix1[0].dot(matrix2[2]).dot(matrix3[1]))
    assert np.allclose(test._olp(0b100100), matrix1[0].dot(matrix2[0]).dot(matrix3[3]))

    assert np.allclose(test._olp(0b011000), matrix1[2].dot(matrix2[2]).dot(matrix3[0]))
    assert np.allclose(test._olp(0b101000), matrix1[2].dot(matrix2[0]).dot(matrix3[2]))

    assert np.allclose(test._olp(0b110000), matrix1[0].dot(matrix2[2]).dot(matrix3[2]))


def test_olp_deriv():
    """Test MatrixProductState._olp_deriv."""
    matrix1 = np.random.rand(4, 1, 2)
    matrix2 = np.random.rand(4, 2, 2)
    matrix3 = np.random.rand(4, 2, 2)
    matrix4 = np.random.rand(4, 2, 1)

    test = MatrixProductState(
        2,
        8,
        dimension=2,
        params=np.hstack([matrix1.flat, matrix2.flat, matrix3.flat, matrix4.flat]),
    )

    assert np.allclose(
        test._olp_deriv(0b00000011, 2), matrix2[1, 0, :].dot(matrix3[0]).dot(matrix4[0])
    )
    assert np.allclose(
        test._olp_deriv(0b00000011, 3), matrix2[1, 1, :].dot(matrix3[0]).dot(matrix4[0])
    )
    assert np.allclose(
        test._olp_deriv(0b00000011, 12), matrix1[1, :, 0] * matrix3[0, 0, :].dot(matrix4[0])
    )
    assert np.allclose(
        test._olp_deriv(0b00000011, 13), matrix1[1, :, 0] * matrix3[0, 1, :].dot(matrix4[0])
    )
    assert np.allclose(
        test._olp_deriv(0b00000011, 14), matrix1[1, :, 1] * matrix3[0, 0, :].dot(matrix4[0])
    )
    assert np.allclose(
        test._olp_deriv(0b00000011, 15), matrix1[1, :, 1] * matrix3[0, 1, :].dot(matrix4[0])
    )
    assert np.allclose(
        test._olp_deriv(0b00000011, 24), matrix1[1].dot(matrix2[1, :, 0]) * matrix4[0, 0, :]
    )
    assert np.allclose(
        test._olp_deriv(0b00000011, 25), matrix1[1].dot(matrix2[1, :, 0]) * matrix4[0, 1, :]
    )
    assert np.allclose(
        test._olp_deriv(0b00000011, 26), matrix1[1].dot(matrix2[1, :, 1]) * matrix4[0, 0, :]
    )
    assert np.allclose(
        test._olp_deriv(0b00000011, 27), matrix1[1].dot(matrix2[1, :, 1]) * matrix4[0, 1, :]
    )
    assert np.allclose(
        test._olp_deriv(0b00000011, 40), matrix1[1].dot(matrix2[1]).dot(matrix3[0, :, 0])
    )
    assert np.allclose(
        test._olp_deriv(0b00000011, 41), matrix1[1].dot(matrix2[1]).dot(matrix3[0, :, 1])
    )


def test_get_overlap():
    """Test MatrixProductState.get_overlap."""
    matrix1 = np.random.rand(4, 1, 2)
    matrix2 = np.random.rand(4, 2, 2)
    matrix3 = np.random.rand(4, 2, 1)

    test = MatrixProductState(
        2, 6, dimension=2, params=np.hstack([matrix1.flat, matrix2.flat, matrix3.flat])
    )

    with pytest.raises(TypeError):
        test.get_overlap(0b0101, 0.0)
    # derivative
    new_wfn = MatrixProductState(2, 6, dimension=2)

    def overlap(params):
        new_wfn.assign_params(params)
        return new_wfn.get_overlap(0b001001)

    grad = nd.Gradient(overlap)(test.params.copy())
    assert np.allclose(grad, test.get_overlap(0b001001, np.arange(test.nparams)))
