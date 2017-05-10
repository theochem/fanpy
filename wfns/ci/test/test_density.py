""" Tests wfns.ci.density
"""
from __future__ import absolute_import, division, print_function
import os
import numpy as np
from scipy.linalg import eigh
from nose.tools import assert_raises
from nose.plugins.attrib import attr
from wfns import __file__ as package_path
from wfns.wrapper.horton import gaussian_fchk
from wfns.wrapper.pyscf import generate_fci_cimatrix
from wfns.ci.density import add_one_density, add_two_density, density_matrix


def test_add_one_density():
    """ Tests density.add_one_density
    """
    # check error
    #  tuple of numpy array
    assert_raises(TypeError, lambda: add_one_density((np.zeros((2, 2)),), 1, 1, 0.5, 'restricted'))
    #  list of nonnumpy array
    assert_raises(TypeError, lambda: add_one_density([np.zeros((2, 2)).tolist(),], 1, 1, 0.5,
                                                     'restricted'))
    #  bad matrix shape
    assert_raises(TypeError, lambda: add_one_density([np.zeros((2, 2, 2))], 1, 1, 0.5,
                                                     'restricted'))
    assert_raises(TypeError, lambda: add_one_density([np.zeros((2, 3))], 1, 1, 0.5, 'generalized'))
    #  bad orbital type
    assert_raises(ValueError, lambda: add_one_density([np.zeros((2, 2)),], 1, 1, 0.5, 'dsfsdfdsf'))
    assert_raises(ValueError, lambda: add_one_density([np.zeros((2, 2)),], 1, 1, 0.5, 'Restricted'))
    #  mismatching orbital type and matrices
    assert_raises(ValueError, lambda: add_one_density([np.zeros((2, 2))]*2, 1, 1, 0.5,
                                                      'restricted'))
    assert_raises(ValueError, lambda: add_one_density([np.zeros((2, 2))]*2, 1, 1, 0.5,
                                                      'generalized'))
    assert_raises(ValueError, lambda: add_one_density([np.zeros((2, 2)),], 1, 1, 0.5,
                                                      'unrestricted'))

    # restricted (3 spatial orbitals, 6 spin orbitals)
    matrices = [np.zeros((3, 3))]
    add_one_density(matrices, 0, 1, 1, 'restricted')
    assert np.allclose(matrices[0], np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]))
    add_one_density(matrices, 3, 1, 1, 'restricted')
    assert np.allclose(matrices[0], np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]))
    add_one_density(matrices, 0, 4, 1, 'restricted')
    assert np.allclose(matrices[0], np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]))
    add_one_density(matrices, 3, 4, 1, 'restricted')
    assert np.allclose(matrices[0], np.array([[0, 2, 0], [0, 0, 0], [0, 0, 0]]))

    # unrestricted (3 spatial orbitals, 6 spin orbitals)
    matrices = [np.zeros((3, 3)), np.zeros((3, 3))]
    add_one_density(matrices, 0, 1, 1, 'unrestricted')
    assert np.allclose(matrices[0], np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]))
    assert np.allclose(matrices[1], np.zeros((3, 3)))
    add_one_density(matrices, 3, 1, 1, 'unrestricted')
    assert np.allclose(matrices[0], np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]))
    assert np.allclose(matrices[1], np.zeros((3, 3)))
    add_one_density(matrices, 0, 4, 1, 'unrestricted')
    assert np.allclose(matrices[0], np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]))
    assert np.allclose(matrices[1], np.zeros((3, 3)))
    add_one_density(matrices, 3, 4, 1, 'unrestricted')
    assert np.allclose(matrices[0], np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]))
    assert np.allclose(matrices[1], np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]))

    # generalized (6 orbitals)
    matrices = [np.zeros((6, 6))]
    add_one_density(matrices, 0, 1, 1, 'generalized')
    assert np.allclose(matrices[0], np.array([[0, 1, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0]]))
    add_one_density(matrices, 3, 1, 1, 'generalized')
    assert np.allclose(matrices[0], np.array([[0, 1, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0],
                                              [0, 1, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0]]))
    add_one_density(matrices, 0, 4, 1, 'generalized')
    assert np.allclose(matrices[0], np.array([[0, 1, 0, 0, 1, 0],
                                              [0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0],
                                              [0, 1, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0]]))
    add_one_density(matrices, 3, 4, 1, 'generalized')
    assert np.allclose(matrices[0], np.array([[0, 1, 0, 0, 1, 0],
                                              [0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0],
                                              [0, 1, 0, 0, 1, 0],
                                              [0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0]]))


def test_add_two_density():
    """ Tests density.add_two_density
    """
    # check error
    #  tuple of numpy array
    assert_raises(TypeError, lambda: add_two_density((np.zeros((2, 2, 2, 2)),), 1, 1, 1, 1, 0.5,
                                                     'restricted'))
    #  list of nonnumpy array
    assert_raises(TypeError, lambda: add_two_density([np.zeros((2, 2, 2, 2)).tolist(),], 1, 1, 1, 1,
                                                     0.5, 'restricted'))
    #  bad matrix shape
    assert_raises(TypeError, lambda: add_two_density([np.zeros((2, 2, 2))], 1, 1, 1, 1, 0.5,
                                                     'restricted'))
    assert_raises(TypeError, lambda: add_two_density([np.zeros((2, 2, 2, 3))], 1, 1, 1, 1, 0.5,
                                                     'unrestricted'))
    #  bad orbital type
    assert_raises(ValueError, lambda: add_two_density([np.zeros((2, 2, 2, 2)),], 1, 1, 1, 1, 0.5,
                                                      'dsfsdfdsf'))
    assert_raises(ValueError, lambda: add_two_density([np.zeros((2, 2, 2, 2)),], 1, 1, 1, 1, 0.5,
                                                      'Restricted'))
    #  mismatching orbital type and matrices
    assert_raises(ValueError, lambda: add_two_density([np.zeros((2, 2, 2, 2))]*3, 1, 1, 1, 1, 0.5,
                                                     'restricted'))
    assert_raises(ValueError, lambda: add_two_density([np.zeros((2, 2, 2, 2))]*3, 1, 1, 1, 1, 0.5,
                                                      'generalized'))
    assert_raises(ValueError, lambda: add_two_density([np.zeros((2, 2, 2, 2))], 1, 1, 1, 1, 0.5,
                                                      'unrestricted'))

    # restricted (3 spatial orbitals, 6 spin orbitals)
    matrices = [np.zeros((4, 4, 4, 4))]
    answer = np.zeros((4, 4, 4, 4))
    add_two_density(matrices, 0, 1, 2, 3, 1, 'restricted')
    answer[0, 1, 2, 3] += 1
    assert np.allclose(matrices[0], answer)

    add_two_density(matrices, 4, 1, 2, 3, 1, 'restricted')
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 0, 5, 2, 3, 1, 'restricted')
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 0, 1, 6, 3, 1, 'restricted')
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 0, 1, 2, 7, 1, 'restricted')
    assert np.allclose(matrices[0], answer)

    add_two_density(matrices, 4, 5, 2, 3, 1, 'restricted')
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 4, 1, 6, 3, 1, 'restricted')
    answer[0, 1, 2, 3] += 1
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 4, 1, 2, 7, 1, 'restricted')
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 0, 5, 6, 3, 1, 'restricted')
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 0, 5, 2, 7, 1, 'restricted')
    answer[0, 1, 2, 3] += 1
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 0, 1, 6, 7, 1, 'restricted')
    assert np.allclose(matrices[0], answer)

    add_two_density(matrices, 4, 5, 6, 3, 1, 'restricted')
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 4, 5, 2, 7, 1, 'restricted')
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 4, 1, 6, 7, 1, 'restricted')
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 0, 5, 6, 7, 1, 'restricted')
    assert np.allclose(matrices[0], answer)

    add_two_density(matrices, 4, 5, 6, 7, 1, 'restricted')
    answer[0, 1, 2, 3] += 1
    assert np.allclose(matrices[0], answer)

    # unrestricted (3 spatial orbitals, 6 spin orbitals)
    # NOTE: [np.zeros((4, 4, 4, 4))]*3 results in three matrices with the same reference
    matrices = [np.zeros((4, 4, 4, 4)), np.zeros((4, 4, 4, 4)), np.zeros((4, 4, 4, 4))]
    answer_alpha = np.zeros((4, 4, 4, 4))
    answer_alpha_beta = np.zeros((4, 4, 4, 4))
    answer_beta = np.zeros((4, 4, 4, 4))
    def check_answer_unrestricted(matrices, answer_alpha, answer_alpha_beta, answer_beta):
        """ Checks unrestricted answers"""
        assert np.allclose(matrices[0], answer_alpha)
        assert np.allclose(matrices[1], answer_alpha_beta)
        assert np.allclose(matrices[2], answer_beta)

    add_two_density(matrices, 0, 1, 2, 3, 1, 'unrestricted')
    answer_alpha[0, 1, 2, 3] += 1
    check_answer_unrestricted(matrices, answer_alpha, answer_alpha_beta, answer_beta)

    add_two_density(matrices, 4, 1, 2, 3, 1, 'unrestricted')
    check_answer_unrestricted(matrices, answer_alpha, answer_alpha_beta, answer_beta)
    add_two_density(matrices, 0, 5, 2, 3, 1, 'unrestricted')
    check_answer_unrestricted(matrices, answer_alpha, answer_alpha_beta, answer_beta)
    add_two_density(matrices, 0, 1, 6, 3, 1, 'unrestricted')
    check_answer_unrestricted(matrices, answer_alpha, answer_alpha_beta, answer_beta)
    add_two_density(matrices, 0, 1, 2, 7, 1, 'unrestricted')
    check_answer_unrestricted(matrices, answer_alpha, answer_alpha_beta, answer_beta)

    add_two_density(matrices, 4, 5, 2, 3, 1, 'unrestricted')
    check_answer_unrestricted(matrices, answer_alpha, answer_alpha_beta, answer_beta)
    add_two_density(matrices, 4, 1, 6, 3, 1, 'unrestricted')
    # this is beta-alpha-beta-alpha, and only the alpha-beta-alpha-beta matrix is stored
    check_answer_unrestricted(matrices, answer_alpha, answer_alpha_beta, answer_beta)
    add_two_density(matrices, 4, 1, 2, 7, 1, 'unrestricted')
    check_answer_unrestricted(matrices, answer_alpha, answer_alpha_beta, answer_beta)
    add_two_density(matrices, 0, 5, 6, 3, 1, 'unrestricted')
    check_answer_unrestricted(matrices, answer_alpha, answer_alpha_beta, answer_beta)
    add_two_density(matrices, 0, 5, 2, 7, 1, 'unrestricted')
    answer_alpha_beta[0, 1, 2, 3] += 1
    check_answer_unrestricted(matrices, answer_alpha, answer_alpha_beta, answer_beta)
    add_two_density(matrices, 0, 1, 6, 7, 1, 'unrestricted')
    check_answer_unrestricted(matrices, answer_alpha, answer_alpha_beta, answer_beta)

    add_two_density(matrices, 4, 5, 6, 3, 1, 'unrestricted')
    check_answer_unrestricted(matrices, answer_alpha, answer_alpha_beta, answer_beta)
    add_two_density(matrices, 4, 5, 2, 7, 1, 'unrestricted')
    check_answer_unrestricted(matrices, answer_alpha, answer_alpha_beta, answer_beta)
    add_two_density(matrices, 4, 1, 6, 7, 1, 'unrestricted')
    check_answer_unrestricted(matrices, answer_alpha, answer_alpha_beta, answer_beta)
    add_two_density(matrices, 0, 5, 6, 7, 1, 'unrestricted')
    check_answer_unrestricted(matrices, answer_alpha, answer_alpha_beta, answer_beta)

    add_two_density(matrices, 4, 5, 6, 7, 1, 'unrestricted')
    answer_beta[0, 1, 2, 3] += 1
    check_answer_unrestricted(matrices, answer_alpha, answer_alpha_beta, answer_beta)

    # generalized (6 orbitals)
    matrices = [np.zeros((8, 8, 8, 8))]
    answer = np.zeros((8, 8, 8, 8))

    add_two_density(matrices, 0, 1, 2, 3, 1, 'generalized')
    answer[0, 1, 2, 3] += 1
    assert np.allclose(matrices[0], answer)

    add_two_density(matrices, 4, 1, 2, 3, 1, 'generalized')
    answer[4, 1, 2, 3] += 1
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 0, 5, 2, 3, 1, 'generalized')
    answer[0, 5, 2, 3] += 1
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 0, 1, 6, 3, 1, 'generalized')
    answer[0, 1, 6, 3] += 1
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 0, 1, 2, 7, 1, 'generalized')
    answer[0, 1, 2, 7] += 1
    assert np.allclose(matrices[0], answer)

    add_two_density(matrices, 4, 5, 2, 3, 1, 'generalized')
    answer[4, 5, 2, 3] += 1
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 4, 1, 6, 3, 1, 'generalized')
    answer[4, 1, 6, 3] += 1
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 4, 1, 2, 7, 1, 'generalized')
    answer[4, 1, 2, 7] += 1
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 0, 5, 6, 3, 1, 'generalized')
    answer[0, 5, 6, 3] += 1
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 0, 5, 2, 7, 1, 'generalized')
    answer[0, 5, 2, 7] += 1
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 0, 1, 6, 7, 1, 'generalized')
    answer[0, 1, 6, 7] += 1
    assert np.allclose(matrices[0], answer)

    add_two_density(matrices, 4, 5, 6, 3, 1, 'generalized')
    answer[4, 5, 6, 3] += 1
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 4, 5, 2, 7, 1, 'generalized')
    answer[4, 5, 2, 7] += 1
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 4, 1, 6, 7, 1, 'generalized')
    answer[4, 1, 6, 7] += 1
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 0, 5, 6, 7, 1, 'generalized')
    answer[0, 5, 6, 7] += 1
    assert np.allclose(matrices[0], answer)

    add_two_density(matrices, 4, 5, 6, 7, 1, 'generalized')
    answer[4, 5, 6, 7] += 1
    assert np.allclose(matrices[0], answer)


def test_density_matrix():
    """ Tests density.density_matrix
    """
    # check type
    assert_raises(TypeError, lambda: density_matrix(np.arange(1, 5),
                                                    [0b0101, 0b1001, 0b0110, 0b1010], 2,
                                                    is_chemist_notation=False, val_threshold=0,
                                                    orbtype='daslkfjaslkdf'))
    # restricted
    one_density_r, two_density_r = density_matrix(np.arange(1, 4), [0b0101, 0b1010, 0b0110], 2,
                                                  is_chemist_notation=False, val_threshold=0,
                                                  orbtype='restricted')
    assert np.allclose(one_density_r, np.array([[1*1 + 1*1 + 3*3, 1*3 + 2*3],
                                                [1*3 + 2*3, 2*2 + 2*2 + 3*3]]))
    assert np.allclose(two_density_r, np.array([[[[1*1 + 1*1, 1*3], [1*3, 1*2 + 1*2]],
                                                 [[1*3, 3*3], [0, 1*3 + 1*3]]],
                                                [[[1*3, 0], [3*3, 1*3 + 1*3]],
                                                 [[2*1 + 2*1, 2*3], [2*3, 2*2 + 2*2]]]]))
    # unrestricted
    one_density_u, two_density_u = density_matrix(np.arange(1, 4), [0b0101, 0b1010, 0b0110], 2,
                                                  is_chemist_notation=False, val_threshold=0,
                                                  orbtype='unrestricted')
    assert np.allclose(one_density_u[0], np.array([[1*1, 1*3],
                                                   [1*3, 2*2 + 3*3]]))
    assert np.allclose(one_density_u[1], np.array([[1*1 + 3*3, 2*3],
                                                   [2*3, 2*2]]))
    assert np.allclose(two_density_u[0], np.array([[[[0, 0], [0, 0]],
                                                    [[0, 0], [0, 0]]],
                                                   [[[0, 0], [0, 0]],
                                                    [[0, 0], [0, 0]]]]))
    assert np.allclose(two_density_u[1], np.array([[[[1*1, 0], [1*3, 1*2]],
                                                    [[0, 0], [0, 0]]],
                                                   [[[1*3, 0], [3*3, 1*3 + 1*3]],
                                                    [[2*1, 0], [2*3, 2*2]]]]))
    assert np.allclose(two_density_u[2], np.array([[[[0, 0], [0, 0]],
                                                    [[0, 0], [0, 0]]],
                                                   [[[0, 0], [0, 0]],
                                                    [[0, 0], [0, 0]]]]))
    # generalized
    one_density_g, two_density_g = density_matrix(np.arange(1, 4), [0b0101, 0b1010, 0b0110], 2,
                                                  is_chemist_notation=False, val_threshold=0,
                                                  orbtype='generalized')
    assert np.allclose(one_density_g[0], np.array([[1*1, 1*3, 0, 0],
                                                   [1*3, 2*2 + 3*3, 0, 0],
                                                   [0, 0, 1*1 + 3*3, 2*3],
                                                   [0, 0, 2*3, 2*2]]))
    assert np.allclose(two_density_g[0],
                       np.array([[[[0, 0, 0, 0]]*4, [[0, 0, 0, 0]]*4,
                                  [[0, 0, 1, 0], [0, 0, 3, 2.], [-1, -3, 0, 0], [0, -2, 0, 0]],
                                  [[0, 0, 0, 0]]*4],
                                 [[[0, 0, 0, 0]]*4, [[0, 0, 0, 0]]*4,
                                  [[0, 0, 3, 0], [0, 0, 9, 6], [-3, -9, 0, 0], [0, -6, 0, 0]],
                                  [[0, 0, 2, 0], [0, 0, 6, 4], [-2, -6, 0, 0], [0, -4, 0, 0]]],
                                 [[[0, 0, -1, 0], [0, 0, -3, -2.], [1, 3, 0, 0], [0, 2, 0, 0]],
                                  [[0, 0, -3, 0], [0, 0, -9, -6.], [3, 9, 0, 0], [0, 6, 0, 0]],
                                  [[0, 0, 0, 0]]*4, [[0, 0, 0, 0]]*4],
                                 [[[0, 0, 0, 0]]*4,
                                  [[0, 0, -2, 0], [0, 0, -6, -4], [2, 6, 0, 0], [0, 4, 0, 0]],
                                  [[0, 0, 0, 0]]*4, [[0, 0, 0, 0]]*4]]))
    # compare restricted with unrestricted
    assert np.allclose(one_density_r[0], one_density_u[0] + one_density_u[1])
    assert np.allclose(two_density_r[0], (two_density_u[0] + two_density_u[2] + two_density_u[1] +
                                          two_density_u[1].transpose((1, 0, 3, 2))))

    # truncate
    one_density, two_density = density_matrix(np.arange(1, 4), [0b0101, 0b1010, 0b0110], 2,
                                              is_chemist_notation=False, val_threshold=2,
                                              orbtype='restricted')
    assert np.allclose(one_density, np.array([[3*3, 1*3 + 2*3],
                                              [1*3 + 2*3, 2*2 + 2*2 + 3*3]]))
    assert np.allclose(two_density, np.array([[[[0, 1*3], [1*3, 1*2 + 1*2]],
                                               [[1*3, 3*3], [0, 1*3 + 1*3]]],
                                              [[[1*3, 0], [3*3, 1*3 + 1*3]],
                                               [[2*1 + 2*1, 2*3], [2*3, 2*2 + 2*2]]]]))

    one_density, two_density = density_matrix(np.arange(1, 4), [0b0101, 0b1010, 0b0110], 2,
                                              is_chemist_notation=False, val_threshold=17,
                                              orbtype='restricted')
    assert np.allclose(one_density, np.array([[3*3, 1*3 + 2*3],
                                              [1*3 + 2*3, 3*3]]))
    assert np.allclose(two_density, np.array([[[[0, 1*3], [1*3, 0]],
                                               [[1*3, 3*3], [0, 1*3 + 1*3]]],
                                              [[[1*3, 0], [3*3, 1*3 + 1*3]],
                                               [[0, 2*3], [2*3, 0]]]]))

    one_density, two_density = density_matrix(np.arange(1, 4), [0b0101, 0b1010, 0b0110], 2,
                                              is_chemist_notation=False, val_threshold=28,
                                              orbtype='restricted')
    assert np.allclose(one_density, np.array([[3*3, 2*3],
                                              [2*3, 3*3]]))
    assert np.allclose(two_density, np.array([[[[0, 0], [0, 0]],
                                               [[0, 3*3], [0, 1*3 + 1*3]]],
                                              [[[0, 0], [3*3, 1*3 + 1*3]],
                                               [[0, 2*3], [2*3, 0]]]]))

    # break particle number symmetry
    one_density, two_density = density_matrix(np.arange(1, 3), [0b1111, 0b0001], 2,
                                              is_chemist_notation=False, val_threshold=0,
                                              orbtype='restricted')
    assert np.allclose(one_density, np.array([[1*1 + 1*1 + 2*2, 0],
                                              [0, 1*1 + 1*1]]))
    assert np.allclose(two_density, np.array([[[[1*1 + 1*1, 0], [0, 0]],
                                               [[0, 1*1 + 1*1 + 1*1 + 1*1], [-1*1 - 1*1, 0]]],
                                              [[[0, -1*1 - 1*1], [1*1 + 1*1 + 1*1 + 1*1, 0]],
                                               [[0, 0], [0, 1*1 + 1*1]]]]))
# Tests using examples
def test_density_matrix_restricted_h2_fci_sto6g():
    """ Tests density.density_matrix using H2 FCI/STO-6G

    Uses numbers obtained from Gaussian and PySCF
        Gaussian's one electron density matrix is used to compare
        PySCF's SD coefficient is used to construct density matrix
        Electronic energy of FCI from Gaussian and PySCF were the same

    Coefficients for the Slater determinants (from PySCF) are [0.993594152, 0.0, 0.0, -0.113007352]
    FCI Electronic energy is -1.85908985 Hartree
    """
    sd_coeffs = np.array([0.993594152, 0.0, 0.0, -0.113007352])
    one_density, two_density = density_matrix(sd_coeffs, [0b0101, 0b1001, 0b0110, 0b1010], 2,
                                              is_chemist_notation=False, val_threshold=0,
                                              orbtype='restricted')
    # Count number of electrons
    assert abs(np.einsum('ii', one_density[0]) - 2) < 1e-8

    # Compare to reference
    ref_one_density = np.array([[0.197446E+01, -0.163909E-14],
                                [-0.163909E-14, 0.255413E-01]])
    assert np.allclose(one_density[0], ref_one_density)

    # Reconstruct FCI energy
    hf_dict = gaussian_fchk(os.path.join(os.path.dirname(package_path), '..', 'data', 'test',
                                         'h2_hf_sto6g.fchk'))
    one_int = hf_dict["one_int"][0]
    two_int = hf_dict["two_int"][0]
    # physicist notation
    assert abs((np.einsum('ij,ij', one_int, one_density[0]) +
                0.5*np.einsum('ijkl,ijkl', two_int, two_density[0])) - (-1.85908985)) < 1e-8
    # chemist notation
    one_density, two_density = density_matrix(sd_coeffs, [0b0101, 0b1001, 0b0110, 0b1010], 2,
                                              is_chemist_notation=True, val_threshold=0,
                                              orbtype='restricted')
    assert abs((np.einsum('ij,ij', one_int, one_density[0]) +
                0.5*np.einsum('ijkl,iklj', two_int, two_density[0])) - (-1.85908985)) < 1e-8


def test_density_matrix_restricted_h2_631gdp():
    """ Tests density.density_matrix using H2 system (FCI/6-31G**)

    Uses numbers obtained from PySCF
        PySCF's SD coefficient is used to construct density matrix
        Electronic energy of FCI

    FCI Electronic energy is -1.87832559 Hartree
    """
    hf_dict = gaussian_fchk(os.path.join(os.path.dirname(package_path), '..', 'data', 'test',
                                         'h2_hf_631gdp.fchk'))

    nelec = 2
    one_int = hf_dict["one_int"][0]
    two_int = hf_dict["two_int"][0]

    # generate ci matrix from pyscf
    # ci_matrix, civec = generate_fci_cimatrix(one_int, two_int, nelec, is_chemist_notation=False)
    ci_matrix = np.load('../../../data/test/h2_hf_631gdp_cimatrix.npy')
    civec = np.load('../../../data/test/h2_hf_631gdp_civec.npy')
    sd_coeffs = eigh(ci_matrix)[1][:, 0]
    energy = eigh(ci_matrix)[0][0]

    one_density, two_density = density_matrix(sd_coeffs, civec, one_int.shape[0],
                                              is_chemist_notation=False, val_threshold=0,
                                              orbtype='restricted')

    # Count number of electrons
    assert abs(np.einsum('ii', one_density[0]) - nelec) < 1e-8

    # Reconstruct FCI energy
    # physicist notation
    assert abs((np.einsum('ij,ij', one_int, one_density[0]) +
                0.5*np.einsum('ijkl,ijkl', two_int, two_density[0])) - (energy)) < 1e-8
    # chemist notation
    one_density, two_density = density_matrix(sd_coeffs, civec, one_int.shape[0],
                                              is_chemist_notation=True, val_threshold=0,
                                              orbtype='restricted')
    assert abs((np.einsum('ij,ij', one_int, one_density[0]) +
                0.5*np.einsum('ijkl,iklj', two_int, two_density[0])) - (energy)) < 1e-8


def test_density_matrix_restricted_lih_sto6g():
    """ Tests density.density_matrix using LiH system (FCI/STO6G)

    Uses numbers obtained from PySCF
        SD coefficient is used to construct density matrix
        Electronic energy of FCI
    """
    hf_dict = gaussian_fchk(os.path.join(os.path.dirname(package_path), '..', 'data', 'test',
                                         'lih_hf_sto6g.fchk'))

    nelec = 4
    one_int = hf_dict["one_int"][0]
    two_int = hf_dict["two_int"][0]

    # generate ci matrix from pyscf
    # ci_matrix, civec = generate_fci_cimatrix(one_int, two_int, nelec, is_chemist_notation=False)
    ci_matrix = np.load('../../../data/test/lih_hf_sto6g_cimatrix.npy')
    civec = np.load('../../../data/test/lih_hf_sto6g_civec.npy')
    sd_coeffs = eigh(ci_matrix)[1][:, 0]
    energy = eigh(ci_matrix)[0][0]

    one_density, two_density = density_matrix(sd_coeffs, civec, one_int.shape[0],
                                              is_chemist_notation=False, val_threshold=0,
                                              orbtype='restricted')

    # Count number of electrons
    assert abs(np.einsum('ii', one_density[0]) - nelec) < 1e-8

    # Reconstruct FCI energy
    # physicist notation
    assert abs((np.einsum('ij,ij', one_int, one_density[0]) +
                0.5*np.einsum('ijkl,ijkl', two_int, two_density[0])) - (energy)) < 1e-8
    # chemist notation
    one_density, two_density = density_matrix(sd_coeffs, civec, one_int.shape[0],
                                              is_chemist_notation=True, val_threshold=0,
                                              orbtype='restricted')
    assert abs((np.einsum('ij,ij', one_int, one_density[0]) +
                0.5*np.einsum('ijkl,iklj', two_int, two_density[0])) - (energy)) < 1e-8


@attr('slow')
def test_density_matrix_restricted_lih_631g():
    """ Tests density.density_matrix using LiH system (FCI/6-31G)

    Uses numbers obtained from PySCF
        SD coefficient is used to construct density matrix
        Electronic energy of FCI
    """
    hf_dict = gaussian_fchk(os.path.join(os.path.dirname(package_path), '..', 'data', 'test',
                                         'lih_hf_631g.fchk'))

    nelec = 4
    one_int = hf_dict["one_int"][0]
    two_int = hf_dict["two_int"][0]

    # generate ci matrix from pyscf
    # ci_matrix, civec = generate_fci_cimatrix(one_int, two_int, nelec, is_chemist_notation=False)
    ci_matrix = np.load('../../../data/test/lih_hf_631g_cimatrix.npy')
    civec = np.load('../../../data/test/lih_hf_631g_civec.npy')
    sd_coeffs = eigh(ci_matrix)[1][:, 0]
    energy = eigh(ci_matrix)[0][0]

    one_density, two_density = density_matrix(sd_coeffs, civec, one_int.shape[0],
                                              is_chemist_notation=False, val_threshold=0,
                                              orbtype='restricted')

    # Count number of electrons
    assert abs(np.einsum('ii', one_density[0]) - nelec) < 1e-8

    # Reconstruct FCI energy
    # physicist notation
    assert abs((np.einsum('ij,ij', one_int, one_density[0]) +
                0.5*np.einsum('ijkl,ijkl', two_int, two_density[0])) - (energy)) < 1e-8
    # chemist notation
    one_density, two_density = density_matrix(sd_coeffs, civec, one_int.shape[0],
                                              is_chemist_notation=True, val_threshold=0,
                                              orbtype='restricted')
    assert abs((np.einsum('ij,ij', one_int, one_density[0]) +
                0.5*np.einsum('ijkl,iklj', two_int, two_density[0])) - (energy)) < 1e-8


def test_density_matrix_unrestricted_lih_sto6g():
    """ Tests density.density_matrix using LiH system (FCI/STO6G) with unrestricted orbitals

    Uses numbers obtained from PySCF
        SD coefficient is used to construct density matrix
        Electronic energy of FCI
    """
    hf_dict = gaussian_fchk(os.path.join(os.path.dirname(package_path), '..', 'data', 'test',
                                         'lih_hf_sto6g.fchk'))

    nelec = 4
    one_int = hf_dict["one_int"][0]
    two_int = hf_dict["two_int"][0]

    ci_matrix, civec = generate_fci_cimatrix(one_int, two_int, nelec, is_chemist_notation=False)
    sd_coeffs = eigh(ci_matrix)[1][:, 0]
    energy = eigh(ci_matrix)[0][0]

    one_density, two_density = density_matrix(sd_coeffs, civec, one_int.shape[0],
                                              is_chemist_notation=False, val_threshold=0,
                                              orbtype='unrestricted')

    # Count number of electrons
    assert abs(sum(np.einsum('ii', i) for i in one_density) - nelec) < 1e-8

    # Reconstruct FCI energy
    # physicist notation
    assert abs((np.einsum('ij,ij', one_int, one_density[0]) +
                np.einsum('ij,ij', one_int, one_density[1]) +
                0.5*np.einsum('ijkl,ijkl', two_int, two_density[0]) +
                np.einsum('ijkl,ijkl', two_int, two_density[1]) +
                0.5*np.einsum('ijkl,ijkl', two_int, two_density[2]))
               - (energy)) < 1e-8
    # chemist notation
    one_density, two_density = density_matrix(sd_coeffs, civec, one_int.shape[0],
                                              is_chemist_notation=True, val_threshold=0,
                                              orbtype='unrestricted')
    assert abs((np.einsum('ij,ij', one_int, one_density[0]) +
                np.einsum('ij,ij', one_int, one_density[1]) +
                0.5*np.einsum('ijkl,iklj', two_int, two_density[0]) +
                np.einsum('ijkl,iklj', two_int, two_density[1]) +
                0.5*np.einsum('ijkl,iklj', two_int, two_density[2]))
               - (energy)) < 1e-8


def test_density_matrix_generalized_lih_sto6g():
    """ Tests density.density_matrix using LiH system (FCI/STO6G) with generalized orbitals

    Uses numbers obtained from PySCF
        SD coefficient is used to construct density matrix
        Electronic energy of FCI
    """
    hf_dict = gaussian_fchk(os.path.join(os.path.dirname(package_path), '..', 'data', 'test',
                                         'lih_hf_sto6g.fchk'))

    nelec = 4
    one_int = hf_dict["one_int"][0]
    two_int = hf_dict["two_int"][0]

    ci_matrix, civec = generate_fci_cimatrix(one_int, two_int, nelec, is_chemist_notation=False)
    sd_coeffs = eigh(ci_matrix)[1][:, 0]
    energy = eigh(ci_matrix)[0][0]

    one_density, two_density = density_matrix(sd_coeffs, civec, 6,
                                              is_chemist_notation=False, val_threshold=0,
                                              orbtype='generalized')

    # Count number of electrons
    assert abs(np.einsum('ii', one_density[0]) - nelec) < 1e-8

    # Reconstruct FCI energy
    # physicist notation
    one_int = np.vstack((np.hstack((one_int, np.zeros((6, 6)))),
                         np.hstack((np.zeros((6, 6)), one_int))))
    # alpha alpha alpha alpha, beta alpha alpha alpha
    # alpha beta alpha alpha, beta beta alpha alpha
    temp1 = np.concatenate((np.concatenate((two_int, np.zeros((6, 6, 6, 6))), axis=0),
                            np.concatenate((np.zeros((6, 6, 6, 6)), np.zeros((6, 6, 6, 6))),
                                           axis=0)),
                           axis=1)
    # alpha alpha beta alpha, beta alpha beta alpha
    # alpha beta beta alpha, beta beta beta alpha
    temp2 = np.concatenate((np.concatenate((np.zeros((6, 6, 6, 6)), two_int), axis=0),
                            np.concatenate((np.zeros((6, 6, 6, 6)), np.zeros((6, 6, 6, 6))),
                                           axis=0)),
                           axis=1)
    # alpha alpha alpha beta, beta alpha alpha beta
    # alpha beta alpha beta, beta beta alpha beta
    temp3 = np.concatenate((np.concatenate((np.zeros((6, 6, 6, 6)), np.zeros((6, 6, 6, 6))),
                                           axis=0),
                            np.concatenate((two_int, np.zeros((6, 6, 6, 6))), axis=0)),
                           axis=1)
    # alpha alpha beta beta, beta alpha beta beta
    # alpha beta beta beta, beta beta beta beta
    temp4 = np.concatenate((np.concatenate((np.zeros((6, 6, 6, 6)), np.zeros((6, 6, 6, 6))),
                                           axis=0),
                            np.concatenate((np.zeros((6, 6, 6, 6)), two_int), axis=0)),
                           axis=1)
    two_int = np.concatenate((np.concatenate((temp1, temp2), axis=2),
                              np.concatenate((temp3, temp4), axis=2)), axis=3)

    assert abs((np.einsum('ij,ij', one_int, one_density[0]) +
                0.5*np.einsum('ijkl,ijkl', two_int, two_density[0])) - (energy)) < 1e-8
    # chemist notation
    one_density, two_density = density_matrix(sd_coeffs, civec, 6,
                                              is_chemist_notation=True, val_threshold=0,
                                              orbtype='generalized')
    assert abs((np.einsum('ij,ij', one_int, one_density[0]) +
                0.5*np.einsum('ijkl,iklj', two_int, two_density[0])) - (energy)) < 1e-8
