from nose.tools import assert_raises
import numpy as np

from .. import ci_matrix
from .. import slater


def test_is_alpha():
    """
    Tests ci_matrix.is_alpha
    """
    # assert assert_raises(AssertionError, lambda:ci_matrix.is_alpha(0, 0))
    # assert assert_raises(AssertionError, lambda:ci_matrix.is_alpha(0, -1))
    assert ci_matrix.is_alpha(0, 1) is True
    # assert assert_raises(AssertionError, lambda:ci_matrix.is_alpha(-1, 1))
    # assert assert_raises(AssertionError, lambda:ci_matrix.is_alpha(2, 1))
    assert ci_matrix.is_alpha(1, 1) is False

    assert ci_matrix.is_alpha(0, 4) is True
    assert ci_matrix.is_alpha(1, 4) is True
    assert ci_matrix.is_alpha(2, 4) is True
    assert ci_matrix.is_alpha(3, 4) is True
    assert ci_matrix.is_alpha(4, 4) is False
    assert ci_matrix.is_alpha(5, 4) is False
    assert ci_matrix.is_alpha(6, 4) is False
    assert ci_matrix.is_alpha(7, 4) is False


def test_spatial_index():
    """
    Tests ci_matrix.spatial_index
    """
    # assert assert_raises(AssertionError, lambda:ci_matrix.spatial_index(0, 0))
    # assert assert_raises(AssertionError, lambda:ci_matrix.spatial_index(0, -1))
    assert ci_matrix.spatial_index(0, 1) == 0
    # assert assert_raises(AssertionError, lambda:ci_matrix.spatial_index(-1, 1))
    # assert assert_raises(AssertionError, lambda:ci_matrix.spatial_index(2, 1))
    assert ci_matrix.spatial_index(1, 1) == 0

    assert ci_matrix.spatial_index(0, 4) == 0
    assert ci_matrix.spatial_index(1, 4) == 1
    assert ci_matrix.spatial_index(2, 4) == 2
    assert ci_matrix.spatial_index(3, 4) == 3
    assert ci_matrix.spatial_index(4, 4) == 0
    assert ci_matrix.spatial_index(5, 4) == 1
    assert ci_matrix.spatial_index(6, 4) == 2
    assert ci_matrix.spatial_index(7, 4) == 3


def test_get_H_value():
    """
    Tests ci_matrix.get_H_value
    """
    H = (np.arange(16).reshape(4, 4),)
    # restricted
    assert ci_matrix.get_H_value(H, 0, 0, 'restricted') == 0.0
    assert ci_matrix.get_H_value(H, 0, 1, 'restricted') == 1.0
    assert ci_matrix.get_H_value(H, 0, 4, 'restricted') == 0.0
    assert ci_matrix.get_H_value(H, 0, 5, 'restricted') == 0.0
    assert ci_matrix.get_H_value(H, 1, 0, 'restricted') == 4.0
    assert ci_matrix.get_H_value(H, 1, 1, 'restricted') == 5.0
    assert ci_matrix.get_H_value(H, 1, 4, 'restricted') == 0.0
    assert ci_matrix.get_H_value(H, 1, 5, 'restricted') == 0.0
    assert ci_matrix.get_H_value(H, 4, 0, 'restricted') == 0.0
    assert ci_matrix.get_H_value(H, 4, 1, 'restricted') == 0.0
    assert ci_matrix.get_H_value(H, 4, 4, 'restricted') == 0.0
    assert ci_matrix.get_H_value(H, 4, 5, 'restricted') == 1.0
    assert ci_matrix.get_H_value(H, 5, 0, 'restricted') == 0.0
    assert ci_matrix.get_H_value(H, 5, 1, 'restricted') == 0.0
    assert ci_matrix.get_H_value(H, 5, 4, 'restricted') == 4.0
    assert ci_matrix.get_H_value(H, 5, 5, 'restricted') == 5.0
    # unrestricted
    H = (np.arange(16).reshape(4, 4), np.arange(16, 32).reshape(4, 4))
    assert ci_matrix.get_H_value(H, 0, 0, 'unrestricted') == 0.0
    assert ci_matrix.get_H_value(H, 0, 1, 'unrestricted') == 1.0
    assert ci_matrix.get_H_value(H, 0, 4, 'unrestricted') == 0.0
    assert ci_matrix.get_H_value(H, 0, 5, 'unrestricted') == 0.0
    assert ci_matrix.get_H_value(H, 1, 0, 'unrestricted') == 4.0
    assert ci_matrix.get_H_value(H, 1, 1, 'unrestricted') == 5.0
    assert ci_matrix.get_H_value(H, 1, 4, 'unrestricted') == 0.0
    assert ci_matrix.get_H_value(H, 1, 5, 'unrestricted') == 0.0
    assert ci_matrix.get_H_value(H, 4, 0, 'unrestricted') == 0.0
    assert ci_matrix.get_H_value(H, 4, 1, 'unrestricted') == 0.0
    print ci_matrix.get_H_value(H, 4, 4, 'unrestricted')
    assert ci_matrix.get_H_value(H, 4, 4, 'unrestricted') == 16.0
    assert ci_matrix.get_H_value(H, 4, 5, 'unrestricted') == 17.0
    assert ci_matrix.get_H_value(H, 5, 0, 'unrestricted') == 0.0
    assert ci_matrix.get_H_value(H, 5, 1, 'unrestricted') == 0.0
    assert ci_matrix.get_H_value(H, 5, 4, 'unrestricted') == 20.0
    assert ci_matrix.get_H_value(H, 5, 5, 'unrestricted') == 21.0
    # generalized
    H = (np.arange(64).reshape(8, 8),)
    assert ci_matrix.get_H_value(H, 0, 0, 'generalized') == 0.0
    assert ci_matrix.get_H_value(H, 0, 1, 'generalized') == 1.0
    assert ci_matrix.get_H_value(H, 0, 4, 'generalized') == 4.0
    assert ci_matrix.get_H_value(H, 0, 5, 'generalized') == 5.0
    assert ci_matrix.get_H_value(H, 1, 0, 'generalized') == 8.0
    assert ci_matrix.get_H_value(H, 1, 1, 'generalized') == 9.0
    assert ci_matrix.get_H_value(H, 1, 4, 'generalized') == 12.0
    assert ci_matrix.get_H_value(H, 1, 5, 'generalized') == 13.0
    assert ci_matrix.get_H_value(H, 4, 0, 'generalized') == 32.0
    assert ci_matrix.get_H_value(H, 4, 1, 'generalized') == 33.0
    assert ci_matrix.get_H_value(H, 4, 4, 'generalized') == 36.0
    assert ci_matrix.get_H_value(H, 4, 5, 'generalized') == 37.0
    assert ci_matrix.get_H_value(H, 5, 0, 'generalized') == 40.0
    assert ci_matrix.get_H_value(H, 5, 1, 'generalized') == 41.0
    assert ci_matrix.get_H_value(H, 5, 4, 'generalized') == 44.0
    assert ci_matrix.get_H_value(H, 5, 5, 'generalized') == 45.0


def test_get_G_value():
    """
    Tests ci_matrix.get_G_value
    """
    # restricted
    G = (np.arange(256).reshape(4, 4, 4, 4),)
    assert ci_matrix.get_G_value(G, 0, 0, 0, 1, 'restricted') == 1.0
    assert ci_matrix.get_G_value(G, 0, 0, 4, 1, 'restricted') == 0.0
    assert ci_matrix.get_G_value(G, 0, 4, 0, 1, 'restricted') == 0.0
    assert ci_matrix.get_G_value(G, 4, 0, 0, 1, 'restricted') == 0.0
    assert ci_matrix.get_G_value(G, 0, 4, 4, 1, 'restricted') == 0.0
    assert ci_matrix.get_G_value(G, 4, 0, 4, 1, 'restricted') == 1.0
    assert ci_matrix.get_G_value(G, 4, 4, 0, 1, 'restricted') == 0.0
    assert ci_matrix.get_G_value(G, 4, 4, 4, 1, 'restricted') == 0.0
    assert ci_matrix.get_G_value(G, 0, 0, 0, 5, 'restricted') == 0.0
    assert ci_matrix.get_G_value(G, 0, 0, 4, 5, 'restricted') == 0.0
    assert ci_matrix.get_G_value(G, 0, 4, 0, 5, 'restricted') == 1.0
    assert ci_matrix.get_G_value(G, 4, 0, 0, 5, 'restricted') == 0.0
    assert ci_matrix.get_G_value(G, 0, 4, 4, 5, 'restricted') == 0.0
    assert ci_matrix.get_G_value(G, 4, 0, 4, 5, 'restricted') == 0.0
    assert ci_matrix.get_G_value(G, 4, 4, 0, 5, 'restricted') == 0.0
    assert ci_matrix.get_G_value(G, 4, 4, 4, 5, 'restricted') == 1.0
    # unrestricted
    G = (np.arange(256).reshape(4, 4, 4, 4),
         np.arange(256, 512).reshape(4, 4, 4, 4),
         np.arange(512, 768).reshape(4, 4, 4, 4))
    assert ci_matrix.get_G_value(G, 0, 0, 0, 1, 'unrestricted') == 1.0
    assert ci_matrix.get_G_value(G, 0, 0, 4, 1, 'unrestricted') == 0.0
    assert ci_matrix.get_G_value(G, 0, 4, 0, 1, 'unrestricted') == 0.0
    assert ci_matrix.get_G_value(G, 4, 0, 0, 1, 'unrestricted') == 0.0
    assert ci_matrix.get_G_value(G, 0, 4, 4, 1, 'unrestricted') == 0.0
    assert ci_matrix.get_G_value(G, 4, 0, 4, 1, 'unrestricted') == 260.0
    assert ci_matrix.get_G_value(G, 4, 4, 0, 1, 'unrestricted') == 0.0
    assert ci_matrix.get_G_value(G, 4, 4, 4, 1, 'unrestricted') == 0.0
    assert ci_matrix.get_G_value(G, 0, 0, 0, 5, 'unrestricted') == 0.0
    assert ci_matrix.get_G_value(G, 0, 0, 4, 5, 'unrestricted') == 0.0
    assert ci_matrix.get_G_value(G, 0, 4, 0, 5, 'unrestricted') == 257.0
    assert ci_matrix.get_G_value(G, 4, 0, 0, 5, 'unrestricted') == 0.0
    assert ci_matrix.get_G_value(G, 0, 4, 4, 5, 'unrestricted') == 0.0
    assert ci_matrix.get_G_value(G, 4, 0, 4, 5, 'unrestricted') == 0.0
    assert ci_matrix.get_G_value(G, 4, 4, 0, 5, 'unrestricted') == 0.0
    assert ci_matrix.get_G_value(G, 4, 4, 4, 5, 'unrestricted') == 513.0
    # generalized
    G = (np.arange(4096).reshape(8, 8, 8, 8),)
    assert ci_matrix.get_G_value(G, 0, 0, 0, 1, 'generalized') == 1.0
    assert ci_matrix.get_G_value(G, 0, 0, 4, 1, 'generalized') == 33.0
    assert ci_matrix.get_G_value(G, 0, 4, 0, 1, 'generalized') == 257.0
    assert ci_matrix.get_G_value(G, 4, 0, 0, 1, 'generalized') == 2049.0
    assert ci_matrix.get_G_value(G, 0, 4, 4, 1, 'generalized') == 289.0
    assert ci_matrix.get_G_value(G, 4, 0, 4, 1, 'generalized') == 2081.0
    assert ci_matrix.get_G_value(G, 4, 4, 0, 1, 'generalized') == 2305.0
    assert ci_matrix.get_G_value(G, 4, 4, 4, 1, 'generalized') == 2337.0
    assert ci_matrix.get_G_value(G, 0, 0, 0, 5, 'generalized') == 5.0
    assert ci_matrix.get_G_value(G, 0, 0, 4, 5, 'generalized') == 37.0
    assert ci_matrix.get_G_value(G, 0, 4, 0, 5, 'generalized') == 261.0
    assert ci_matrix.get_G_value(G, 4, 0, 0, 5, 'generalized') == 2053.0
    assert ci_matrix.get_G_value(G, 0, 4, 4, 5, 'generalized') == 293.0
    assert ci_matrix.get_G_value(G, 4, 0, 4, 5, 'generalized') == 2085.0
    assert ci_matrix.get_G_value(G, 4, 4, 0, 5, 'generalized') == 2309.0
    assert ci_matrix.get_G_value(G, 4, 4, 4, 5, 'generalized') == 2341.0


def test_ci_matrix():
    """
    Tests ci_matrix.ci_matrix
    """
    pass


def test_doci_matrix():
    """
    Tests ci_matrix.doci_matrix
    """
    pass
