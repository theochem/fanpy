import numpy as np

import sys
sys.path.append('../')
import ci_matrix
import slater
from common import raises_exception

def test_is_alpha():
    """
    Tests ci_matrix.is_alpha
    """
    # assert raises_exception(lambda:ci_matrix.is_alpha(0, 0))
    # assert raises_exception(lambda:ci_matrix.is_alpha(0, -1))
    assert ci_matrix.is_alpha(0, 1) == True
    # assert raises_exception(lambda:ci_matrix.is_alpha(-1, 1))
    # assert raises_exception(lambda:ci_matrix.is_alpha(2, 1))
    assert ci_matrix.is_alpha(1, 1) == False

    assert ci_matrix.is_alpha(0, 4) == True
    assert ci_matrix.is_alpha(1, 4) == True
    assert ci_matrix.is_alpha(2, 4) == True
    assert ci_matrix.is_alpha(3, 4) == True
    assert ci_matrix.is_alpha(4, 4) == False
    assert ci_matrix.is_alpha(5, 4) == False
    assert ci_matrix.is_alpha(6, 4) == False
    assert ci_matrix.is_alpha(7, 4) == False

def test_spatial_index():
    """
    Tests ci_matrix.spatial_index
    """
    # assert raises_exception(lambda:ci_matrix.spatial_index(0, 0))
    # assert raises_exception(lambda:ci_matrix.spatial_index(0, -1))
    assert ci_matrix.spatial_index(0, 1) == 0
    # assert raises_exception(lambda:ci_matrix.spatial_index(-1, 1))
    # assert raises_exception(lambda:ci_matrix.spatial_index(2, 1))
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
    H = np.arange(16).reshape(4,4)
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
    H = np.arange(64).reshape(8,8)
    assert ci_matrix.get_H_value(H, 0, 0, 'unrestricted') == 0.0
    assert ci_matrix.get_H_value(H, 0, 1, 'unrestricted') == 1.0
    assert ci_matrix.get_H_value(H, 0, 4, 'unrestricted') == 0.0
    assert ci_matrix.get_H_value(H, 0, 5, 'unrestricted') == 0.0
    assert ci_matrix.get_H_value(H, 1, 0, 'unrestricted') == 8.0
    assert ci_matrix.get_H_value(H, 1, 1, 'unrestricted') == 9.0
    assert ci_matrix.get_H_value(H, 1, 4, 'unrestricted') == 0.0
    assert ci_matrix.get_H_value(H, 1, 5, 'unrestricted') == 0.0
    assert ci_matrix.get_H_value(H, 4, 0, 'unrestricted') == 0.0
    assert ci_matrix.get_H_value(H, 4, 1, 'unrestricted') == 0.0
    assert ci_matrix.get_H_value(H, 4, 4, 'unrestricted') == 36.0
    assert ci_matrix.get_H_value(H, 4, 5, 'unrestricted') == 37.0
    assert ci_matrix.get_H_value(H, 5, 0, 'unrestricted') == 0.0
    assert ci_matrix.get_H_value(H, 5, 1, 'unrestricted') == 0.0
    assert ci_matrix.get_H_value(H, 5, 4, 'unrestricted') == 44.0
    assert ci_matrix.get_H_value(H, 5, 5, 'unrestricted') == 45.0
    # generalized
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
    G = np.arange(256).reshape(4,4,4,4)
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
    G = np.arange(4096).reshape(8,8,8,8)
    assert ci_matrix.get_G_value(G, 0, 0, 0, 1, 'unrestricted') == 1.0
    assert ci_matrix.get_G_value(G, 0, 0, 4, 1, 'unrestricted') == 0.0
    assert ci_matrix.get_G_value(G, 0, 4, 0, 1, 'unrestricted') == 0.0
    assert ci_matrix.get_G_value(G, 4, 0, 0, 1, 'unrestricted') == 0.0
    assert ci_matrix.get_G_value(G, 0, 4, 4, 1, 'unrestricted') == 0.0
    assert ci_matrix.get_G_value(G, 4, 0, 4, 1, 'unrestricted') == 2081.0
    assert ci_matrix.get_G_value(G, 4, 4, 0, 1, 'unrestricted') == 0.0
    assert ci_matrix.get_G_value(G, 4, 4, 4, 1, 'unrestricted') == 0.0
    assert ci_matrix.get_G_value(G, 0, 0, 0, 5, 'unrestricted') == 0.0
    assert ci_matrix.get_G_value(G, 0, 0, 4, 5, 'unrestricted') == 0.0
    assert ci_matrix.get_G_value(G, 0, 4, 0, 5, 'unrestricted') == 261.0
    assert ci_matrix.get_G_value(G, 4, 0, 0, 5, 'unrestricted') == 0.0
    assert ci_matrix.get_G_value(G, 0, 4, 4, 5, 'unrestricted') == 0.0
    assert ci_matrix.get_G_value(G, 4, 0, 4, 5, 'unrestricted') == 0.0
    assert ci_matrix.get_G_value(G, 4, 4, 0, 5, 'unrestricted') == 0.0
    assert ci_matrix.get_G_value(G, 4, 4, 4, 5, 'unrestricted') == 2341.0
    # generalized
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
