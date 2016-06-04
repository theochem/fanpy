from __future__ import absolute_import, division, print_function

import numpy as np

from ..math_tools import binomial, permanent_combinatoric, permanent_ryser


def test_binomial():

    assert binomial(1000,    10) == 263409560461970212832400
    assert binomial(1000.99, 10) == 263409560461970212832400
    assert binomial(1001,    10) == 266067578226470416796400


def test_permanent_combinatoric():

    assert np.allclose(permanent_combinatoric(np.arange(1, 10).reshape(3, 3)), 450)
    assert np.allclose(permanent_combinatoric(np.arange(1, 13).reshape(3, 4)), 3900)


def test_permanent_ryser():

    for i in range(1, 6):
        for j in range(1, 6):
            assert np.allclose(
                permanent_ryser(np.arange(1, i * j + 1).reshape(i, j)),
                permanent_combinatoric(np.arange(1, i * j + 1).reshape(i, j)),
            )
