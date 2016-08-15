from __future__ import absolute_import, division, print_function

import numpy as np

from geminals.math_tools import binomial, permanent_combinatoric, permanent_ryser, permanent_borchardt


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


def test_permanent_borchardt_square():
    p = 6
    params = np.random.rand(3*p)
    lambdas = params[:p]
    epsilons = params[p:2*p]
    zetas = params[2*p:]
    gem_coeffs = zetas / (lambdas[:, np.newaxis] - epsilons)

    assert np.allclose(permanent_combinatoric(gem_coeffs),
                       permanent_borchardt(lambdas, epsilons, zetas))

def test_permanent_borchardt_rectangular():
    #FIXME this fails if x_i != 1, because perm(A*D) != perm(A)*perm(D) when A rectangular (D = diagonal)
    p = 6
    k = 8
    params = np.random.rand(p + 2*k)
    lambdas = params[:p]
    epsilons = params[p:p+k]
    zetas = params[p+k:p+2*k]
    gem_coeffs = zetas / (lambdas[:, np.newaxis] - epsilons)
    perm_comb = permanent_combinatoric(gem_coeffs)
    perm_borch = permanent_borchardt(lambdas, epsilons, zetas)
    perm_rys = permanent_ryser(gem_coeffs)
    assert np.allclose(perm_comb, perm_rys)
    assert np.allclose(perm_comb, perm_borch)
    assert np.allclose(perm_rys, perm_borch)
