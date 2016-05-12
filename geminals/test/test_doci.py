from __future__ import absolute_import, division, print_function

from nose.tools import assert_raises
import numpy as np

from geminals import DOCI, FullCI
from geminals.hort import hartreefock
from geminals.math import binomial


def test_init():

    nelec = 2
    nspin = 8
    npair = nelec // 2
    nspatial = nspin // 2
    H = np.ones((nspatial, nspatial))
    G = np.ones((nspatial, nspatial, nspatial, nspatial))
    nuc_nuc = 4.3
    dtype = np.complex128
    nproj = binomial(nspatial, npair)
    x = np.ones(nproj + 1, dtype=dtype)

    geminal = DOCI(nelec, H, G)

    test = lambda z: DOCI(z, H, G)
    assert_raises(ValueError, test, 3)
    assert_raises(TypeError, test, 3.2)

    test = lambda z: DOCI(nelec, z, G)
    assert_raises(ValueError, test, np.ones((nspatial, nspatial + 1)))
    assert_raises(ValueError, test, np.ones((nspatial,)))
    assert_raises(TypeError, test, (H, np.ones((nspatial,))))
    assert_raises(TypeError, test, [[1.0] * nspatial] * nspatial)
    assert_raises(TypeError, test, np.ones((nspatial, nspatial), dtype=int))

    test = lambda z: DOCI(nelec, H, z)
    assert_raises(ValueError, test, np.ones((nspatial, nspatial, nspatial, nspatial + 1)))
    assert_raises(ValueError, test, np.ones((nspatial, nspatial, nspatial)))
    assert_raises(TypeError, test, (G, G))
    assert_raises(TypeError, test, (0, 1, 2))
    assert_raises(TypeError, test, np.ones((nspatial, nspatial, nspatial, nspatial), dtype=int))
    test = lambda z: DOCI(nelec, (H, H), z)
    assert_raises(TypeError, test, G)

    test = lambda z: DOCI(nelec, H, G, nuc_nuc=z)
    assert_raises(TypeError, test, "qq")

    test = lambda z: DOCI(nelec, H, G, dtype=z)
    assert_raises(TypeError, test, str)
    assert_raises(TypeError, test, 1.0)

    test = lambda z: DOCI(nelec, H, G, x=z)
    assert_raises(TypeError, test, np.ones(nproj + 1, dtype=int))
    assert_raises(TypeError, test, [1.0] * (nproj - 1))


def test_civec():

    nelec = 2
    nspin = 8
    nspatial = nspin // 2
    npair = nelec // 2
    H = np.ones((nspatial, nspatial))
    G = np.ones((nspatial, nspatial, nspatial, nspatial))
    nsd = binomial(nspatial, npair)
    geminal = DOCI(nelec, H, G)
    assert len(geminal.civec) == nsd
    #assert np.all(geminal.civec[0, :nelec])
    #assert not np.any(geminal.civec[0, nelec:])
    #assert np.all(geminal.civec[-1, (nspin - nelec):])
    #assert not np.any(geminal.civec[-1, :(nspin - nelec)])
    #assert np.sum(geminal.civec) == nsd * nelec
    #sd_dict = {}
    #for i in range(geminal.nproj):
        #assert tuple(geminal.civec[i, :]) not in sd_dict
        #sd_dict[tuple(geminal.civec[i, :])] = 1


#def test_compute_projection():
#
    #nelec = 2
    #nspin = 8
    #nspatial = nspin // 2
    #H = np.ones((nspatial, nspatial))
    #G = np.ones((nspatial, nspatial, nspatial, nspatial))
    geminal = DOCI(nelec, H, G)
    #assert np.allclose(geminal.compute_projection(0), 1.0)
    #geminal.C[1] = 1.2345
    #assert np.allclose(geminal.compute_projection(1), 1.2345)


def test_solve_eigh():

    nelec = 2
    hf_dict = hartreefock(fn="test/h2.xyz", basis="sto-3g", nelec=nelec)
    e_hf, H, G, nuc_nuc = hf_dict["energy"], hf_dict["H"], hf_dict["G"], hf_dict["nuc_nuc"]
    dgem = DOCI(nelec, H, G, nuc_nuc=nuc_nuc)
    dgem()
    fgem = FullCI(nelec, H, G, nuc_nuc=nuc_nuc)
    fgem()
    assert fgem.compute_energy() <= dgem.compute_energy() < e_hf
