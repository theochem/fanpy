""" Tests wfns.ci.ci_pairs
"""
from __future__ import absolute_import, division, print_function
from nose.tools import assert_raises
import numpy as np
from wfns.sd_list import sd_list
from wfns.wrapper.horton import gaussian_fchk
from wfns.ci.ci_pairs import CIPairs
from wfns.ci.solver import solve
from wfns.proj.geminals.ap1rog import AP1roG
from wfns.proj.solver import solve as proj_solve


def test_generate_civec():
    """ Tests wfns.ci.ci_pairs.CIPairs.generate_civec
    """
    test = CIPairs(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)))
    assert test.generate_civec() == sd_list(test.nelec, test.nspatial, num_limit=None, spin=0,
                                            seniority=0, exc_orders=[2])


def test_to_ap1rog():
    """ Tests wfns.ci.ci_pairs.CIPairs.to_ap1rog
    """
    test = CIPairs(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)), excs=[0, 1, 2])
    test.sd_coeffs = np.arange(1, 10).reshape(3, 3)
    assert_raises(TypeError, test.to_ap1rog, 1.0)
    assert_raises(ValueError, test.to_ap1rog, -1)
    ap1rog = test.to_ap1rog(exc_lvl=0)
    assert np.allclose(ap1rog.params, np.array([4, 7, 14]))
    ap1rog = test.to_ap1rog(exc_lvl=1)
    assert np.allclose(ap1rog.params, np.array([5/2, 8/2, 9.5]))
    ap1rog = test.to_ap1rog(exc_lvl=2)
    assert np.allclose(ap1rog.params, np.array([6/3, 9/3, 8]))


def test_to_ap1rog_h2_sto6g():
    """ Tests wfns.ci.ci_pairs.CIPairs.to_ap1rog using H2 with HF/STO6G orbitals
    """
    hf_dict = gaussian_fchk('test/h2_hf_sto6g.fchk')
    nelec = 2
    one_int = hf_dict["one_int"]
    two_int = hf_dict["two_int"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]

    cipair = CIPairs(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(cipair)
    test_ap1rog = cipair.to_ap1rog(exc_lvl=0)
    test_ap1rog.normalize()
    ap1rog = AP1roG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    proj_solve(ap1rog)
    assert np.allclose(test_ap1rog.params, ap1rog.params)


def test_to_ap1rog_ilh_sto6g():
    """ Tests wfns.ci.ci_pairs.CIPairs.to_ap1rog with LiH with HF/STO6G orbitals
    """
    hf_dict = gaussian_fchk('test/lih_hf_sto6g.fchk')
    nelec = 4
    one_int = hf_dict["one_int"]
    two_int = hf_dict["two_int"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]

    cipair = CIPairs(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(cipair)
    test_ap1rog = cipair.to_ap1rog(exc_lvl=0)
    test_ap1rog.normalize()
    ap1rog = AP1roG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    proj_solve(ap1rog)
    assert np.all(test_ap1rog.params / ap1rog.params > 0.99)
