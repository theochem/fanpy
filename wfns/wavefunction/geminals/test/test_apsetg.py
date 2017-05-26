from __future__ import absolute_import, division, print_function
import numpy as np
from nose.plugins.attrib import attr
import types
# from wfns.solver.solver import solve
from wfns.backend import graphs
from wfns.wavefunction.geminals.apsetg import APsetG
from wfns.tools import find_datafile


class TestAPsetG(APsetG):
    """APsetG that skips initialization."""
    def __init__(self):
        pass


def test_assign_pmatch_generator():
    """Test APsetG.generate_possible_orbpairs"""
    test = TestAPsetG()
    test.assign_nspin(6)
    sd = (0, 1, 2, 3, 4, 5)
    assert isinstance(test.generate_possible_orbpairs(sd), types.GeneratorType)
    for i, j in zip(test.generate_possible_orbpairs(sd),
                    graphs.generate_biclique_pmatch((0, 1, 2), (3, 4, 5))):
        assert i == j


@attr('slow')
def test_apsetg_wavefunction_h2():
    #### H2 ####
    # HF Value :       -1.84444667247
    # Old Code Value : -1.86968284431
    # FCI Value :      -1.87832550029
    nelec = 2
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/h2_hf_631gdp.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile('test/h2_hf_631gdp_oneint.npy'))
    two_int = np.load(find_datafile('test/h2_hf_631gdp_twoint.npy'))
    nuc_nuc = 0.71317683129
    # Solve with Jacobian using energy as a parameter
    apsetg = APsetG(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc)
    solve(apsetg, solver_type='cma_guess')
    results = solve(apsetg, solver_type='least_squares', use_jac=True)
    print('HF energy', -1.84444667247)
    print('new energy', apsetg.compute_energy())
    print('FCI value', -1.87832550029)
    assert results.success
    assert -1.84444667247 > apsetg.compute_energy() > -1.87832550029
    assert abs(apsetg.compute_energy(include_nuc=False) - (-1.86968284431)) < 1e-7


@attr('slow')
def test_apsetg_wavefunction_lih():
    #### LiH ####
    # HF Value :       -8.9472891719
    # Old Code Value : -8.96353105152
    # FCI Value :      -8.96741814557
    nelec = 4
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/lih_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = (np.load(find_datafile('test/lih_hf_sto6g_oneint.npy')), )
    two_int = (np.load(find_datafile('test/lih_hf_sto6g_twoint.npy')), )
    nuc_nuc = 0.995317634356
    # Compare apsetg energy with old code
    # Solve with Jacobian using energy as a parameter
    apsetg = APsetG(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc)
    solve(apsetg, solver_type='cma_guess')
    results = solve(apsetg, solver_type='least_squares', use_jac=True)
    print('HF energy', -8.9472891719)
    print('new energy', apsetg.compute_energy())
    print('FCI value', -8.96741814557)
    assert results.success
    assert -8.9472891719 > apsetg.compute_energy() > -8.96741814557
    assert False
