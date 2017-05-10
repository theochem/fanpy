from __future__ import absolute_import, division, print_function
import numpy as np
from nose.plugins.attrib import attr
from wfns.solver.solver import solve
from wfns.wavefunction.apseqg import APseqG
from wfns.tools import find_datafile

def test_find_gem_indices():
    class TempAPseqG(APseqG):
        nspatial = 4
        seq_list = [0]
        def __init__(self):
            self.dict_orbpair_gem = {}
            self.dict_gem_orbpair = {}

    # sequence 0
    apseq = TempAPseqG()
    apseq.find_gem_indices(0b00110011, raise_error=False)
    answer = [(0, 4), (1, 5)]
    for gem in answer:
        assert gem in apseq.dict_orbpair_gem.keys()
    assert len(answer) == len(apseq.dict_orbpair_gem)

    # sequence 1
    apseq.seq_list = [1]
    apseq.dict_orbpair_gem = {}
    apseq.dict_gem_orbpair = {}
    apseq.find_gem_indices(0b00110011, raise_error=False)
    answer = [(0, 1), (4, 5)]
    for gem in answer:
        assert gem in apseq.dict_orbpair_gem.keys()
    assert len(answer) == len(apseq.dict_orbpair_gem)

    # sequence 0 and 1
    apseq.seq_list = [0, 1]
    apseq.dict_orbpair_gem = {}
    apseq.dict_gem_orbpair = {}
    apseq.find_gem_indices(0b00110011, raise_error=False)
    answer = [(0, 4), (1, 5)]
    # because orbitals (0, 4, 1, 5) are selected already in sequence 0
    for gem in answer:
        assert gem in apseq.dict_orbpair_gem.keys()
    assert len(answer) == len(apseq.dict_orbpair_gem)

    # another example
    apseq.seq_list = [0, 1]
    apseq.dict_orbpair_gem = {}
    apseq.dict_gem_orbpair = {}
    apseq.find_gem_indices(0b01110001, raise_error=False)
    answer = [(0, 4), (5, 6)]
    # first, the orbitals 0 and 4 are removed from seq 0 selection
    # then, only orbitals 5 and 6 are availble for seq 1 selection
    for gem in answer:
        assert gem in apseq.dict_orbpair_gem.keys()
    assert len(answer) == len(apseq.dict_orbpair_gem)

    # another example
    apseq.seq_list = [1]
    apseq.dict_orbpair_gem = {}
    apseq.dict_gem_orbpair = {}
    apseq.find_gem_indices(0b00110110, raise_error=False)
    answer = [(1, 2), (4, 5)]
    print(apseq.dict_orbpair_gem.keys())
    for gem in answer:
        assert gem in apseq.dict_orbpair_gem.keys()
    assert len(answer) == len(apseq.dict_orbpair_gem)

def test_config_gem_config():
    class TempAPseqG(APseqG):
        nspatial = 4
        nspin = 8
        seq_list = [0]
        pspace = [0b00110011]
        def __init__(self):
            self.dict_orbpair_gem = {}
            self.dict_gem_orbpair = {}
    # NOTE: pspace is 0b00110101

    # sequence 0
    apseq = TempAPseqG()
    apseq.config_gem_orbpair()
    answer = [(0, 4), (1, 5), (2, 6), (3, 7), (2, 5), (3, 6), (1, 4)]
    for gem in answer:
        assert gem in apseq.dict_orbpair_gem.keys()
    assert len(answer) == len(apseq.dict_orbpair_gem)

    # sequence 1
    apseq.seq_list = [1]
    apseq.dict_orbpair_gem = {}
    apseq.dict_gem_orbpair = {}
    answer = [(0, 1), (4, 5), (1, 2), (5, 6), (2, 3), (6, 7)]
    apseq.config_gem_orbpair()
    for gem in answer:
        assert gem in apseq.dict_orbpair_gem.keys()
    assert len(answer) == len(apseq.dict_orbpair_gem)


@attr('slow')
def test_apseqg_wavefunction_h2():
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
    # Reproduce HF energy
    apseqg = APseqG(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc, seq_list=[0])
    apseqg.params[:-1] = apseqg.template_coeffs.flatten()
    apseqg.cache = {}
    apseqg.d_cache = {}
    assert abs(apseqg.compute_energy(include_nuc=False, ref_sds=apseqg.default_ref_sds) - (-1.84444667247)) < 1e-7
    # Solve with Jacobian using energy as a parameter
    apseqg = APseqG(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc)
    solve(apseqg, solver_type='cma_guess')
    results = solve(apseqg, solver_type='least_squares', use_jac=True)
    print('HF energy', -1.84444667247)
    print('APseqG energy', apseqg.compute_energy())
    print('FCI value', -1.87832550029)
    assert results.success
    assert -1.84444667247 > apseqg.compute_energy() > -1.87832550029
    assert False


@attr('slow')
def test_apseqg_wavefunction_lih():
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
    # Reproduce HF energy
    apseqg = APseqG(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc, seq_list=[0])
    apseqg.params[:-1] = apseqg.template_coeffs.flatten()
    apseqg.cache = {}
    apseqg.d_cache = {}
    assert abs(apseqg.compute_energy(include_nuc=False, ref_sds=apseqg.default_ref_sds) - (-8.9472891719)) < 1e-7
    # Solve with Jacobian using energy as a parameter
    apseqg = APseqG(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc)
    # print(apseqg.params[:-1].reshape(apseqg.template_coeffs.shape))
    solve(apseqg, solver_type='cma_guess')
    results = solve(apseqg, solver_type='least_squares', use_jac=True)
    print('HF energy', -8.9472891719)
    print('APseqG energy', apseqg.compute_energy())
    print('FCI value', -8.96741814557)
    assert results.success
    assert -8.9472891719 > apseqg.compute_energy() > -8.96741814557
    assert False
