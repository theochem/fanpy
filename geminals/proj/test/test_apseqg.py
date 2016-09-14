from __future__ import absolute_import, division, print_function
import os
import numpy as np
from geminals.proj.apseqg import APseqG
from geminals.wrapper.horton import gaussian_fchk
import geminals.slater as slater

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

def test_apseqg_wavefunction_h2():
    #### H2 ####
    # HF Value :       -1.84444667247
    # Old Code Value : -1.86968284431
    # FCI Value :      -1.87832550029
    data_path = os.path.join(os.path.dirname(__file__), '../../../data/test/h2_hf_631gdp.fchk')
    hf_dict = gaussian_fchk(data_path)

    nelec = 2
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]
    # Reproduce HF energy
    apseqg = APseqG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc, seq_list=[0])
    apseqg.params[:-1] = apseqg.template_coeffs.flatten()
    apseqg.cache = {}
    apseqg.d_cache = {}
    assert abs(apseqg.compute_energy(include_nuc=False, ref_sds=apseqg.default_ref_sds) - (-1.84444667247)) < 1e-7
    # Solve with Jacobian using energy as a parameter
    apseqg = APseqG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    apseqg()
    energy = apseqg.compute_energy()
    print('HF energy', -1.84444667247)
    print('APseqG energy', energy)
    print('FCI value', -1.87832550029)
    assert -1.84444667247 > energy > -1.87832550029
    assert False

def test_apseqg_wavefunction_lih():
    #### LiH ####
    # HF Value :       -8.9472891719
    # Old Code Value : -8.96353105152
    # FCI Value :      -8.96741814557
    data_path = os.path.join(os.path.dirname(__file__), '../../../data/test/lih_hf_sto6g.fchk')
    hf_dict = gaussian_fchk(data_path)

    nelec = 4
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]
    # Reproduce HF energy
    apseqg = APseqG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc, seq_list=[0])
    apseqg.params[:-1] = apseqg.template_coeffs.flatten()
    apseqg.cache = {}
    apseqg.d_cache = {}
    assert abs(apseqg.compute_energy(include_nuc=False, ref_sds=apseqg.default_ref_sds) - (-8.9472891719)) < 1e-7
    # Solve with Jacobian using energy as a parameter
    apseqg = APseqG(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    # print(apseqg.params[:-1].reshape(apseqg.template_coeffs.shape))
    apseqg()
    energy = apseqg.compute_energy()
    print('HF energy', -8.9472891719)
    print('APseqG energy', energy)
    print('FCI value', -8.96741814557)
    # print(apseqg.params[:-1].reshape(apseqg.template_coeffs.shape))
    # print([apseqg.dict_gem_orbpair[i] for i,j in enumerate(apseqg.dict_orbpair_gem)])
    # print(apseqg.compute_overlap(apseqg.pspace[0]))
    # print(apseqg.find_gem_indices(apseqg.pspace[0]))
    assert -8.9472891719 > energy > -8.96741814557
    assert False
