from __future__ import absolute_import, division, print_function
import os
import numpy as np
from geminals.ci.fci import FCI
from geminals.wrapper.horton import gaussian_fchk


def test_fci_h2():
    #### H2 ####
    # HF energy: -1.13126983927
    # FCI energy: -1.1651487496
    data_path = os.path.join(os.path.dirname(__file__), '../../../data/test/h2_hf_631gdp.fchk')
    hf_dict = gaussian_fchk(data_path)

    nelec = 2
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]
    fci = FCI(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc, spin=0)
    ci_matrix = fci.compute_ci_matrix()
    # compare HF numbers
    assert abs(ci_matrix[0, 0] + fci.nuc_nuc - (-1.131269841877) < 1e-8)
    # check that hamiltonian is symmetric
    assert np.allclose(ci_matrix, ci_matrix.T)
    # solve
    fci()
    # compare with number from Gaussian
    assert abs(fci.compute_energy() - (-1.1651486697)) < 1e-7

def test_fci_lih():
    #### LiH ####
    # HF energy: -7.95197153880
    # FCI energy: -7.9723355823
    data_path = os.path.join(os.path.dirname(__file__), '../../../data/test/lih_hf_sto6g.fchk')
    hf_dict = gaussian_fchk(data_path)

    nelec = 4
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]
    # manually put in slater determinants (from gaussian)
    # sds = [
    #     0b000011000011, 0b000101000011, 0b000101000101, 0b000110000011, 0b001001000011, 0b000110000101,
    #     0b001001000101, 0b001010000011, 0b010001000011, 0b000110000110, 0b001010000101, 0b001001001001,
    #     0b010001000101, 0b001100000011, 0b010010000011, 0b100001000011, 0b001010000110, 0b001010001001,
    #     0b001100000101, 0b010010000101, 0b010001001001, 0b100001000101, 0b010100000011, 0b100010000011,
    #     0b001010001010, 0b001100000110, 0b010010000110, 0b001100001001, 0b010010001001, 0b010100000101,
    #     0b100010000101, 0b010001010001, 0b100001001001, 0b011000000011, 0b100100000011, 0b001100001010,
    #     0b010010001010, 0b010100000110, 0b100010000110, 0b010100001001, 0b010010010001, 0b100010001001,
    #     0b011000000101, 0b100100000101, 0b100001010001, 0b101000000011, 0b001100001100, 0b010100001010,
    #     0b010010010010, 0b100010001010, 0b011000000110, 0b100100000110, 0b010100010001, 0b011000001001,
    #     0b100100001001, 0b100010010001, 0b101000000101, 0b100001100001, 0b110000000011, 0b010100001100,
    #     0b010100010010, 0b011000001010, 0b100100001010, 0b100010010010, 0b101000000110, 0b011000010001,
    #     0b100100010001, 0b101000001001, 0b100010100001, 0b110000000101, 0b010100010100, 0b011000001100,
    #     0b100100001100, 0b011000010010, 0b100100010010, 0b101000001010, 0b100010100010, 0b110000000110,
    #     0b101000010001, 0b100100100001, 0b110000001001, 0b011000010100, 0b100100010100, 0b101000001100,
    #     0b101000010010, 0b100100100010, 0b110000001010, 0b101000100001, 0b110000010001, 0b011000011000,
    #     0b101000010100, 0b100100100100, 0b110000001100, 0b101000100010, 0b110000010010, 0b110000100001,
    #     0b101000011000, 0b101000100100, 0b110000010100, 0b110000100010, 0b101000101000, 0b110000011000,
    #     0b110000100100, 0b110000101000, 0b110000110000
    # ]
    # fci = FCI(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc, spin=0, civec=sds)
    fci = FCI(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc, spin=0)
    ci_matrix = fci.compute_ci_matrix()
    # compare HF numbers
    assert abs(ci_matrix[0, 0] + fci.nuc_nuc - (-1.131269841877) < 1e-8)
    # check that hamiltonian is symmetric
    assert np.allclose(ci_matrix, ci_matrix.T)
    # solve
    fci()
    # check norm
    # check dimensions
    print(fci.nci)
    print(len(fci.civec))
    # compare with number from Gaussian
    print(fci.compute_energy(include_nuc=True), fci.compute_energy(include_nuc=False), -7.9723355823)
    print(fci.sd_coeffs)
    assert abs(fci.compute_energy()-(-7.9723355823)) < 1e-7
