from __future__ import absolute_import, division, print_function
from ..cisd import CISD
from ..hort import hartreefock


def test_cisd_wavefunction():
    #### H2 ####
    nelec = 2
    hf_dict = hartreefock(fn="./h2.xyz", basis="6-31g**", nelec=nelec)
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]
    cisd = CISD(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    # compare HF numbers
    assert abs(cisd.compute_ci_matrix()[0, 0] + cisd.nuc_nuc - (-1.131269841877) < 1e-8)
    # solve
    cisd()
    # compare with number from Gaussian
    assert abs(cisd.compute_energy() - (-1.1651486697)) < 1e-7
    #### LiH ####
    nelec = 4
    hf_dict = hartreefock(fn="./lih.xyz", basis="6-31G", nelec=nelec)
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]
    cisd = CISD(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc)
    cisd()
    print(cisd.compute_energy(), -7.9980021297, -7.99800204, E_hf)
    assert abs(cisd.compute_energy() - (-7.9980021297)) < 1e-7
