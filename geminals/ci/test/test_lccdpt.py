from __future__ import absolute_import, division, print_function
from geminals.ci.cisd import CISD
from geminals.hort import hartreefock
from geminals.ci.lccdpt import solve_lccdpt
from geminals.sd_list import apsetg_doubles_sd_list
from geminals.proj.ap1rog import AP1roG

def test_lccdpt_wavefunction():
    #### Li2 ####
    nelec = 6
    hf_dict = hartreefock(fn="test/li2.xyz", basis="3-21g", nelec=nelec)
    E_hf = hf_dict["energy"]
    H = hf_dict["H"]
    G = hf_dict["G"]
    nuc_nuc = hf_dict["nuc_nuc"]
    geminal = AP1roG(nelec, H, G)
    energy, coeffs, sds = solve_lccdpt(geminal, (H,), (G,), 'apig')
    energy, coeffs, sds = solve_lccdpt(geminal, (H,), (G,), 'apsetg')
    energy, coeffs, sds = solve_lccdpt(geminal, (H,), (G,), 'apg')
