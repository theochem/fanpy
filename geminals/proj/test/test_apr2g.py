from __future__ import absolute_import, division, print_function
from geminals.proj.apr2g import APr2G
from geminals.hort import hartreefock
import numpy as np


def test_apr2g_wavefunction_li2():
	#### Li2 ####
	# FCI Value :
	nelec = 6
	hf_dict = hartreefock(fn="test/li2.xyz", basis="3-21g", nelec=nelec)
	E_hf = hf_dict["energy"]
	H = hf_dict["H"]
	G = hf_dict["G"]
	nuc_nuc = hf_dict["nuc_nuc"]
	apr2g = APr2G(nelec=nelec, H=H, G=G, nuc_nuc=nuc_nuc, energy_is_param=False)
	apr2g()
	energy = apr2g.compute_energy(include_nuc=False)
	print("apr2g energy", energy)
	assert abs(energy - (-14.796070)) < 1e-7

