from __future__ import absolute_import, division, print_function
import numpy as np
from nose.plugins.attrib import attr
import types
from wfns.backend import graphs
from wfns.tools import find_datafile
from wfns.wavefunction.geminals.apsetg import APsetG
from wfns.ham.chemical import ChemicalHamiltonian
from wfns import solver


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


def answer_apsetg_h2_sto6g():
    """Find the APsetG/STO-6G wavefunction variationally for H2 system."""
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/h2_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile('test/h2_hf_sto6g_oneint.npy'))
    two_int = np.load(find_datafile('test/h2_hf_sto6g_twoint.npy'))
    nuc_nuc = 0.71317683129
    ham = ChemicalHamiltonian(one_int, two_int, orbtype='restricted', energy_nuc_nuc=nuc_nuc)

    nelec = 2
    nspin = 4
    full_sds = (0b0101, 0b1001, 0b0110, 0b1010)

    apsetg = APsetG(nelec, nspin)
    results = solver.equation_solver.optimize_wfn_variational(apsetg, ham, ref_sds=full_sds)
    print(results)
    return apsetg.params


def test_apsetg_h2_sto6g():
    """Test APsetG wavefunction using H2 with HF/STO-6G orbitals.

    Answers obtained from answer_APsetG_h2_sto6g

    HF (Electronic) Energy : -1.838434256
    APsetG Energy : -1.859089844148893
    APsetG Coeffs : [0.00000000e+00, 9.99041662e-01, 4.39588902e-16, 4.39588902e-16,
                     -1.13626930e-01, 0.00000000e+00])
    """
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/h2_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile('test/h2_hf_sto6g_oneint.npy'))
    two_int = np.load(find_datafile('test/h2_hf_sto6g_twoint.npy'))
    nuc_nuc = 0.71317683129
    ham = ChemicalHamiltonian(one_int, two_int, orbtype='restricted', energy_nuc_nuc=nuc_nuc)

    nelec = 2
    nspin = 4
    full_sds = (0b0101, 0b1001, 0b0110, 0b1010)

    # Least squares system solver.
    apsetg = APsetG(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apsetg, ham, energy_is_param=False,
                                                       ref_sds=full_sds,
                                                       solver_kwargs={'jac': None})
    assert abs(results['energy'] - (-1.8590898441488932)) < 1e-7
    # FIXME: energy is a little off
    apsetg = APsetG(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apsetg, ham, energy_is_param=False,
                                                       ref_sds=None,
                                                       solver_kwargs={'jac': None})
    assert abs(results['energy'] - (-1.8590898441488932)) < 1e-5
    # FIXME: optimization with jacobian requires a good guess
    apsetg = APsetG(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apsetg, ham, energy_is_param=False,
                                                       ref_sds=full_sds)
    assert abs(results['energy'] - (-1.8590898441488932)) < 1e-7
    apsetg = APsetG(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apsetg, ham, energy_is_param=False,
                                                       ref_sds=None)
    assert abs(results['energy'] - (-1.8590898441488932)) < 1e-7
    apsetg = APsetG(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apsetg, ham, energy_is_param=True,
                                                       ref_sds=full_sds,
                                                       solver_kwargs={'jac': None})
    assert abs(results['energy'] - (-1.8590898441488932)) < 1e-7
    apsetg = APsetG(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apsetg, ham, energy_is_param=True,
                                                       ref_sds=None,
                                                       solver_kwargs={'jac': None})
    assert abs(results['energy'] - (-1.8590898441488932)) < 1e-7
    # FIXME: optimization with jacobian requires a good guess
    apsetg = APsetG(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apsetg, ham, energy_is_param=True,
                                                       ref_sds=full_sds)
    assert abs(results['energy'] - (-1.8590898441488932)) < 1e-7
    apsetg = APsetG(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apsetg, ham, energy_is_param=True,
                                                       ref_sds=None)
    assert abs(results['energy'] - (-1.8590898441488932)) < 1e-7

    # Quasi Newton equation solver
    apsetg = APsetG(nelec, nspin)
    results = solver.equation_solver.optimize_wfn_variational(apsetg, ham,
                                                              left_pspace=full_sds,
                                                              right_pspace=None,
                                                              ref_sds=full_sds,
                                                              solver_kwargs={'jac': None},
                                                              norm_constrained=False)
    assert abs(results['energy'] - (-1.8590898441488932)) < 1e-7
    apsetg = APsetG(nelec, nspin)
    results = solver.equation_solver.optimize_wfn_variational(apsetg, ham,
                                                              left_pspace=full_sds,
                                                              right_pspace=full_sds[:10],
                                                              ref_sds=full_sds,
                                                              solver_kwargs={'jac': None},
                                                              norm_constrained=False)
    # FIXME: energy is a bit off
    assert abs(results['energy'] - (-1.8590898441488932)) < 1e-4
    # FIXME: optimization with jacobian requires a good guess
    apsetg = APsetG(nelec, nspin)
    results = solver.equation_solver.optimize_wfn_variational(apsetg, ham,
                                                              left_pspace=full_sds,
                                                              right_pspace=None,
                                                              ref_sds=full_sds,
                                                              norm_constrained=False)
    assert abs(results['energy'] - (-1.8590898441488932)) < 1e-7


def answer_apsetg_h2_631gdp():
    """Test APsetG wavefunction using H2 with HF/6-31G** orbitals.

    HF Value :       -1.84444667247
    Old Code Value : -1.86968284431
    FCI Value :      -1.87832550029
    """
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/h2_hf_631gdp.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile('test/h2_hf_631gdp_oneint.npy'))
    two_int = np.load(find_datafile('test/h2_hf_631gdp_twoint.npy'))
    nuc_nuc = 0.71317683129
    ham = ChemicalHamiltonian(one_int, two_int, orbtype='restricted', energy_nuc_nuc=nuc_nuc)

    nelec = 2
    nspin = 20
    full_sds = [1 << i | 1 << j for i in range(10) for j in range(10, 20)]

    apsetg = APsetG(nelec, nspin)
    results = solver.equation_solver.optimize_wfn_variational(apsetg, ham, ref_sds=full_sds)
    print(results)
    return apsetg.params


@attr('slow')
def test_apsetg_lih_sto6g():
    """Test APsetG with LiH using HF/STO-6G orbitals.

    HF Value :       -8.9472891719
    Old Code Value : -8.96353105152
    FCI Value :      -8.96741814557
    """
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/lih_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = (np.load(find_datafile('test/lih_hf_sto6g_oneint.npy')), )
    two_int = (np.load(find_datafile('test/lih_hf_sto6g_twoint.npy')), )
    nuc_nuc = 0.995317634356
    ham = ChemicalHamiltonian(one_int, two_int, orbtype='restricted', energy_nuc_nuc=nuc_nuc)

    nelec = 4
    nspin = 12
    full_sds = [1 << i | 1 << j | 1 << k | 1 << l for i in range(6) for j in range(i+1, 6)
                for k in range(6, 12) for l in range(k+1, 12)]

    # FIXME: no answer b/cc it took too long to calculate
    energy_answer = None

    # Least squares system solver.
    apsetg = APsetG(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apsetg, ham, energy_is_param=False,
                                                       ref_sds=full_sds,
                                                       solver_kwargs={'jac': None})
    assert abs(results['energy'] - (-energy_answer)) < 1e-7
    apsetg = APsetG(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apsetg, ham, energy_is_param=False,
                                                       ref_sds=None,
                                                       solver_kwargs={'jac': None})
    # FIXME: energy is a little off
    assert abs(results['energy'] - (-energy_answer)) < 1e-5
    apsetg = APsetG(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apsetg, ham, energy_is_param=False,
                                                       ref_sds=full_sds)
    assert abs(results['energy'] - (-energy_answer)) < 1e-7
    apsetg = APsetG(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apsetg, ham, energy_is_param=False,
                                                       ref_sds=None)
    assert abs(results['energy'] - (-energy_answer)) < 1e-7
    apsetg = APsetG(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apsetg, ham, energy_is_param=True,
                                                       ref_sds=full_sds,
                                                       solver_kwargs={'jac': None})
    assert abs(results['energy'] - (-energy_answer)) < 1e-7
    apsetg = APsetG(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apsetg, ham, energy_is_param=True,
                                                       ref_sds=None,
                                                       solver_kwargs={'jac': None})
    assert abs(results['energy'] - (-energy_answer)) < 1e-7
    apsetg = APsetG(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apsetg, ham, energy_is_param=True,
                                                       ref_sds=full_sds)
    assert abs(results['energy'] - (-energy_answer)) < 1e-7
    apsetg = APsetG(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apsetg, ham, energy_is_param=True,
                                                       ref_sds=None)
    assert abs(results['energy'] - (-energy_answer)) < 1e-7

    # Quasi Newton equation solver
    apsetg = APsetG(nelec, nspin)
    apg.assign_params(add_noise=True)
    results = solver.equation_solver.optimize_wfn_variational(apsetg, ham,
                                                              left_pspace=full_sds,
                                                              right_pspace=None,
                                                              ref_sds=full_sds,
                                                              solver_kwargs={'jac': None},
                                                              norm_constrained=False)
    assert abs(results['energy'] - (-energy_answer)) < 1e-7
    apsetg = APsetG(nelec, nspin)
    results = solver.equation_solver.optimize_wfn_variational(apsetg, ham,
                                                              left_pspace=full_sds,
                                                              right_pspace=full_sds[:5],
                                                              ref_sds=full_sds,
                                                              solver_kwargs={'jac': None},
                                                              norm_constrained=False)
    assert abs(results['energy'] - (-energy_answer)) < 1e-7
    apsetg = APsetG(nelec, nspin)
    results = solver.equation_solver.optimize_wfn_variational(apsetg, ham,
                                                              left_pspace=full_sds,
                                                              right_pspace=None,
                                                              ref_sds=full_sds,
                                                              norm_constrained=False)
    assert abs(results['energy'] - (-energy_answer)) < 1e-7
