"""Test wfns.wavefunction.geminals.apg."""
from __future__ import absolute_import, division, print_function
from nose.plugins.attrib import attr
import types
import numpy as np
from wfns.backend.graphs import generate_complete_pmatch
from wfns.tools import find_datafile
from wfns.wavefunction.geminals.apg import APG
from wfns.wavefunction.geminals.apig import APIG
from wfns.hamiltonian.chemical_hamiltonian import ChemicalHamiltonian
from wfns import solver


class TestAPG(APG):
    """APG that skips initialization."""
    def __init__(self):
        pass


def test_assign_pmatch_generator():
    """Test APG.generate_possible_orbpairs"""
    test = TestAPG()
    sd = (0, 1, 2, 3, 4, 5)
    assert isinstance(test.generate_possible_orbpairs(sd), types.GeneratorType)
    for i, j in zip(test.generate_possible_orbpairs(sd), generate_complete_pmatch(sd)):
        assert i == j


def answer_apg_h2_sto6g():
    """Find the APG/STO-6G wavefunction variationally for H2 system."""
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
    full_sds = (0b0011, 0b0101, 0b1001, 0b0110, 0b1010, 0b1100)

    apg = APG(nelec, nspin)
    results = solver.equation_solver.optimize_wfn_variational(apg, ham, ref_sds=full_sds)
    print(results)
    return apg.params


def test_apg_h2_sto6g():
    """Test APG wavefunction using H2 with HF/STO-6G orbitals.

    Answers obtained from answer_apg_h2_sto6g

    HF (Electronic) Energy : -1.838434256
    APG Energy : -1.8590898441488932
    APG Coeffs : [0.00000000e+00, 1.00061615e+00, 4.43043808e-16, 4.43043808e-16, -1.13806005e-01,
                  0.00000000e+00]
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
    full_sds = (0b0011, 0b0101, 0b1001, 0b0110, 0b1010, 0b1100)

    # Least squares system solver.
    apg = APG(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apg, ham, energy_is_param=False,
                                                       ref_sds=full_sds,
                                                       solver_kwargs={'jac': None})
    assert abs(results['energy'] - (-1.8590898441488932)) < 1e-7
    # FIXME: energy is a little off
    apg = APG(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apg, ham, energy_is_param=False,
                                                       ref_sds=None,
                                                       solver_kwargs={'jac': None})
    assert abs(results['energy'] - (-1.8590898441488932)) < 1e-5
    # FIXME: optimization with jacobian requires a good guess
    apg = APG(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apg, ham, energy_is_param=False,
                                                       ref_sds=full_sds)
    assert abs(results['energy'] - (-1.8590898441488932)) < 1e-7
    apg = APG(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apg, ham, energy_is_param=False,
                                                       ref_sds=None)
    assert abs(results['energy'] - (-1.8590898441488932)) < 1e-7
    apg = APG(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apg, ham, energy_is_param=True,
                                                       ref_sds=full_sds,
                                                       solver_kwargs={'jac': None})
    assert abs(results['energy'] - (-1.8590898441488932)) < 1e-7
    apg = APG(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apg, ham, energy_is_param=True,
                                                       ref_sds=None,
                                                       solver_kwargs={'jac': None})
    assert abs(results['energy'] - (-1.8590898441488932)) < 1e-7
    # FIXME: optimization with jacobian requires a good guess
    apg = APG(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apg, ham, energy_is_param=True,
                                                       ref_sds=full_sds)
    assert abs(results['energy'] - (-1.8590898441488932)) < 1e-7
    apg = APG(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apg, ham, energy_is_param=True,
                                                       ref_sds=None)
    assert abs(results['energy'] - (-1.8590898441488932)) < 1e-7

    # Quasi Newton equation solver
    apg = APG(nelec, nspin)
    results = solver.equation_solver.optimize_wfn_variational(apg, ham,
                                                              left_pspace=full_sds,
                                                              right_pspace=None,
                                                              ref_sds=full_sds,
                                                              solver_kwargs={'jac': None},
                                                              norm_constrained=False)
    assert abs(results['energy'] - (-1.8590898441488932)) < 1e-7
    apg = APG(nelec, nspin)
    results = solver.equation_solver.optimize_wfn_variational(apg, ham,
                                                              left_pspace=full_sds,
                                                              right_pspace=full_sds[:10],
                                                              ref_sds=full_sds,
                                                              solver_kwargs={'jac': None},
                                                              norm_constrained=False)
    # FIXME: energy is a bit off
    assert abs(results['energy'] - (-1.8590898441488932)) < 1e-4
    # FIXME: optimization with jacobian requires a good guess
    apg = APG(nelec, nspin)
    results = solver.equation_solver.optimize_wfn_variational(apg, ham,
                                                              left_pspace=full_sds,
                                                              right_pspace=None,
                                                              ref_sds=full_sds,
                                                              norm_constrained=False)
    assert abs(results['energy'] - (-1.8590898441488932)) < 1e-7


def answer_apg_h2_631gdp():
    """Find the APG/6-31G** wavefunction variationally for H2 system."""
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
    full_sds = [1 << i | 1 << j for i in range(20) for j in range(i+1, 20)]

    apg = APG(nelec, nspin)
    results = solver.equation_solver.optimize_wfn_variational(apg, ham, ref_sds=full_sds)
    print(results)
    print(apg.params)
    return apg.params


@attr('slow')
def test_apg_h2_631gdp():
    """ Tests APG wavefunction using H2 with HF/6-31G** orbital

    Answers obtained from answer_apg_h2_631gdp

    HF (Electronic) Energy : -1.84444667247
    APG Energy : -1.8783255857444985
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
    full_sds = [1 << i | 1 << j for i in range(20) for j in range(i+1, 20)]
    apig_params = np.array([[0.995079200788, -0.059166892062, -0.054284175189, -0.036920061272,
                             -0.028848919079, -0.028847742282, -0.013108383833, -0.008485392433,
                             -0.008485285973, -0.005149411511]])

    apig = APIG(nelec, nspin, params=apig_params)

    # Least squares system solver.
    apg = APG(nelec, nspin, params=apig)
    results = solver.system_solver.optimize_wfn_system(apg, ham, energy_is_param=False,
                                                       ref_sds=full_sds,
                                                       solver_kwargs={'jac': None})
    assert abs(results['energy'] - (-1.8783255857444985)) < 1e-7
    apg = APG(nelec, nspin, params=apig)
    results = solver.system_solver.optimize_wfn_system(apg, ham, energy_is_param=False,
                                                       ref_sds=None,
                                                       solver_kwargs={'jac': None})
    # FIXME: energy is a little off
    assert abs(results['energy'] - (-1.8783255857444985)) < 1e-5
    apg = APG(nelec, nspin, params=apig)
    results = solver.system_solver.optimize_wfn_system(apg, ham, energy_is_param=False,
                                                       ref_sds=full_sds)
    assert abs(results['energy'] - (-1.8783255857444985)) < 1e-7
    apg = APG(nelec, nspin, params=apig)
    results = solver.system_solver.optimize_wfn_system(apg, ham, energy_is_param=False,
                                                       ref_sds=None)
    assert abs(results['energy'] - (-1.8783255857444985)) < 1e-7
    apg = APG(nelec, nspin, params=apig)
    results = solver.system_solver.optimize_wfn_system(apg, ham, energy_is_param=True,
                                                       ref_sds=full_sds,
                                                       solver_kwargs={'jac': None})
    assert abs(results['energy'] - (-1.8783255857444985)) < 1e-7
    apg = APG(nelec, nspin, params=apig)
    results = solver.system_solver.optimize_wfn_system(apg, ham, energy_is_param=True,
                                                       ref_sds=None,
                                                       solver_kwargs={'jac': None})
    assert abs(results['energy'] - (-1.8783255857444985)) < 1e-7
    apg = APG(nelec, nspin, params=apig)
    results = solver.system_solver.optimize_wfn_system(apg, ham, energy_is_param=True,
                                                       ref_sds=full_sds)
    assert abs(results['energy'] - (-1.8783255857444985)) < 1e-7
    apg = APG(nelec, nspin, params=apig)
    results = solver.system_solver.optimize_wfn_system(apg, ham, energy_is_param=True,
                                                       ref_sds=None)
    assert abs(results['energy'] - (-1.8783255857444985)) < 1e-7

    # Quasi Newton equation solver
    apg = APG(nelec, nspin, params=apig)
    apg.assign_params(add_noise=True)
    results = solver.equation_solver.optimize_wfn_variational(apg, ham,
                                                              left_pspace=full_sds,
                                                              right_pspace=None,
                                                              ref_sds=full_sds,
                                                              solver_kwargs={'jac': None},
                                                              norm_constrained=False)
    assert abs(results['energy'] - (-1.8783255857444985)) < 1e-7
    apg = APG(nelec, nspin, params=apig)
    results = solver.equation_solver.optimize_wfn_variational(apg, ham,
                                                              left_pspace=full_sds,
                                                              right_pspace=full_sds[:5],
                                                              ref_sds=full_sds,
                                                              solver_kwargs={'jac': None},
                                                              norm_constrained=False)
    assert abs(results['energy'] - (-1.8783255857444985)) < 1e-7
    apg = APG(nelec, nspin, params=apig)
    results = solver.equation_solver.optimize_wfn_variational(apg, ham,
                                                              left_pspace=full_sds,
                                                              right_pspace=None,
                                                              ref_sds=full_sds,
                                                              norm_constrained=False)
    assert abs(results['energy'] - (-1.8783255857444985)) < 1e-7


def answer_apg_lih_sto6g():
    """Find the APG/STO-6G wavefunction variationally for LiH system."""
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
    full_sds = [1 << i | 1 << j | 1 << k | 1 << l for i in range(12) for j in range(i+1, 12)
                for k in range(j+1, 12) for l in range(k+1, 12)]

    apg = APG(nelec, nspin)
    results = solver.equation_solver.optimize_wfn_variational(apg, ham, ref_sds=full_sds)
    print(results)
    print(apg.params)
    return apg.params


@attr('slow')
def test_apg_lih_sto6g():
    """Test APG wavefunction using H2 with LiH/STO-6G orbital.

    Answers obtained from answer_apg_h2_631gdp

    HF (Electronic) Energy : -8.9472891719
    APG Energy :
    APG Coeffs :
    """
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/lih_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile('test/lih_hf_sto6g_oneint.npy'))
    two_int = np.load(find_datafile('test/lih_hf_sto6g_twoint.npy'))
    nuc_nuc = 0.995317634356
    ham = ChemicalHamiltonian(one_int, two_int, orbtype='restricted', energy_nuc_nuc=nuc_nuc)

    nelec = 4
    nspin = 12
    full_sds = [1 << i | 1 << j | 1 << k | 1 << l for i in range(12) for j in range(i+1, 12)
                for k in range(j+1, 12) for l in range(k+1, 12)]

    # FIXME: no answer b/cc it took too long to calculate
    energy_answer = None

    # Least squares system solver.
    apg = APG(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apg, ham, energy_is_param=False,
                                                       ref_sds=full_sds,
                                                       solver_kwargs={'jac': None})
    assert abs(results['energy'] - (-energy_answer)) < 1e-7
    apg = APG(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apg, ham, energy_is_param=False,
                                                       ref_sds=None,
                                                       solver_kwargs={'jac': None})
    # FIXME: energy is a little off
    assert abs(results['energy'] - (-energy_answer)) < 1e-5
    apg = APG(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apg, ham, energy_is_param=False,
                                                       ref_sds=full_sds)
    assert abs(results['energy'] - (-energy_answer)) < 1e-7
    apg = APG(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apg, ham, energy_is_param=False,
                                                       ref_sds=None)
    assert abs(results['energy'] - (-energy_answer)) < 1e-7
    apg = APG(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apg, ham, energy_is_param=True,
                                                       ref_sds=full_sds,
                                                       solver_kwargs={'jac': None})
    assert abs(results['energy'] - (-energy_answer)) < 1e-7
    apg = APG(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apg, ham, energy_is_param=True,
                                                       ref_sds=None,
                                                       solver_kwargs={'jac': None})
    assert abs(results['energy'] - (-energy_answer)) < 1e-7
    apg = APG(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apg, ham, energy_is_param=True,
                                                       ref_sds=full_sds)
    assert abs(results['energy'] - (-energy_answer)) < 1e-7
    apg = APG(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apg, ham, energy_is_param=True,
                                                       ref_sds=None)
    assert abs(results['energy'] - (-energy_answer)) < 1e-7

    # Quasi Newton equation solver
    apg = APG(nelec, nspin)
    apg.assign_params(add_noise=True)
    results = solver.equation_solver.optimize_wfn_variational(apg, ham,
                                                              left_pspace=full_sds,
                                                              right_pspace=None,
                                                              ref_sds=full_sds,
                                                              solver_kwargs={'jac': None},
                                                              norm_constrained=False)
    assert abs(results['energy'] - (-energy_answer)) < 1e-7
    apg = APG(nelec, nspin)
    results = solver.equation_solver.optimize_wfn_variational(apg, ham,
                                                              left_pspace=full_sds,
                                                              right_pspace=full_sds[:5],
                                                              ref_sds=full_sds,
                                                              solver_kwargs={'jac': None},
                                                              norm_constrained=False)
    assert abs(results['energy'] - (-energy_answer)) < 1e-7
    apg = APG(nelec, nspin)
    results = solver.equation_solver.optimize_wfn_variational(apg, ham,
                                                              left_pspace=full_sds,
                                                              right_pspace=None,
                                                              ref_sds=full_sds,
                                                              norm_constrained=False)
    assert abs(results['energy'] - (-energy_answer)) < 1e-7
