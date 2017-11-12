"""Test wfns.wavefunction.geminals.apr2g.APr2G."""
from __future__ import absolute_import, division, print_function
from nose.plugins.attrib import attr
import numpy as np
from wfns.tools import find_datafile
from wfns.wfn.geminal.apr2g import APr2G
from wfns.ham.senzero import SeniorityZeroHamiltonian
from wfns import solver


class TestAPr2G(APr2G):
    """APr2G that skips initialization."""
    def __init__(self):
        self._cache_fns = {}


def answer_apr2g_h2_631gdp():
    """Find the APr2G/6-31G** wavefunction by scanning through the coefficients.

    Note
    ----
    Uses APIG answer from test_apr2g_apig.answer_apig_h2_631gdp, converting it to APr2G
    """
    one_int = np.load(find_datafile('test/h2_hf_631gdp_oneint.npy'))
    two_int = np.load(find_datafile('test/h2_hf_631gdp_twoint.npy'))
    nuc_nuc = 0.71317683129
    ham = SeniorityZeroHamiltonian(one_int, two_int, orbtype='restricted', energy_nuc_nuc=nuc_nuc)

    nelec = 2
    nspin = 20
    full_sds = (0b00000000010000000001, 0b00000000100000000010, 0b00000001000000000100,
                0b00000010000000001000, 0b00000100000000010000, 0b00001000000000100000,
                0b00010000000001000000, 0b00100000000010000000, 0b01000000000100000000,
                0b10000000001000000000)

    apr2g_params = APr2G.params_from_apig(np.array([[0.995079200788, -0.059166892062,
                                                     -0.054284175189, -0.036920061272,
                                                     -0.028848919079, -0.028847742282,
                                                     -0.013108383833, -0.008485392433,
                                                     -0.008485285973, -0.005149411511]]))
    apr2g = APr2G(nelec, nspin, params=apr2g_params)
    results = solver.equation_solver.optimize_wfn_variational(apr2g, ham, ref_sds=full_sds)
    print(results)
    print(apr2g.params)


@attr('slow')
def test_apr2g_apr2g_h2_631gdp():
    """Test ground state APr2G wavefunction using H2 with HF/6-31G** orbitals.

    Answers obtained from answer_apr2g_h2_631gdp

    HF (Electronic) Energy : -1.84444667247
    APr2G Energy : -1.86968286083049e+00
    APr2G Coeffs : [5.00024139947352e-01, -5.04920993272312e-01, 1.74433861913406e-03,
                    1.46912778996944e-03, 6.80650579364171e-04, 4.15804099437146e-04,
                    4.15770205441596e-04, 8.59042503314636e-05, 3.60000884233533e-05,
                    3.59991851562772e-05, 1.32585077026253e-05, 9.99999999977375e-01,
                    -2.94816671672589e-02, -2.70636476435323e-02, -1.84357922467577e-02,
                    -1.44131604476661e-02, -1.44125734820933e-02, -6.55338227991008e-03,
                    -4.24259557902134e-03, -4.24254235743836e-03, -2.57476173228552e-03,]
    """
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/h2_hf_631gdp.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile('test/h2_hf_631gdp_oneint.npy'))
    two_int = np.load(find_datafile('test/h2_hf_631gdp_twoint.npy'))
    nuc_nuc = 0.71317683129
    ham = SeniorityZeroHamiltonian(one_int, two_int, orbtype='restricted', energy_nuc_nuc=nuc_nuc)

    nelec = 2
    nspin = 20
    full_sds = (0b00000000010000000001, 0b00000000100000000010, 0b00000001000000000100,
                0b00000010000000001000, 0b00000100000000010000, 0b00001000000000100000,
                0b00010000000001000000, 0b00100000000010000000, 0b01000000000100000000,
                0b10000000001000000000)

    # initial guess close to answer (apig answer)
    params_answer = np.array([5.00024140e-01, -5.04920993e-01, 1.74433862e-03, 1.46912779e-03,
                              6.80650578e-04, 4.15804100e-04, 4.15770206e-04, 8.59042504e-05,
                              3.60000884e-05, 3.59991852e-05, 1.32585080e-05, 1.00000000e+00,
                              -2.94816672e-02, -2.70636476e-02, -1.84357922e-02, -1.44131605e-02,
                              -1.44125735e-02, -6.55338229e-03, -4.24259558e-03, -4.24254236e-03,
                              -2.57476179e-03])

    # Least squares system solver.
    apr2g = APr2G(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apr2g, ham, energy_is_param=False,
                                                       ref_sds=full_sds,
                                                       solver_kwargs={'jac': None})
    assert abs(results['energy'] - (-1.86968286083049)) < 1e-7
    apr2g = APr2G(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apr2g, ham, energy_is_param=False,
                                                       ref_sds=None,
                                                       solver_kwargs={'jac': None})
    # FIXME: energy is a little off
    assert abs(results['energy'] - (-1.86968286083049)) < 1e-5
    # FIXME: optimization with jacobian requires a good guess
    apr2g = APr2G(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apr2g, ham, energy_is_param=False,
                                                       ref_sds=full_sds)
    assert abs(results['energy'] - (-1.86968286083049)) < 1e-7
    # FIXME: this one fails for some reason
    # apr2g = APr2G(nelec, nspin)
    # apr2g.assign_params(params_answer + 1e-4 * (np.random.rand(*apr2g.params_shape) - 0.5))
    # results = solver.system_solver.optimize_wfn_system(apr2g, ham, energy_is_param=False,
    #                                                    ref_sds=None)
    # assert abs(results['energy'] - (-1.86968286083049)) < 1e-7
    apr2g = APr2G(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apr2g, ham, energy_is_param=True,
                                                       ref_sds=full_sds,
                                                       solver_kwargs={'jac': None})
    assert abs(results['energy'] - (-1.86968286083049)) < 1e-7
    apr2g = APr2G(nelec, nspin)
    results = solver.system_solver.optimize_wfn_system(apr2g, ham, energy_is_param=True,
                                                       ref_sds=None,
                                                       solver_kwargs={'jac': None})
    assert abs(results['energy'] - (-1.86968286083049)) < 1e-7
    # FIXME: optimization with jacobian requires a good guess
    apr2g = APr2G(nelec, nspin)
    apr2g.assign_params(params_answer + 1e-4 * (np.random.rand(*apr2g.params_shape) - 0.5))
    results = solver.system_solver.optimize_wfn_system(apr2g, ham, energy_is_param=True,
                                                       ref_sds=full_sds)
    assert abs(results['energy'] - (-1.86968286083049)) < 1e-7
    # FIXME: this one fails for some reason
    # apr2g = APr2G(nelec, nspin)
    # apr2g.assign_params(params_answer + 1e-4 * (np.random.rand(*apr2g.params_shape) - 0.5))
    # results = solver.system_solver.optimize_wfn_system(apr2g, ham, energy_is_param=True,
    #                                                    ref_sds=None)
    # assert abs(results['energy'] - (-1.86968286083049)) < 1e-7

    # Quasi Newton equation solver
    apr2g = APr2G(nelec, nspin)
    results = solver.equation_solver.optimize_wfn_variational(apr2g, ham,
                                                              left_pspace=full_sds,
                                                              right_pspace=None,
                                                              ref_sds=full_sds,
                                                              solver_kwargs={'jac': None},
                                                              norm_constrained=False)
    assert abs(results['energy'] - (-1.86968286083049)) < 1e-7
    apr2g = APr2G(nelec, nspin)
    results = solver.equation_solver.optimize_wfn_variational(apr2g, ham,
                                                              left_pspace=full_sds,
                                                              right_pspace=full_sds[:10],
                                                              ref_sds=full_sds,
                                                              solver_kwargs={'jac': None},
                                                              norm_constrained=False)
    # FIXME: energy is a bit off
    assert abs(results['energy'] - (-1.86968286083049)) < 1e-4
    # FIXME: optimization with jacobian requires a good guess
    apr2g = APr2G(nelec, nspin)
    apr2g.assign_params(params_answer + 1e-4 * (np.random.rand(*apr2g.params_shape) - 0.5))
    results = solver.equation_solver.optimize_wfn_variational(apr2g, ham,
                                                              left_pspace=full_sds,
                                                              right_pspace=None,
                                                              ref_sds=full_sds,
                                                              norm_constrained=False)
    assert abs(results['energy'] - (-1.86968286083049)) < 1e-7


def answer_apr2g_lih_sto6g():
    """Find the APr2G/STO-6G wavefunction for LiH by scanning through the coefficients.

    Note
    ----
    Uses APIG answer from test_apr2g_apig.answer_apig_lih_sto6g, converting it to APr2G
    """
    one_int = (np.load(find_datafile('test/lih_hf_sto6g_oneint.npy')), )
    two_int = (np.load(find_datafile('test/lih_hf_sto6g_twoint.npy')), )
    nuc_nuc = 0.995317634356
    ham = SeniorityZeroHamiltonian(one_int, two_int, orbtype='restricted', energy_nuc_nuc=nuc_nuc)

    nelec = 4
    nspin = 12
    full_sds = (0b000011000011, 0b000101000101, 0b001001001001, 0b010001010001, 0b100001100001,
                0b000110000110, 0b001010001010, 0b010010010010, 0b100010100010, 0b001100001100,
                0b010100010100, 0b100100100100, 0b011000011000, 0b101000101000, 0b110000110000)

    apr2g_params = APr2G.params_from_apig(np.array([[1.033593181822e+00, 3.130903350751e-04,
                                                     -4.321247538977e-03, -1.767251395337e-03,
                                                     -1.769214953534e-03, -1.169729179981e-03],
                                                    [-5.327889357199e-01, 9.602580629349e-01,
                                                     -1.139839360648e-02, -2.858698370621e-02,
                                                     -2.878270043699e-02, -1.129324573431e-01]]),
                                          rmsd=0.0000001)
    print(apr2g_params)
    apr2g = APr2G(nelec, nspin, params=apr2g_params)
    results = solver.equation_solver.optimize_wfn_variational(apr2g, ham,
                                                              left_pspace=full_sds,
                                                              right_pspace=None,
                                                              ref_sds=full_sds,
                                                              solver_kwargs={'jac': None},
                                                              norm_constrained=False)
    print(apr2g.params)
    print(results)


@attr('slow')
def test_apr2g_apr2g_lih_sto6g():
    """Test APr2G wavefunction using LiH with HF/STO-6G orbitals.

    Answers obtained from answer_apr2g_lih_sto6g

    HF (Electronic) Energy : -8.9472891719
    APIG Energy : -8.96353110958190e+00
    APIG Coeffs : [2.52192912907165e+00, -3.22730595025637e-01, 1.55436195824780e+00,
                   -3.23195369339517e-01, -2.05933948914417e+00, -5.09311039678737e-01,
                   -5.08826074574783e-01, -3.52888520424345e-01, 9.99590527398775e-01,
                   4.46478511362153e-04, -1.98165460930347e-02, -5.36827124240751e-03,
                   -5.35329619359940e-03, -3.40758975682932e-03]
    """
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/lih_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = (np.load(find_datafile('test/lih_hf_sto6g_oneint.npy')), )
    two_int = (np.load(find_datafile('test/lih_hf_sto6g_twoint.npy')), )
    nuc_nuc = 0.995317634356
    ham = SeniorityZeroHamiltonian(one_int, two_int, orbtype='restricted', energy_nuc_nuc=nuc_nuc)

    nelec = 4
    nspin = 12
    full_sds = (0b000011000011, 0b000101000101, 0b001001001001, 0b010001010001, 0b100001100001,
                0b000110000110, 0b001010001010, 0b010010010010, 0b100010100010, 0b001100001100,
                0b010100010100, 0b100100100100, 0b011000011000, 0b101000101000, 0b110000110000)

    # initial guess close to answer (apig answer)
    params_answer = np.array([2.52192912907165e+00, -3.22730595025637e-01, 1.55436195824780e+00,
                              -3.23195369339517e-01, -2.05933948914417e+00, -5.09311039678737e-01,
                              -5.08826074574783e-01, -3.52888520424345e-01, 9.99590527398775e-01,
                              4.46478511362153e-04, -1.98165460930347e-02, -5.36827124240751e-03,
                              -5.35329619359940e-03, -3.40758975682932e-03])

    # Least squares system solver.
    apr2g = APr2G(nelec, nspin)
    apr2g.assign_params(params_answer + 1e-4 * (np.random.rand(*apr2g.params_shape) - 0.5))
    results = solver.system_solver.optimize_wfn_system(apr2g, ham, energy_is_param=False,
                                                       ref_sds=full_sds,
                                                       solver_kwargs={'jac': None})
    assert abs(results['energy'] - (-8.96353110958190)) < 1e-7
    # FIXME: energy is a little off
    apr2g = APr2G(nelec, nspin)
    apr2g.assign_params(params_answer + 1e-4 * (np.random.rand(*apr2g.params_shape) - 0.5))
    results = solver.system_solver.optimize_wfn_system(apr2g, ham, energy_is_param=False,
                                                       ref_sds=None,
                                                       solver_kwargs={'jac': None})
    assert abs(results['energy'] - (-8.96353110958190)) < 1e-5
    # FIXME: optimization with jacobian requires a good guess
    apr2g = APr2G(nelec, nspin)
    apr2g.assign_params(params_answer + 1e-4 * (np.random.rand(*apr2g.params_shape) - 0.5))
    results = solver.system_solver.optimize_wfn_system(apr2g, ham, energy_is_param=False,
                                                       ref_sds=full_sds)
    assert abs(results['energy'] - (-8.96353110958190)) < 1e-7
    # FIXME: this one fails for some reason
    # apr2g = APr2G(nelec, nspin)
    # apr2g.assign_params(params_answer + 1e-4 * (np.random.rand(*apr2g.params_shape) - 0.5))
    # results = solver.system_solver.optimize_wfn_system(apr2g, ham, energy_is_param=False,
    #                                                    ref_sds=None)
    # assert abs(results['energy'] - (-8.96353110958190)) < 1e-7
    apr2g = APr2G(nelec, nspin)
    apr2g.assign_params(params_answer + 1e-4 * (np.random.rand(*apr2g.params_shape) - 0.5))
    results = solver.system_solver.optimize_wfn_system(apr2g, ham, energy_is_param=True,
                                                       ref_sds=full_sds,
                                                       solver_kwargs={'jac': None})
    assert abs(results['energy'] - (-8.96353110958190)) < 1e-7
    apr2g = APr2G(nelec, nspin)
    apr2g.assign_params(params_answer + 1e-4 * (np.random.rand(*apr2g.params_shape) - 0.5))
    results = solver.system_solver.optimize_wfn_system(apr2g, ham, energy_is_param=True,
                                                       ref_sds=None,
                                                       solver_kwargs={'jac': None})
    assert abs(results['energy'] - (-8.96353110958190)) < 1e-7
    # FIXME: optimization with jacobian requires a good guess
    apr2g = APr2G(nelec, nspin)
    apr2g.assign_params(params_answer + 1e-4 * (np.random.rand(*apr2g.params_shape) - 0.5))
    results = solver.system_solver.optimize_wfn_system(apr2g, ham, energy_is_param=True,
                                                       ref_sds=full_sds)
    assert abs(results['energy'] - (-8.96353110958190)) < 1e-7
    # FIXME: this one fails for some reason
    # apr2g = APr2G(nelec, nspin)
    # apr2g.assign_params(params_answer + 1e-4 * (np.random.rand(*apr2g.params_shape) - 0.5))
    # results = solver.system_solver.optimize_wfn_system(apr2g, ham, energy_is_param=True,
    #                                                    ref_sds=None)
    # assert abs(results['energy'] - (-8.96353110958190)) < 1e-7

    # Quasi Newton equation solver
    apr2g = APr2G(nelec, nspin)
    apr2g.assign_params(params_answer + 1e-4 * (np.random.rand(*apr2g.params_shape) - 0.5))
    results = solver.equation_solver.optimize_wfn_variational(apr2g, ham,
                                                              left_pspace=full_sds,
                                                              right_pspace=None,
                                                              ref_sds=full_sds,
                                                              solver_kwargs={'jac': None},
                                                              norm_constrained=False)
    assert abs(results['energy'] - (-8.96353110958190)) < 1e-7
    apr2g = APr2G(nelec, nspin)
    apr2g.assign_params(params_answer + 1e-4 * (np.random.rand(*apr2g.params_shape) - 0.5))
    results = solver.equation_solver.optimize_wfn_variational(apr2g, ham,
                                                              left_pspace=full_sds,
                                                              right_pspace=full_sds[:10],
                                                              ref_sds=full_sds,
                                                              solver_kwargs={'jac': None},
                                                              norm_constrained=False)
    # FIXME: energy is a bit off
    assert abs(results['energy'] - (-8.96353110958190)) < 1e-4
    # FIXME: optimization with jacobian requires a good guess
    apr2g = APr2G(nelec, nspin)
    apr2g.assign_params(params_answer + 1e-4 * (np.random.rand(*apr2g.params_shape) - 0.5))
    results = solver.equation_solver.optimize_wfn_variational(apr2g, ham,
                                                              left_pspace=full_sds,
                                                              right_pspace=None,
                                                              ref_sds=full_sds,
                                                              norm_constrained=False)
    assert abs(results['energy'] - (-8.96353110958190)) < 1e-7
