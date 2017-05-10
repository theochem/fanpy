""" Tests wfns.wavefunction.apr2g.APr2G
"""
from __future__ import absolute_import, division, print_function
from nose.tools import assert_raises
from nose.plugins.attrib import attr
import numpy as np
from wfns.solver.solver import solve
from wfns.wavefunction.apr2g import APr2G
from wfns.wavefunction.apig import APIG
from wfns.tools import find_datafile


def test_template_coeffs():
    """ Tests APr2G.template_coeffs
    """
    one_int = np.arange(-16, 0, 1, dtype=float).reshape(4, 4)
    # ngem 1
    test = APr2G(2, one_int, np.ones((4, 4, 4, 4)))
    print(test.template_coeffs)
    assert np.allclose(test.template_coeffs, np.array([1, -16, -12, -8, -4, 1, 0.01, 0.01, 0.01]))
    # ngem 2
    test = APr2G(4, one_int, np.ones((4, 4, 4, 4)))
    assert np.allclose(test.template_coeffs, np.array([1, 0.9, -16, -12, -8, -4, 1, 1, 0.01, 0.01]))


def test_assign_params():
    """ Tests APr2G.assign_params
    """
    one_int = np.arange(-1, -17, -1, dtype=float).reshape(4, 4)
    test = APr2G(4, one_int, np.ones((4, 4, 4, 4)))
    # check error
    assert_raises(ValueError, test.assign_params, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0]))
    assert_raises(ValueError, test.assign_params, np.array([1, 2, 1, 9, 9, 9, 0, 0, 0, 0, 1.0]))
    assert_raises(ValueError, test.assign_params, np.array([1, 2, 9, 1, 9, 9, 0, 0, 0, 0, 1.0]))
    assert_raises(ValueError, test.assign_params, np.array([1, 2, 9, 9, 1, 9, 0, 0, 0, 0, 1.0]))
    assert_raises(ValueError, test.assign_params, np.array([1, 2, 9, 9, 9, 1, 0, 0, 0, 0, 1.0]))
    assert_raises(ValueError, test.assign_params, np.array([1, 2, 2, 9, 9, 9, 0, 0, 0, 0, 1.0]))
    assert_raises(ValueError, test.assign_params, np.array([1, 2, 9, 2, 9, 9, 0, 0, 0, 0, 1.0]))
    assert_raises(ValueError, test.assign_params, np.array([1, 2, 9, 9, 2, 9, 0, 0, 0, 0, 1.0]))
    assert_raises(ValueError, test.assign_params, np.array([1, 2, 9, 9, 9, 2, 0, 0, 0, 0, 1.0]))


def test_compute_overlap():
    """ Tests APr2G.compute_overlap manually
    """
    # two electrons
    one_int = np.arange(-1, -17, -1, dtype=float).reshape(4, 4)
    test = APr2G(2, one_int, np.ones((4, 4, 4, 4)))

    # bad SD
    assert_raises(ValueError, test.compute_overlap, 0b00110011)
    assert_raises(ValueError, test.compute_overlap, 0b00010010)

    lambdas = test.params[:test.npair]
    epsilons = test.params[test.npair:test.npair+test.nspatial]
    zetas = test.params[test.npair+test.nspatial:test.npair+test.nspatial*2]
    coeffs = zetas / (lambdas[:, np.newaxis] - epsilons)
    # overlap
    assert np.allclose(test.compute_overlap(0b00010001, deriv=None), coeffs[0, 0])
    assert np.allclose(test.compute_overlap(0b00100010, deriv=None), coeffs[0, 1])
    assert np.allclose(test.compute_overlap(0b01000100, deriv=None), coeffs[0, 2])
    assert np.allclose(test.compute_overlap(0b10001000, deriv=None), coeffs[0, 3])
    # differentiate
    assert np.allclose(test.compute_overlap(0b00010001, deriv=0),
                       -zetas[0]/(lambdas[0] - epsilons[0])**2)
    assert np.allclose(test.compute_overlap(0b00010001, deriv=1),
                       zetas[0]/(lambdas[0]-epsilons[0])**2)
    assert np.allclose(test.compute_overlap(0b00010001, deriv=2), 0)
    assert np.allclose(test.compute_overlap(0b00010001, deriv=3), 0)
    assert np.allclose(test.compute_overlap(0b00010001, deriv=4), 0)
    assert np.allclose(test.compute_overlap(0b00010001, deriv=5), 1.0/(lambdas[0]-epsilons[0]))
    assert np.allclose(test.compute_overlap(0b00010001, deriv=6), 0)
    assert np.allclose(test.compute_overlap(0b00010001, deriv=7), 0)
    assert test.compute_overlap(0b00010001, deriv=8) == 0
    assert test.compute_overlap(0b00010001, deriv=99) == 0

    # four electron
    one_int = np.arange(-1, -17, -1, dtype=float).reshape(4, 4)
    test = APr2G(4, one_int, np.ones((4, 4, 4, 4)))

    # bad SD
    assert_raises(ValueError, test.compute_overlap, 0b00010001)
    assert_raises(ValueError, test.compute_overlap, 0b00110101)

    lambdas = test.params[:test.npair]
    epsilons = test.params[test.npair:test.npair+test.nspatial]
    zetas = test.params[test.npair+test.nspatial:test.npair+test.nspatial*2]
    coeffs = zetas / (lambdas[:, np.newaxis] - epsilons)
    # overlap
    assert np.allclose(test.compute_overlap(0b00110011, deriv=None), (coeffs[0, 0]*coeffs[1, 1]
                                                                      + coeffs[0, 1]*coeffs[1, 0]))
    assert np.allclose(test.compute_overlap(0b01010101, deriv=None), (coeffs[0, 0]*coeffs[1, 2]
                                                                      + coeffs[1, 0]*coeffs[0, 2]))
    assert np.allclose(test.compute_overlap(0b10011001, deriv=None), (coeffs[0, 0]*coeffs[1, 3]
                                                                      + coeffs[1, 0]*coeffs[0, 3]))
    assert np.allclose(test.compute_overlap(0b01100110, deriv=None), (coeffs[0, 1]*coeffs[1, 2]
                                                                      + coeffs[1, 1]*coeffs[0, 2]))
    assert np.allclose(test.compute_overlap(0b10101010, deriv=None), (coeffs[0, 1]*coeffs[1, 3]
                                                                      + coeffs[1, 1]*coeffs[0, 3]))
    assert np.allclose(test.compute_overlap(0b11001100, deriv=None), (coeffs[0, 2]*coeffs[1, 3]
                                                                      + coeffs[1, 2]*coeffs[0, 3]))
    # differentiate
    assert np.allclose(test.compute_overlap(0b00110011, deriv=0),
                       (coeffs[1, 1]*(-zetas[0]/(lambdas[0] - epsilons[0])**2)
                        + coeffs[1, 0]*(-zetas[1]/(lambdas[0] - epsilons[1])**2)))
    assert np.allclose(test.compute_overlap(0b00110011, deriv=1),
                       (coeffs[0, 1]*(-zetas[0]/(lambdas[1] - epsilons[0])**2)
                        + coeffs[0, 0]*(-zetas[1]/(lambdas[1] - epsilons[1])**2)))
    assert np.allclose(test.compute_overlap(0b00110011, deriv=2),
                       (coeffs[1, 1]*(zetas[0]/(lambdas[0]-epsilons[0])**2)
                        + coeffs[0, 1]*(zetas[0]/(lambdas[1] - epsilons[0])**2)))
    assert np.allclose(test.compute_overlap(0b00110011, deriv=3),
                       (coeffs[1, 0]*(zetas[1]/(lambdas[0]-epsilons[1])**2)
                        + coeffs[0, 0]*(zetas[1]/(lambdas[1] - epsilons[1])**2)))
    assert np.allclose(test.compute_overlap(0b00110011, deriv=4), 0)
    assert np.allclose(test.compute_overlap(0b00110011, deriv=5), 0)
    assert np.allclose(test.compute_overlap(0b00110011, deriv=6),
                       (coeffs[1, 1]*(1.0/(lambdas[0]-epsilons[0]))
                        + coeffs[0, 1]*(1.0/(lambdas[1] - epsilons[0]))))
    assert np.allclose(test.compute_overlap(0b00110011, deriv=7),
                       (coeffs[1, 0]*(1.0/(lambdas[0]-epsilons[1]))
                        + coeffs[0, 0]*(1.0/(lambdas[1] - epsilons[1]))))
    assert test.compute_overlap(0b00110011, deriv=8) == 0
    assert test.compute_overlap(0b00110011, deriv=99) == 0


def test_compute_overlap_apig():
    """ Tests APr2G.compute_overlap by comparing results with equivalent APIG wavefunction

    Note
    ----
    Does not compare derivative (b/c they won't be the same)
    """
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/lih_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = (np.load(find_datafile('test/lih_hf_sto6g_oneint.npy')), )
    two_int = (np.load(find_datafile('test/lih_hf_sto6g_twoint.npy')), )
    nuc_nuc = 0.995317634356
    params = np.array([2.52189752e+00, -3.22516976e-01, 1.55439888e+00, -3.23444695e-01,
                       -2.05929322e+00, -5.09945972e-01, -5.08808399e-01, -3.52287139e-01,
                       1.00000000e+00, 8.90849147e-04, -1.97964592e-02, -5.35802964e-03,
                       -5.36197023e-03, -3.36201766e-03, -8.96352586e+00])
    test = APr2G(4, one_int, two_int, params=params, nuc_nuc=nuc_nuc)

    # equivalent APIG wavefunction
    apig_params = np.array([1.033593181822e+00, 3.130903350751e-04, -4.321247538977e-03,
                            -1.767251395337e-03, -1.769214953534e-03, -1.169729179981e-03,
                            -5.327889357199e-01, 9.602580629349e-01, -1.139839360648e-02,
                            -2.858698370621e-02, -2.878270043699e-02, -1.129324573431e-01,
                            -8.96352586e+00])
    apig = APIG(4, one_int, two_int, params=apig_params, nuc_nuc=nuc_nuc)

    for sd in apig.pspace:
        assert np.allclose(apig.compute_overlap(sd), test.compute_overlap(sd))


def test_normalize():
    """ Tests APr2G.normalize
    """
    one_int = np.arange(-1, -5, -1, dtype=float).reshape(2, 2)
    test = APr2G(2, one_int, np.ones((2, 2, 2, 2)), dtype=float)
    #check
    test.assign_ref_sds(0b0101)
    test.assign_params(np.array([0, 1, 1, 0, 0, 1.0]))
    assert_raises(ValueError, test.normalize)
    test.assign_dtype(complex)
    test.assign_params(np.array([0, 1j, 1, 1, 1, 1.0]))
    assert_raises(ValueError, test.normalize)

    # one geminal, one reference
    test = APr2G(2, one_int, np.ones((2, 2, 2, 2)), dtype=float)
    test.assign_params(np.array([0, 1, 1, 2, 1, 1.0]))
    test.assign_ref_sds(0b0101)
    test.normalize()
    assert test.compute_norm() == 1
    # one geminal, two reference
    test.assign_params(np.array([0, 1, 1, 2, 3, 1.0]))
    test.assign_ref_sds([0b0101, 0b1010])
    test.normalize()
    assert test.compute_norm() == 1

    one_int = np.arange(-1, -17, -1, dtype=float).reshape(4, 4)
    test = APr2G(4, one_int, np.ones((4, 4, 4, 4)))
    # multiple geminal, one reference
    test.assign_params(np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0], dtype=float))
    test.assign_ref_sds(0b00110011)
    test.normalize()
    assert test.compute_norm() == 1
    # multiple geminal, multiple reference
    test.assign_params(np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0], dtype=float))
    test.assign_ref_sds([0b00110011, 0b01010101])
    test.normalize()
    assert test.compute_norm() == 1


def answer_apr2g_h2_631gdp():
    """ Finds the APr2G/6-31G** wavefunction by scanning through the coefficients for the lowest
    energy

    Note
    ----
    Uses APIG answer from test_apig.answer_apig_h2_631gdp, converting it to APr2G
    """
    nelec = 2
    one_int = np.load(find_datafile('test/h2_hf_631gdp_oneint.npy'))
    two_int = np.load(find_datafile('test/h2_hf_631gdp_twoint.npy'))
    nuc_nuc = 0.71317683129
    apig = APIG(nelec, one_int, two_int, nuc_nuc=nuc_nuc,
                params=np.array([0.995079200788, -0.059166892062, -0.054284175189, -0.036920061272,
                                 -0.028848919079, -0.028847742282, -0.013108383833, -0.008485392433,
                                 -0.008485285973, -0.005149411511, -1.8696828608304896]),
                ref_sds=(0b00000000010000000001, 0b00000000100000000010, 0b00000001000000000100,
                         0b00000010000000001000, 0b00000100000000010000, 0b00001000000000100000,
                         0b00010000000001000000, 0b00100000000010000000, 0b01000000000100000000,
                         0b10000000001000000000,))
    apr2g = apig.to_apr2g(rmsd=0.0000001)
    print(apr2g.params)
    solve(apr2g, solver_type='variational', use_jac=True)
    np.set_printoptions(precision=14)
    print(apr2g.params)


@attr('slow')
def test_apr2g_h2_631gdp():
    """ Tests ground state APr2G wavefunction using H2 with HF/6-31G** orbital

    Answers obtained from answer_apr2g_h2_631

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
    nelec = 2
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/h2_hf_631gdp.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile('test/h2_hf_631gdp_oneint.npy'))
    two_int = np.load(find_datafile('test/h2_hf_631gdp_twoint.npy'))
    nuc_nuc = 0.71317683129

    # initial guess close to answer (apig answer)
    params = np.array([5.00024140e-01, -5.04920993e-01, 1.74433862e-03, 1.46912779e-03,
                       6.80650578e-04, 4.15804100e-04, 4.15770206e-04, 8.59042504e-05,
                       3.60000884e-05, 3.59991852e-05, 1.32585080e-05, 1.00000000e+00,
                       -2.94816672e-02, -2.70636476e-02, -1.84357922e-02, -1.44131605e-02,
                       -1.44125735e-02, -6.55338229e-03, -4.24259558e-03, -4.24254236e-03,
                       -2.57476179e-03, -1.86968286e+00])
    # # Solve with least squares solver with jacobian
    # # FIXME: gets stuck near the guess
    # apr2g = APr2G(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    # solve(apr2g, solver_type='least_squares', use_jac=True)
    # print(apr2g.params)
    # assert abs(apr2g.get_energy(include_nuc=False) - (-1.8696828608304896)) < 1e-7
    # Solve with least squares solver with jacobian with guess
    apr2g = APr2G(nelec, one_int, two_int, nuc_nuc=nuc_nuc, params=params)
    solve(apr2g, solver_type='least_squares', use_jac=True)
    print(apr2g.params)
    assert abs(apr2g.get_energy(include_nuc=False) - (-1.8696828608304896)) < 1e-7
    # Solve with least squares solver without jacobian
    # FIXME: energy difference is quite large
    apr2g = APr2G(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(apr2g, solver_type='least_squares', use_jac=False)
    print(apr2g.params)
    assert abs(apr2g.get_energy(include_nuc=False) - (-1.8696828608304896)) < 1e-4
    # Solve with least squares solver without jacobian with guess
    apr2g = APr2G(nelec, one_int, two_int, nuc_nuc=nuc_nuc, params=params)
    solve(apr2g, solver_type='least_squares', use_jac=False)
    print(apr2g.params)
    assert abs(apr2g.get_energy(include_nuc=False) - (-1.8696828608304896)) < 1e-4
    # Solve with cma solver
    apr2g = APr2G(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(apr2g, solver_type='cma', use_jac=False)
    print(apr2g.params)
    assert abs(apr2g.get_energy(include_nuc=False) - (-1.8696828608304896)) < 1e-3
    # Solve with cma solver with guess
    apr2g = APr2G(nelec, one_int, two_int, nuc_nuc=nuc_nuc, params=params)
    solve(apr2g, solver_type='cma', use_jac=False)
    print(apr2g.params)
    assert abs(apr2g.get_energy(include_nuc=False) - (-1.8696828608304896)) < 1e-3


def answer_apr2g_lih_sto6g():
    """ Finds the APr2G/STO-6G wavefunction by scanning through the coefficients for the lowest
    energy for LiH

    Note
    ----
    Uses APIG answer from test_apig.answer_apig_lih_sto6g, converting it to APr2G
    """
    nelec = 4
    one_int = (np.load(find_datafile('test/lih_hf_sto6g_oneint.npy')), )
    two_int = (np.load(find_datafile('test/lih_hf_sto6g_twoint.npy')), )
    nuc_nuc = 0.995317634356
    apig = APIG(nelec, one_int, two_int, nuc_nuc=nuc_nuc,
                params=np.array([1.033593181822e+00, 3.130903350751e-04, -4.321247538977e-03,
                                 -1.767251395337e-03, -1.769214953534e-03, -1.169729179981e-03,
                                 -5.327889357199e-01, 9.602580629349e-01, -1.139839360648e-02,
                                 -2.858698370621e-02, -2.878270043699e-02, -1.129324573431e-01,
                                 -8.963531109581904]),
                ref_sds=(0b000011000011, 0b000101000101, 0b001001001001, 0b010001010001,
                         0b100001100001, 0b000110000110, 0b001010001010, 0b010010010010,
                         0b100010100010, 0b001100001100, 0b010100010100, 0b100100100100,
                         0b011000011000, 0b101000101000, 0b110000110000))
    apr2g = apig.to_apr2g(rmsd=0.0000001)
    np.set_printoptions(precision=14)
    print(apr2g.params)
    solve(apr2g, solver_type='variational', use_jac=True)
    print(apr2g.params)


@attr('slow')
def test_apr2g_lih_sto6g():
    """ Tests APr2G wavefunction using LiH with HF/STO-6G orbital

    Answers obtained from answer_apr2g_lih_sto6g

    HF (Electronic) Energy : -8.9472891719
    APIG Energy : -8.96353110958190e+00
    APIG Coeffs : [2.52192912907165e+00, -3.22730595025637e-01, 1.55436195824780e+00,
                   -3.23195369339517e-01, -2.05933948914417e+00, -5.09311039678737e-01,
                   -5.08826074574783e-01, -3.52888520424345e-01, 9.99590527398775e-01,
                   4.46478511362153e-04, -1.98165460930347e-02, -5.36827124240751e-03,
                   -5.35329619359940e-03, -3.40758975682932e-03]
    """
    nelec = 4
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/lih_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = (np.load(find_datafile('test/lih_hf_sto6g_oneint.npy')), )
    two_int = (np.load(find_datafile('test/lih_hf_sto6g_twoint.npy')), )
    nuc_nuc = 0.995317634356

    # initial guess close to answer (apig answer)
    params = np.array([2.52192912907165e+00, -3.22730595025637e-01, 1.55436195824780e+00,
                       -3.23195369339517e-01, -2.05933948914417e+00, -5.09311039678737e-01,
                       -5.08826074574783e-01, -3.52888520424345e-01, 9.99590527398775e-01,
                       4.46478511362153e-04, -1.98165460930347e-02, -5.36827124240751e-03,
                       -5.35329619359940e-03, -3.40758975682932e-03, -8.96353110958190e+00])
    # # Solve with least squares solver with jacobian
    # # FIXME: for some reason, this optimization stops very quickly
    # apr2g = APr2G(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    # solve(apr2g, solver_type='least_squares', use_jac=True)
    # print(apr2g.params)
    # assert abs(apr2g.get_energy(include_nuc=False) - (-8.96353110958190)) < 1e-7
    # Solve with least squares solver with jacobian with guess
    apr2g = APr2G(nelec, one_int, two_int, nuc_nuc=nuc_nuc, params=params)
    solve(apr2g, solver_type='least_squares', use_jac=True)
    print(apr2g.params)
    assert abs(apr2g.get_energy(include_nuc=False) - (-8.96353110958190)) < 1e-6
    # Solve with least squares solver without jacobian
    apr2g = APr2G(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(apr2g, solver_type='least_squares', use_jac=False)
    print(apr2g.params)
    assert abs(apr2g.get_energy(include_nuc=False) - (-8.96353110958190)) < 1e-3
    # Solve with least squares solver without jacobian with guess
    apr2g = APr2G(nelec, one_int, two_int, nuc_nuc=nuc_nuc, params=params)
    solve(apr2g, solver_type='least_squares', use_jac=False)
    print(apr2g.params)
    assert abs(apr2g.get_energy(include_nuc=False) - (-8.96353110958190)) < 1e-6
    # Solve with cma solver
    # FIXME: CMA solver is quite bad
    apr2g = APr2G(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(apr2g, solver_type='cma', use_jac=False)
    print(apr2g.params)
    assert abs(apr2g.get_energy(include_nuc=False) - (-8.96353110958190)) < 1e-1
    # Solve with cma solver with guess
    # FIXME: CMA solver is quite bad
    apr2g = APr2G(nelec, one_int, two_int, nuc_nuc=nuc_nuc, params=params)
    solve(apr2g, solver_type='cma', use_jac=False)
    print(apr2g.params)
    assert abs(apr2g.get_energy(include_nuc=False) - (-8.96353110958190)) < 1e-2
