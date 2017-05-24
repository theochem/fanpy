""" Tests wfns.wavefunction.apg
"""
from __future__ import absolute_import, division, print_function
from nose.plugins.attrib import attr
import types
import numpy as np
from wfns.backend.graphs import generate_complete_pmatch
from wfns.wavefunction.geminals.apg import APG
from wfns.tools import find_datafile


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


# FIXME: refactor when solver is finished
def answer_apg_h2_sto6g():
    """ Finds the APG/STO-6G wavefunction variationally for H2 system
    """
    nelec = 2
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/h2_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile('test/h2_hf_sto6g_oneint.npy'))
    two_int = np.load(find_datafile('test/h2_hf_sto6g_twoint.npy'))
    nuc_nuc = 0.71317683129
    apg = APG(nelec, one_int, two_int, nuc_nuc=nuc_nuc, ref_sds=(0b0011, 0b0101, 0b1001, 0b0110,
                                                                 0b1010, 0b1100),
              params=np.array([0, 1, 0, 0, 0, 0, 0], dtype=float))
    res = solve(apg, solver_type='variational', use_jac=True)
    print(res)
    return apg.params


def test_apg_h2_sto6g():
    """ Tests APG wavefunction using H2 with HF/STO-6G orbital

    Answers obtained from answer_apg_h2_sto6g

    HF (Electronic) Energy : -1.838434256
    APG Energy : -1.8590898441488932
    APG Coeffs : [0.00000000e+00, 1.00061615e+00, 4.43043808e-16, 4.43043808e-16, -1.13806005e-01,
                  0.00000000e+00]
    """
    nelec = 2
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/h2_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile('test/h2_hf_sto6g_oneint.npy'))
    two_int = np.load(find_datafile('test/h2_hf_sto6g_twoint.npy'))
    nuc_nuc = 0.71317683129

    # see if we can reproduce HF numbers
    apg = APG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    apg.cache = {}
    apg.d_cache = {}
    assert abs(apg.compute_energy(include_nuc=False) - (-1.838434256)) < 1e-7

    # Solve with least squares solver with jacobian
    apg = APG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(apg, solver_type='least_squares', use_jac=True)
    assert abs(apg.compute_energy(include_nuc=False) - (-1.8590898441488932)) < 1e-7
    assert abs(apg.get_energy(include_nuc=False) - (-1.8590898441488932)) < 1e-7
    # Solve with least squares solver without jacobian
    apg = APG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(apg, solver_type='least_squares', use_jac=False)
    assert abs(apg.compute_energy(include_nuc=False) - (-1.8590898441488932)) < 1e-7
    # FIXME: get_energy is different from compute_energy here
    # assert abs(apg.get_energy(include_nuc=False) - (-1.8590898441488932)) < 1e-7
    # Solve with root (newton) solver with jacobian
    apg = APG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(apg, solver_type='root', use_jac=True)
    assert abs(apg.compute_energy(include_nuc=False) - (-1.8590898441488932)) < 1e-7
    assert abs(apg.get_energy(include_nuc=False) - (-1.8590898441488932)) < 1e-7
    # Solve with root (newton) solver without jacobian
    apg = APG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(apg, solver_type='root', use_jac=False)
    assert abs(apg.compute_energy(include_nuc=False) - (-1.8590898441488932)) < 1e-7
    assert abs(apg.get_energy(include_nuc=False) - (-1.8590898441488932)) < 1e-7
    # Solve with cma solver
    apg = APG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(apg, solver_type='cma', use_jac=False)
    assert abs(apg.compute_energy(include_nuc=False) - (-1.8590898441488932)) < 1e-7
    assert abs(apg.get_energy(include_nuc=False) - (-1.8590898441488932)) < 1e-7


def answer_apg_h2_631gdp():
    """ Finds the APG/6-31G** wavefunction variationally for H2 system
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
    ref_sds = []
    for i in range(20):
        for j in range(i+1, 20):
            ref_sds.append(1 << i | 1 << j)
    params = np.zeros(191)
    params[9] = 1
    apg = APG(nelec, one_int, two_int, nuc_nuc=nuc_nuc, ref_sds=ref_sds, params=params)
    res = solve(apg, solver_type='variational', use_jac=True)
    print(res)
    return apg.params


@attr('slow')
def test_apg_h2_631gdp():
    """ Tests APG wavefunction using H2 with HF/6-31G** orbital

    Answers obtained from answer_apg_h2_631gdp

    HF (Electronic) Energy : -1.84444667247
    APG Energy : -1.8783255857444985
    APG Coeffs : [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                   0.00000000e+00, 4.76963140e-01, -5.11185783e-13, 3.95580800e-03,
                   9.33832668e-13, 0.00000000e+00, 1.54477241e-28, 1.57599733e-03,
                   -9.75297474e-16, 0.00000000e+00, -1.12994919e-12, 0.00000000e+00,
                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -4.92710809e-13,
                   -3.35526972e-02, -6.87029169e-14, 2.05431134e-02, 0.00000000e+00,
                   -1.74246910e-17, -1.72962999e-12, -1.42398875e-28, 0.00000000e+00,
                   1.86216420e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                   3.95580800e-03, -9.06696757e-14, -2.56291259e-02, 4.00943618e-13,
                   0.00000000e+00, 1.60041532e-28, 5.48983891e-03, -1.24777527e-15,
                   0.00000000e+00, -3.47123236e-13, 0.00000000e+00, 0.00000000e+00,
                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                   8.92611494e-13, 2.05431134e-02, 3.37778485e-13, -1.91578104e-02,
                   0.00000000e+00, -7.56085638e-16, 4.03623843e-12, 3.60599613e-28,
                   0.00000000e+00, -3.06402210e-03, 0.00000000e+00, 0.00000000e+00,
                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -1.34972889e-02,
                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -1.06079845e-12,
                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                   0.00000000e+00, 1.44608590e-28, -1.74246910e-17, 1.32068568e-28,
                   -7.56085638e-16, 0.00000000e+00, -1.34972889e-02, -2.71996518e-27,
                   -1.06079845e-12, 0.00000000e+00, 5.13813448e-14, 0.00000000e+00,
                   0.00000000e+00, 0.00000000e+00, 1.57599733e-03, -1.67584099e-12,
                   5.48983891e-03, 3.99896369e-12, 0.00000000e+00, -3.12318680e-27,
                   -5.93842553e-03, 9.07873922e-15, 0.00000000e+00, -5.55676782e-11,
                   0.00000000e+00, 0.00000000e+00, -9.75297474e-16, -1.14491204e-28,
                   -1.24777527e-15, 2.42404817e-28, 0.00000000e+00, -1.02340522e-12,
                   9.07873922e-15, -3.70472179e-03, 0.00000000e+00, 3.64177367e-27,
                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                   0.00000000e+00, -1.02340522e-12, 0.00000000e+00, 0.00000000e+00,
                   0.00000000e+00, -3.70472179e-03, 0.00000000e+00, -1.06602165e-12,
                   1.86216420e-03, -1.54081158e-13, -3.06402210e-03, 0.00000000e+00,
                   5.13813448e-14, -5.08878434e-11, 2.56095053e-28, 0.00000000e+00,
                   -2.29457483e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                   0.00000000e+00, 0.00000000e+00, -8.76902435e-01]
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

    # see if we can reproduce HF numbers
    apg = APG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    apg.cache = {}
    apg.d_cache = {}
    assert abs(apg.compute_energy(include_nuc=False) - (-1.84444667247)) < 1e-7
    # set params
    apig_params = [0.995079200788, -0.059166892062, -0.054284175189, -0.036920061272,
                   -0.028848919079, -0.028847742282, -0.013108383833, -0.008485392433,
                   -0.008485285973, -0.005149411511, -1.8696828608304896]
    orbpairs = {(i, j) for i in range(20) for j in range(i+1, 20)}
    dict_orbpair_ind = {orbpair:i for i, orbpair in enumerate(orbpairs)}
    apg_params = np.zeros(len(orbpairs) + 1)
    for i, param in enumerate(apig_params[:-1]):
        apg_params[dict_orbpair_ind[(i, i+10)]] = param
    apg_params[-1] = apig_params[-1]
    # Solve with least squares solver with jacobian with apig guess
    # FIXME: fails
    # apg = APG(nelec, one_int, two_int, nuc_nuc=nuc_nuc, params=apg_params)
    # solve(apg, solver_type='least_squares', use_jac=True)
    # print(apg.compute_energy(include_nuc=False), apg.get_energy(include_nuc=False))
    # assert abs(apg.compute_energy(include_nuc=False) - (-1.8696828608304896)) < 1e-2
    # assert abs(apg.get_energy(include_nuc=False) - (-1.8696828608304896)) < 1e-2
    # # Solve with least squares solver without jacobian
    # FIXME: fails
    # apg = APG(nelec, one_int, two_int, nuc_nuc=nuc_nuc, params=apg_params)
    # solve(apg, solver_type='least_squares', use_jac=False)
    # print(apg.compute_energy(include_nuc=False), apg.get_energy(include_nuc=False))
    # assert abs(apg.compute_energy(include_nuc=False) - (-1.8696828608304896)) < 1e-7
    # assert abs(apg.get_energy(include_nuc=False) - (-1.8696828608304896)) < 1e-7
    # Solve with root (newton) solver with jacobian
    apg = APG(nelec, one_int, two_int, nuc_nuc=nuc_nuc, params=apg_params)
    solve(apg, solver_type='root', use_jac=True)
    print(apg.compute_energy(include_nuc=False), apg.get_energy(include_nuc=False))
    assert abs(apg.compute_energy(include_nuc=False) - (-1.8696828608304896)) < 1e-7
    assert abs(apg.get_energy(include_nuc=False) - (-1.8696828608304896)) < 1e-7
    # Solve with root (newton) solver without jacobian
    apg = APG(nelec, one_int, two_int, nuc_nuc=nuc_nuc, params=apg_params)
    solve(apg, solver_type='root', use_jac=False)
    print(apg.compute_energy(include_nuc=False), apg.get_energy(include_nuc=False))
    assert abs(apg.compute_energy(include_nuc=False) - (-1.8696828608304896)) < 1e-7
    assert abs(apg.get_energy(include_nuc=False) - (-1.8696828608304896)) < 1e-7
    # Solve with cma solver
    # FIXME: CMA answer is quite bad
    apg = APG(nelec, one_int, two_int, nuc_nuc=nuc_nuc, params=apg_params)
    solve(apg, solver_type='cma', use_jac=False)
    print(apg.compute_energy(include_nuc=False), apg.get_energy(include_nuc=False))
    assert abs(apg.compute_energy(include_nuc=False) - (-1.8696828608304896)) < 1e-1
    assert abs(apg.get_energy(include_nuc=False) - (-1.8696828608304896)) < 1e-1


def answer_apg_lih_sto6g():
    """ Finds the APG/STO-6G wavefunction variationally for LiH system
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
    ref_sds = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                for l in range(k+1, 12):
                    ref_sds.append(1 << i | 1 << j | 1 << k | 1 << l)
    params = np.zeros((2, 66))
    params[0, 6] = 1
    params[1, 17] = 1
    params = np.hstack((params.flat, 0))
    apg = APG(nelec, one_int, two_int, nuc_nuc=nuc_nuc, ref_sds=ref_sds, params=params)
    res = solve(apg, solver_type='variational', use_jac=True)
    print(res)
    return apg.params

@attr('slow')
def test_apg_lih_sto6g():
    """ Tests APG wavefunction using H2 with HF/6-31G** orbital

    Answers obtained from answer_apg_h2_631gdp

    HF (Electronic) Energy : -8.9472891719
    APG Energy :
    APG Coeffs :
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

    # see if we can reproduce HF numbers
    apg = APG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    apg.cache = {}
    apg.d_cache = {}
    assert abs(apg.compute_energy(include_nuc=False) - (-8.9472891719)) < 1e-7
