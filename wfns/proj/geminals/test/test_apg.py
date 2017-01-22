""" Tests wfns.proj.geminals.apg
"""
from __future__ import absolute_import, division, print_function
from nose.plugins.attrib import attr
from nose.tools import assert_raises
import numpy as np
from wfns.graphs import generate_complete_pmatch
from wfns.proj.geminals.apg import APG
from wfns.proj.solver import solve
from wfns.wrapper.horton import gaussian_fchk

def test_template_orbpairs():
    """ Tests wfns.proj.geminals.apg.APG.template_orbpairs
    """
    test = APG(2, np.ones((4, 4)), np.ones((4, 4, 4, 4)))
    assert test.template_orbpairs == ((0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
                                      (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
                                      (2, 3), (2, 4), (2, 5), (2, 6), (2, 7),
                                      (3, 4), (3, 5), (3, 6), (3, 7),
                                      (4, 5), (4, 6), (4, 7),
                                      (5, 6), (5, 7),
                                      (6, 7))
    test = APG(4, np.ones((4, 4)), np.ones((4, 4, 4, 4)))
    assert test.template_orbpairs == ((0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
                                      (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
                                      (2, 3), (2, 4), (2, 5), (2, 6), (2, 7),
                                      (3, 4), (3, 5), (3, 6), (3, 7),
                                      (4, 5), (4, 6), (4, 7),
                                      (5, 6), (5, 7),
                                      (6, 7))
    test = APG(6, np.ones((4, 4)), np.ones((4, 4, 4, 4)))
    assert test.template_orbpairs == ((0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
                                      (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
                                      (2, 3), (2, 4), (2, 5), (2, 6), (2, 7),
                                      (3, 4), (3, 5), (3, 6), (3, 7),
                                      (4, 5), (4, 6), (4, 7),
                                      (5, 6), (5, 7),
                                      (6, 7))
    test = APG(8, np.ones((4, 4)), np.ones((4, 4, 4, 4)))
    assert test.template_orbpairs == ((0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
                                      (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
                                      (2, 3), (2, 4), (2, 5), (2, 6), (2, 7),
                                      (3, 4), (3, 5), (3, 6), (3, 7),
                                      (4, 5), (4, 6), (4, 7),
                                      (5, 6), (5, 7),
                                      (6, 7))


def test_template_coeffs():
    """ Tests wfns.proj.geminals.apg.APG.template_coeffs
    """
    test = APG(2, np.ones((4, 4)), np.ones((4, 4, 4, 4)))
    # ngem 1
    print(test.template_coeffs)
    assert np.allclose(test.template_coeffs, np.array([0]*3 + [1] + [0]*24))
    # ngem 2
    test.assign_ngem(2)
    assert np.allclose(test.template_coeffs, np.array([[0]*3 + [1] + [0]*24,
                                                       [0]*28]))
    # ngem 3
    test.assign_ngem(3)
    assert np.allclose(test.template_coeffs, np.array([[0]*3 + [1] + [0]*24,
                                                       [0]*28,
                                                       [0]*28]))
    # different reference sd
    test.assign_ngem(1)
    test.assign_ref_sds(0b00100010)
    # NOTE: this still uses ground state HF as reference
    assert np.allclose(test.template_coeffs, np.array([0]*3 + [1] + [0]*24))

    test = APG(4, np.ones((4, 4)), np.ones((4, 4, 4, 4)))
    # ngem 2
    test.assign_ngem(2)
    assert np.allclose(test.template_coeffs, np.array([[0]*3 + [1] + [0]*24,
                                                       [0]*10 + [1] + [0]*17]))
    # ngem 3
    test.assign_ngem(3)
    assert np.allclose(test.template_coeffs, np.array([[0]*3 + [1] + [0]*24,
                                                       [0]*10 + [1] + [0]*17,
                                                       [0]*28]))


def test_assign_pmatch_generator():
    """Tests wfns.proj.geminals.apg.APG.assign_pmatch_generator
    """
    test = APG(2, np.ones((4, 4)), np.ones((4, 4, 4, 4)))
    # default
    test.assign_pmatch_generator(None)
    assert test.pmatch_generator == generate_complete_pmatch
    # if not callable (function)
    assert_raises(TypeError, test.assign_pmatch_generator, 1)
    assert_raises(TypeError, test.assign_pmatch_generator, [])
    assert_raises(TypeError, test.assign_pmatch_generator, {})
    # if function does not return tuple of 2-tuples
    func = lambda x: (((0, 1), (2, 3, 4)), )
    assert_raises(ValueError, test.assign_pmatch_generator, func)
    func = lambda x: (((1, ), (2, 3)), )
    assert_raises(ValueError, test.assign_pmatch_generator, func)
    # if function does not return tuple of npair 2-tuple
    func = lambda x: (((0, 1), ), )
    assert_raises(ValueError, test.assign_pmatch_generator, func)
    func = lambda x: (((0, 1), (2, 3), (4, 5)), )
    assert_raises(ValueError, test.assign_pmatch_generator, func)
    # if function does not return perfect matching that corresponds to given occupied indices
    func = lambda x: (((0, 1), (2, 3)), )
    assert_raises(ValueError, test.assign_pmatch_generator, func)
    func = lambda x: (zip(x[0::2], x[1::2]) + [(0, 1)], )
    assert_raises(ValueError, test.assign_pmatch_generator, func)
    # proper pmatch generator
    func = lambda x: (zip(x[0::2], x[1::2]), )
    test.assign_pmatch_generator(func)
    assert test.pmatch_generator == func


@attr('hello')
def test_compute_overlap():
    """Tests wfns.proj.geminals.apg.APG.assign_compute_overlap
    """
    # two electrons
    test = APG(2, np.ones((4, 4)), np.ones((4, 4, 4, 4)))
    test.assign_params(np.hstack((np.arange(1, 29, dtype=float), 0)))
    # bad SD
    assert_raises(ValueError, test.compute_overlap, 0b00110011)
    # overlap
    assert test.compute_overlap(0b00000011, deriv=None) == -1
    assert test.compute_overlap(0b00010001, deriv=None) == -4
    assert test.compute_overlap(0b00100010, deriv=None) == -11
    assert test.compute_overlap(0b01000100, deriv=None) == -17
    assert test.compute_overlap(0b10001000, deriv=None) == -22
    # differentiate
    assert test.compute_overlap(0b00010001, deriv=0) == 0
    assert test.compute_overlap(0b00010001, deriv=1) == 0
    assert test.compute_overlap(0b00010001, deriv=2) == 0
    assert test.compute_overlap(0b00010001, deriv=3) == -1
    assert test.compute_overlap(0b00010001, deriv=4) == 0
    assert test.compute_overlap(0b00010001, deriv=5) == 0
    assert test.compute_overlap(0b00010001, deriv=6) == 0
    assert test.compute_overlap(0b00010001, deriv=7) == 0
    assert test.compute_overlap(0b00010001, deriv=8) == 0
    assert test.compute_overlap(0b00010001, deriv=9) == 0
    assert test.compute_overlap(0b00010001, deriv=10) == 0
    assert test.compute_overlap(0b00010001, deriv=11) == 0
    assert test.compute_overlap(0b00010001, deriv=12) == 0
    assert test.compute_overlap(0b00010001, deriv=13) == 0
    assert test.compute_overlap(0b00010001, deriv=14) == 0
    assert test.compute_overlap(0b00010001, deriv=15) == 0
    assert test.compute_overlap(0b00010001, deriv=16) == 0
    assert test.compute_overlap(0b00010001, deriv=17) == 0
    assert test.compute_overlap(0b00010001, deriv=18) == 0
    assert test.compute_overlap(0b00010001, deriv=19) == 0
    assert test.compute_overlap(0b00010001, deriv=20) == 0
    assert test.compute_overlap(0b00010001, deriv=21) == 0
    assert test.compute_overlap(0b00010001, deriv=22) == 0
    assert test.compute_overlap(0b00010001, deriv=23) == 0
    assert test.compute_overlap(0b00010001, deriv=24) == 0
    assert test.compute_overlap(0b00010001, deriv=25) == 0
    assert test.compute_overlap(0b00010001, deriv=26) == 0
    assert test.compute_overlap(0b00010001, deriv=27) == 0
    assert test.compute_overlap(0b00010001, deriv=28) == 0

    # (1:(0, 1), 2:(0, 2), 3:(0, 3), 4:(0, 4), 5:(0, 5), 6:(0, 6), 7:(0, 7),
    #  8:(1, 2), 9:(1, 3), 10:(1, 4), 11:(1, 5), 12:(1, 6), 13:(1, 7),
    #  14:(2, 3), 15:(2, 4), 16:(2, 5), 17:(2, 6), 18:(2, 7),
    #  19:(3, 4), 20:(3, 5), 21:(3, 6), 22:(3, 7),
    #  23:(4, 5), 24:(4, 6), 25:(4, 7),
    #  26:(5, 6), 27:(5, 7),
    #  28:(6, 7))

    # four electrons
    test = APG(4, np.ones((4, 4)), np.ones((4, 4, 4, 4)))
    test.assign_params(np.hstack((np.arange(1, 57, dtype=float), 0)))
    # overlap
    assert test.compute_overlap(0b00001111, deriv=None) == ((1*(14+28) + 14*(1+28))
                                                            - (2*(9+28) + 9*(2+28))
                                                            + (3*(8+28) + 8*(3+28)))
    assert test.compute_overlap(0b00010111, deriv=None) == ((1*(15+28) + 15*(1+28))
                                                            - (2*(10+28) + 10*(2+28))
                                                            + (4*(8+28) + 8*(4+28)))
    assert test.compute_overlap(0b00110011, deriv=None) == ((1*(23+28) + 23*(1+28))
                                                            - (4*(11+28) + 11*(4+28))
                                                            + (5*(10+28) + 10*(5+28)))
    # differentiate
    assert test.compute_overlap(0b00001111, deriv=0) == 14+28
    assert test.compute_overlap(0b00001111, deriv=1) == -(9+28)
    assert test.compute_overlap(0b00001111, deriv=2) == 8+28
    assert test.compute_overlap(0b00001111, deriv=3) == 0
    assert test.compute_overlap(0b00001111, deriv=4) == 0
    assert test.compute_overlap(0b00001111, deriv=5) == 0
    assert test.compute_overlap(0b00001111, deriv=6) == 0
    assert test.compute_overlap(0b00001111, deriv=7) == 3+28
    assert test.compute_overlap(0b00001111, deriv=8) == -(2+28)
    assert test.compute_overlap(0b00001111, deriv=9) == 0
    assert test.compute_overlap(0b00001111, deriv=10) == 0
    assert test.compute_overlap(0b00001111, deriv=11) == 0
    assert test.compute_overlap(0b00001111, deriv=12) == 0
    assert test.compute_overlap(0b00001111, deriv=13) == 1+28
    assert test.compute_overlap(0b00001111, deriv=14) == 0
    assert test.compute_overlap(0b00001111, deriv=15) == 0
    assert test.compute_overlap(0b00001111, deriv=16) == 0
    assert test.compute_overlap(0b00001111, deriv=17) == 0
    assert test.compute_overlap(0b00001111, deriv=18) == 0
    assert test.compute_overlap(0b00001111, deriv=19) == 0
    assert test.compute_overlap(0b00001111, deriv=20) == 0
    assert test.compute_overlap(0b00001111, deriv=21) == 0
    assert test.compute_overlap(0b00001111, deriv=22) == 0
    assert test.compute_overlap(0b00001111, deriv=23) == 0
    assert test.compute_overlap(0b00001111, deriv=24) == 0
    assert test.compute_overlap(0b00001111, deriv=25) == 0
    assert test.compute_overlap(0b00001111, deriv=26) == 0
    assert test.compute_overlap(0b00001111, deriv=27) == 0
    assert test.compute_overlap(0b00001111, deriv=28) == 14
    assert test.compute_overlap(0b00001111, deriv=29) == -9
    assert test.compute_overlap(0b00001111, deriv=30) == 8
    assert test.compute_overlap(0b00001111, deriv=31) == 0
    assert test.compute_overlap(0b00001111, deriv=32) == 0
    assert test.compute_overlap(0b00001111, deriv=33) == 0
    assert test.compute_overlap(0b00001111, deriv=34) == 0
    assert test.compute_overlap(0b00001111, deriv=35) == 3
    assert test.compute_overlap(0b00001111, deriv=36) == -2
    assert test.compute_overlap(0b00001111, deriv=37) == 0
    assert test.compute_overlap(0b00001111, deriv=38) == 0
    assert test.compute_overlap(0b00001111, deriv=39) == 0
    assert test.compute_overlap(0b00001111, deriv=40) == 0
    assert test.compute_overlap(0b00001111, deriv=41) == 1
    assert test.compute_overlap(0b00001111, deriv=42) == 0
    assert test.compute_overlap(0b00001111, deriv=43) == 0
    assert test.compute_overlap(0b00001111, deriv=44) == 0
    assert test.compute_overlap(0b00001111, deriv=45) == 0
    assert test.compute_overlap(0b00001111, deriv=46) == 0
    assert test.compute_overlap(0b00001111, deriv=47) == 0
    assert test.compute_overlap(0b00001111, deriv=48) == 0
    assert test.compute_overlap(0b00001111, deriv=49) == 0
    assert test.compute_overlap(0b00001111, deriv=50) == 0
    assert test.compute_overlap(0b00001111, deriv=51) == 0
    assert test.compute_overlap(0b00001111, deriv=52) == 0
    assert test.compute_overlap(0b00001111, deriv=53) == 0
    assert test.compute_overlap(0b00001111, deriv=54) == 0
    assert test.compute_overlap(0b00001111, deriv=55) == 0
    assert test.compute_overlap(0b00001111, deriv=56) == 0


def test_normalize():
    """Tests wfns.proj.geminals.apg.APG.normalize
    """
    test = APG(2, np.ones((2, 2)), np.ones((2, 2, 2, 2)), dtype=float)
    #check
    test.assign_ref_sds(0b0101)
    test.assign_params(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    assert_raises(ValueError, test.normalize)
    test.assign_dtype(complex)
    test.assign_params(np.array([1j, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    assert_raises(ValueError, test.normalize)

    # one geminal, one reference
    test = APG(2, np.ones((2, 2)), np.ones((2, 2, 2, 2)), dtype=float)
    test.assign_params(np.array([2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    test.assign_ref_sds(0b0101)
    test.normalize()
    assert test.compute_norm() == 1
    # one geminal, two reference
    test.assign_params(np.array([2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    test.assign_ref_sds([0b0101, 0b1010])
    test.normalize()
    assert test.compute_norm() == 1

    test = APG(4, np.ones((4, 4)), np.ones((4, 4, 4, 4)))
    # multiple geminal, one reference
    test.assign_params(np.array(range(1, 57) + [0], dtype=float))
    test.assign_ref_sds(0b00110011)
    test.normalize()
    assert test.compute_norm() == 1
    # multiple geminal, multiple reference
    test.assign_params(np.array(range(1, 57) + [0], dtype=float))
    test.assign_ref_sds([0b00110011, 0b01010101])
    test.normalize()
    assert test.compute_norm() == 1


def answer_apg_h2_sto6g():
    """ Finds the APG/STO-6G wavefunction variationally for H2 system
    """
    hf_dict = gaussian_fchk('test/h2_hf_sto6g.fchk')
    nelec = 2
    one_int = hf_dict["one_int"]
    two_int = hf_dict["two_int"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]
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
    hf_dict = gaussian_fchk('test/h2_hf_sto6g.fchk')
    nelec = 2
    one_int = hf_dict["one_int"]
    two_int = hf_dict["two_int"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]

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
    hf_dict = gaussian_fchk('test/h2_hf_631gdp.fchk')
    nelec = 2
    one_int = hf_dict["one_int"]
    two_int = hf_dict["two_int"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]
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
    hf_dict = gaussian_fchk('test/h2_hf_631gdp.fchk')
    nelec = 2
    one_int = hf_dict["one_int"]
    two_int = hf_dict["two_int"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]

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
    apg = APG(nelec, one_int, two_int, nuc_nuc=nuc_nuc, params=apg_params)
    solve(apg, solver_type='least_squares', use_jac=True)
    print(apg.compute_energy(include_nuc=False), apg.get_energy(include_nuc=False))
    assert abs(apg.compute_energy(include_nuc=False) - (-1.8696828608304896)) < 1e-2
    assert abs(apg.get_energy(include_nuc=False) - (-1.8696828608304896)) < 1e-2
    # Solve with least squares solver without jacobian
    apg = APG(nelec, one_int, two_int, nuc_nuc=nuc_nuc, params=apg_params)
    solve(apg, solver_type='least_squares', use_jac=False)
    print(apg.compute_energy(include_nuc=False), apg.get_energy(include_nuc=False))
    assert abs(apg.compute_energy(include_nuc=False) - (-1.8696828608304896)) < 1e-7
    assert abs(apg.get_energy(include_nuc=False) - (-1.8696828608304896)) < 1e-7
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
    hf_dict = gaussian_fchk('test/lih_hf_sto6g.fchk')
    nelec = 4
    one_int = hf_dict["one_int"]
    two_int = hf_dict["two_int"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]
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
    hf_dict = gaussian_fchk('test/lih_hf_sto6g.fchk')
    nelec = 4
    one_int = hf_dict["one_int"]
    two_int = hf_dict["two_int"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]

    # see if we can reproduce HF numbers
    apg = APG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    apg.cache = {}
    apg.d_cache = {}
    assert abs(apg.compute_energy(include_nuc=False) - (-8.9472891719)) < 1e-7
