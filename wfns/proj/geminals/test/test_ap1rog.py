""" Tests wfns.proj.geminals.ap1rog.AP1roG
"""
from __future__ import absolute_import, division, print_function
from nose.tools import assert_raises
import numpy as np
from wfns.proj.solver import solve
from wfns.proj.geminals.ap1rog import AP1roG
from wfns.wrapper.horton import gaussian_fchk

def test_assign_ngem():
    """ Tests AP1roG.assign_ngem
    """
    test = AP1roG(4, np.ones((4, 4)), np.ones((4, 4, 4, 4)))
    # check error
    assert_raises(TypeError, test.assign_ngem, 4.0)
    assert_raises(ValueError, test.assign_ngem, 1)
    assert_raises(NotImplementedError, test.assign_ngem, 3)
    # None
    test.assign_ngem(None)
    assert test.ngem == 2


def test_assign_ref_sds():
    """ Tests AP1roG.assign_ref_sds
    """
    test = AP1roG(4, np.ones((4, 4)), np.ones((4, 4, 4, 4)))
    assert_raises(NotImplementedError, test.assign_ref_sds, 0b01010101)
    assert_raises(NotImplementedError, test.assign_ref_sds, [0b00110011, 0b01010101])
    test.assign_ref_sds(None)
    assert test.ref_sds == (0b00110011, )
    test.assign_ref_sds(0b00110011)
    assert test.ref_sds == (0b00110011, )


def test_template_coeffs():
    """ Tests wfns.proj.geminals.ap1rog.AP1roG.template_coeffs
    """
    # ngem 1
    test = AP1roG(2, np.ones((4, 4)), np.ones((4, 4, 4, 4)))
    np.allclose(test.template_coeffs, np.array([0, 0, 0]))
    # ngem 2
    test = AP1roG(4, np.ones((4, 4)), np.ones((4, 4, 4, 4)))
    np.allclose(test.template_coeffs, np.array([[0, 0],
                                                [0, 0]]))


def test_compute_overlap():
    """ Tests wfns.proj.geminals.ap1rog.AP1roG.compute_overlap
    """
    # two electrons
    test = AP1roG(2, np.ones((4, 4)), np.ones((4, 4, 4, 4)))
    test.assign_params(np.array([2, 3, 4, 0], dtype=float))
    # bad SD
    assert_raises(ValueError, test.compute_overlap, 0b00010010)
    assert_raises(ValueError, test.compute_overlap, 0b00110011)
    # overlap
    assert test.compute_overlap(0b00010001, deriv=None) == 1
    assert test.compute_overlap(0b00100010, deriv=None) == 2
    assert test.compute_overlap(0b01000100, deriv=None) == 3
    assert test.compute_overlap(0b10001000, deriv=None) == 4
    # differentiate
    assert test.compute_overlap(0b00010001, deriv=0) == 0
    assert test.compute_overlap(0b00010001, deriv=1) == 0
    assert test.compute_overlap(0b00010001, deriv=2) == 0
    assert test.compute_overlap(0b00010001, deriv=3) == 0
    assert test.compute_overlap(0b00100010, deriv=0) == 1
    assert test.compute_overlap(0b00100010, deriv=1) == 0
    assert test.compute_overlap(0b00100010, deriv=2) == 0
    assert test.compute_overlap(0b00100010, deriv=5) == 0
    assert test.compute_overlap(0b01000100, deriv=0) == 0
    assert test.compute_overlap(0b01000100, deriv=1) == 1
    assert test.compute_overlap(0b01000100, deriv=2) == 0
    assert test.compute_overlap(0b01000100, deriv=6) == 0
    assert test.compute_overlap(0b10001000, deriv=0) == 0
    assert test.compute_overlap(0b10001000, deriv=1) == 0
    assert test.compute_overlap(0b10001000, deriv=2) == 1
    assert test.compute_overlap(0b10001000, deriv=99) == 0
    # fout electrons
    test = AP1roG(4, np.ones((4, 4)), np.ones((4, 4, 4, 4)))
    test.assign_params(np.array([3, 4,
                                 7, 8, 0], dtype=float))
    # bad SD
    assert_raises(ValueError, test.compute_overlap, 0b00010010)
    assert_raises(ValueError, test.compute_overlap, 0b00110101)
    # overlap
    assert test.compute_overlap(0b00110011, deriv=None) == 1*1 + 0*0
    assert test.compute_overlap(0b01010101, deriv=None) == 1*7 + 3*0
    assert test.compute_overlap(0b10011001, deriv=None) == 1*8 + 4*0
    assert test.compute_overlap(0b01100110, deriv=None) == 0*7 + 3*1
    assert test.compute_overlap(0b10101010, deriv=None) == 0*8 + 4*1
    assert test.compute_overlap(0b11001100, deriv=None) == 3*8 + 4*7
    # differentiate
    assert test.compute_overlap(0b00110011, deriv=0) == 0
    assert test.compute_overlap(0b00110011, deriv=1) == 0
    assert test.compute_overlap(0b00110011, deriv=2) == 0
    assert test.compute_overlap(0b00110011, deriv=3) == 0
    assert test.compute_overlap(0b00110011, deriv=4) == 0
    assert test.compute_overlap(0b01010101, deriv=0) == 0
    assert test.compute_overlap(0b01010101, deriv=1) == 0
    assert test.compute_overlap(0b01010101, deriv=2) == 1
    assert test.compute_overlap(0b01010101, deriv=3) == 0
    assert test.compute_overlap(0b01010101, deriv=4) == 0
    assert test.compute_overlap(0b10011001, deriv=0) == 0
    assert test.compute_overlap(0b10011001, deriv=1) == 0
    assert test.compute_overlap(0b10011001, deriv=2) == 0
    assert test.compute_overlap(0b10011001, deriv=3) == 1
    assert test.compute_overlap(0b10011001, deriv=4) == 0
    assert test.compute_overlap(0b01100110, deriv=0) == 1
    assert test.compute_overlap(0b01100110, deriv=1) == 0
    assert test.compute_overlap(0b01100110, deriv=2) == 0
    assert test.compute_overlap(0b01100110, deriv=3) == 0
    assert test.compute_overlap(0b01100110, deriv=4) == 0
    assert test.compute_overlap(0b10101010, deriv=0) == 0
    assert test.compute_overlap(0b10101010, deriv=1) == 1
    assert test.compute_overlap(0b10101010, deriv=2) == 0
    assert test.compute_overlap(0b10101010, deriv=3) == 0
    assert test.compute_overlap(0b10101010, deriv=4) == 0
    assert test.compute_overlap(0b11001100, deriv=0) == 8
    assert test.compute_overlap(0b11001100, deriv=1) == 7
    assert test.compute_overlap(0b11001100, deriv=2) == 4
    assert test.compute_overlap(0b11001100, deriv=3) == 3
    assert test.compute_overlap(0b11001100, deriv=4) == 0


def answer_apig_h2_sto6g():
    """ Finds the APIG/STO-6G wavefunction by scanning through the coefficients for the lowest
    energy
    """
    hf_dict = gaussian_fchk('test/h2_hf_sto6g.fchk')
    nelec = 2
    one_int = hf_dict["one_int"]
    two_int = hf_dict["two_int"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]
    ap1rog = AP1roG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)

    # # plot all possible values (within normalization constraint)
    # xs = []
    # energies = []
    # for i in np.arange(-1, 1, 0.001):
    #     ap1rog.assign_params(np.array([i, 0.0]))
    #     if np.allclose(ap1rog.compute_norm(), 0):
    #         continue
    #     ap1rog.normalize()
    #     xs.append(ap1rog.params[0])
    #     energies.append(ap1rog.compute_energy(ref_sds=(0b0101, 0b1010)))

    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(xs, energies)
    # print(sorted(zip(xs, energies), key=lambda x:x[1])[0])
    # plt.show()

    # find exact minimum
    ap1rog.assign_params(np.array([-0.114, 0]))
    # manually set reference SD (need toget proper energy)
    res = solve(ap1rog, solver_type='variational', use_jac=True, ref_sds=(0b0101, 0b1010))
    print('Minimum energy')
    print(res)


def answer_apig_h2_631gdp():
    """ Finds the APIG/6-31G** wavefunction by scanning through the coefficients for the lowest
    energy
    """
    hf_dict = gaussian_fchk('test/h2_hf_631gdp.fchk')
    nelec = 2
    one_int = hf_dict["one_int"]
    two_int = hf_dict["two_int"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]
    ap1rog = AP1roG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)

    # find exact minimum
    res = solve(ap1rog, solver_type='variational', use_jac=True, ref_sds=(0b00000000010000000001,
                                                                          0b00000000100000000010,
                                                                          0b00000001000000000100,
                                                                          0b00000010000000001000,
                                                                          0b00000100000000010000,
                                                                          0b00001000000000100000,
                                                                          0b00010000000001000000,
                                                                          0b00100000000010000000,
                                                                          0b01000000000100000000,
                                                                          0b10000000001000000000,))
    print('Minimum energy')
    print(res)


def answer_apig_lih_sto6g():
    """ Finds the APIG/6-31G** wavefunction by scanning through the coefficients for the lowest
    energy
    """
    hf_dict = gaussian_fchk('test/lih_hf_sto6g.fchk')
    nelec = 4
    one_int = hf_dict["one_int"]
    two_int = hf_dict["two_int"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]
    ap1rog = AP1roG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)

    # find exact minimum
    res = solve(ap1rog, solver_type='variational', use_jac=True, ref_sds=(0b000011000011,
                                                                          0b000101000101,
                                                                          0b001001001001,
                                                                          0b010001010001,
                                                                          0b100001100001,
                                                                          0b000110000110,
                                                                          0b001010001010,
                                                                          0b010010010010,
                                                                          0b100010100010,
                                                                          0b001100001100,
                                                                          0b010100010100,
                                                                          0b100100100100,
                                                                          0b011000011000,
                                                                          0b101000101000,
                                                                          0b110000110000))
    print('Minimum energy')
    print(res)


def test_ap1rog_h2_sto6g():
    """ Tests ground state APIG wavefunction using H2 with HF/STO-6G orbital

    Answers obtained from answer_apig_h2_sto6g

    HF (Electronic) Energy : -1.838434256
    AP1roG Energy : -1.859089844148
    AP1roG Coeffs : [-0.113735924428061]
    """
    hf_dict = gaussian_fchk('test/h2_hf_sto6g.fchk')
    nelec = 2
    one_int = hf_dict["one_int"]
    two_int = hf_dict["two_int"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]

    ap1rog = AP1roG(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc)
    # see if we can reproduce HF numbers
    ap1rog.cache = {}
    ap1rog.d_cache = {}
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-1.838434256)) < 1e-7

    # Solve with least squares with Jacobian
    ap1rog = AP1roG(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc)
    solve(ap1rog, solver_type='least_squares', use_jac=True)
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-1.859089844148)) < 1e-8
    assert np.allclose(ap1rog.params[:-1], -0.113735924428061)
    # Solve with least squares without Jacobian
    ap1rog = AP1roG(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc)
    solve(ap1rog, solver_type='least_squares', use_jac=False)
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-1.859089844148)) < 1e-8
    assert np.allclose(ap1rog.params[:-1], -0.113735924428061)
    # Solve with root with Jacobian
    ap1rog = AP1roG(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc)
    solve(ap1rog, solver_type='root', use_jac=True)
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-1.859089844148)) < 1e-8
    assert np.allclose(ap1rog.params[:-1], -0.113735924428061)
    # Solve with root without Jacobian
    ap1rog = AP1roG(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc)
    solve(ap1rog, solver_type='root', use_jac=False)
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-1.859089844148)) < 1e-8
    assert np.allclose(ap1rog.params[:-1], -0.113735924428061)
    # CMA is quite terrible here
    # ap1rog = AP1roG(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc)
    # solve(ap1rog, solver_type='cma', use_jac=False)
    # print(ap1rog.compute_energy(include_nuc=False))
    # print(ap1rog.params)
    # assert abs(ap1rog.compute_energy(include_nuc=False) - (-1.859089844148)) < 1e-4


def test_ap1rog_h2_631gdp():
    """ Tests ground state APIG wavefunction using H2 with HF/6-31G** orbital

    Answers obtained from answer_apig_h2_631gdp

    HF (Electronic) Energy : -1.838434256
    AP1roG Energy : -1.8696828608304892
    AP1roG Coeffs : [-0.05949796, -0.05454253, -0.03709503, -0.02899231, -0.02899231, -0.01317386,
                     -0.00852702, -0.00852702, -0.00517996]
    """
    hf_dict = gaussian_fchk('test/h2_hf_631gdp.fchk')
    nelec = 2
    one_int = hf_dict["one_int"]
    two_int = hf_dict["two_int"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]

    ap1rog = AP1roG(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc)
    # see if we can reproduce HF numbers
    ap1rog.cache = {}
    ap1rog.d_cache = {}
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-1.84444667247)) < 1e-7

    # Solve with least squares with Jacobian
    ap1rog = AP1roG(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc)
    solve(ap1rog, solver_type='least_squares', use_jac=True)
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-1.8696828608304892)) < 1e-8
    assert np.allclose(ap1rog.params[:-1], np.array([-0.05949796, -0.05454253, -0.03709503,
                                                     -0.02899231, -0.02899231, -0.01317386,
                                                     -0.00852702, -0.00852702, -0.00517996]))
    # Solve with least squares without Jacobian
    ap1rog = AP1roG(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc)
    solve(ap1rog, solver_type='least_squares', use_jac=False)
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-1.8696828608304892)) < 1e-8
    assert np.allclose(ap1rog.params[:-1], np.array([-0.05949796, -0.05454253, -0.03709503,
                                                     -0.02899231, -0.02899231, -0.01317386,
                                                     -0.00852702, -0.00852702, -0.00517996]))
    # Solve with root with Jacobian
    ap1rog = AP1roG(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc)
    solve(ap1rog, solver_type='root', use_jac=True)
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-1.8696828608304892)) < 1e-8
    assert np.allclose(ap1rog.params[:-1], np.array([-0.05949796, -0.05454253, -0.03709503,
                                                     -0.02899231, -0.02899231, -0.01317386,
                                                     -0.00852702, -0.00852702, -0.00517996]))
    # Solve with root without Jacobian
    ap1rog = AP1roG(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc)
    solve(ap1rog, solver_type='root', use_jac=False)
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-1.8696828608304892)) < 1e-8
    assert np.allclose(ap1rog.params[:-1], np.array([-0.05949796, -0.05454253, -0.03709503,
                                                     -0.02899231, -0.02899231, -0.01317386,
                                                     -0.00852702, -0.00852702, -0.00517996]))
    # Solve with cma
    ap1rog = AP1roG(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc)
    solve(ap1rog, solver_type='cma', use_jac=False)
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-1.8696828608304892)) < 1e-4
    assert np.allclose(ap1rog.params[:-1], np.array([-0.05949796, -0.05454253, -0.03709503,
                                                     -0.02899231, -0.02899231, -0.01317386,
                                                     -0.00852702, -0.00852702, -0.00517996]))


def test_ap1rog_lih_sto6g():
    """ Tests ground state APIG wavefunction using LiH with HF/STO-6G orbital

    Answers obtained from answer_apig_lih_sto6g

    HF (Electronic) Energy : -8.9472891719
    AP1roG Energy : -8.963531105243506
    AP1roG Coeffs : [-4.18979612e-03, -1.71684359e-03, -1.71684359e-03, -1.16341258e-03,
                     -9.55342486e-03, -2.90031933e-02, -2.90031933e-02, -1.17012605e-01]
    """
    hf_dict = gaussian_fchk('test/lih_hf_sto6g.fchk')
    nelec = 4
    one_int = hf_dict["one_int"]
    two_int = hf_dict["two_int"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]

    # see if we can reproduce HF numbers
    ap1rog = AP1roG(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc)
    ap1rog.cache = {}
    ap1rog.d_cache = {}
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-8.9472891719)) < 1e-7

    # Solve with least squares with Jacobian
    ap1rog = AP1roG(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc)
    solve(ap1rog, solver_type='least_squares', use_jac=True)
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-8.963531105243506)) < 2e-8
    assert np.allclose(ap1rog.params[:-1], np.array([-4.18979612e-03, -1.71684359e-03,
                                                     -1.71684359e-03, -1.16341258e-03,
                                                     -9.55342486e-03, -2.90031933e-02,
                                                     -2.90031933e-02, -1.17012605e-01]),
                       rtol=1e-4, atol=1e-7)
    # Solve with least squares without Jacobian
    ap1rog = AP1roG(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc)
    solve(ap1rog, solver_type='least_squares', use_jac=False)
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-8.963531105243506)) < 2e-8
    assert np.allclose(ap1rog.params[:-1], np.array([-4.18979612e-03, -1.71684359e-03,
                                                     -1.71684359e-03, -1.16341258e-03,
                                                     -9.55342486e-03, -2.90031933e-02,
                                                     -2.90031933e-02, -1.17012605e-01]),
                       rtol=1e-4, atol=1e-7)
    # Solve with root with Jacobian
    ap1rog = AP1roG(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc)
    solve(ap1rog, solver_type='root', use_jac=True)
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-8.963531105243506)) < 1e-7
    assert np.allclose(ap1rog.params[:-1], np.array([-4.18979612e-03, -1.71684359e-03,
                                                     -1.71684359e-03, -1.16341258e-03,
                                                     -9.55342486e-03, -2.90031933e-02,
                                                     -2.90031933e-02, -1.17012605e-01]),
                       rtol=1e-3, atol=1e-6)
    # Solve with root without Jacobian
    ap1rog = AP1roG(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc)
    solve(ap1rog, solver_type='root', use_jac=False)
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-8.963531105243506)) < 1e-7
    assert np.allclose(ap1rog.params[:-1], np.array([-4.18979612e-03, -1.71684359e-03,
                                                     -1.71684359e-03, -1.16341258e-03,
                                                     -9.55342486e-03, -2.90031933e-02,
                                                     -2.90031933e-02, -1.17012605e-01]),
                       rtol=1e-3, atol=1e-6)
    # Solve with cma
    ap1rog = AP1roG(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc)
    solve(ap1rog, solver_type='cma', use_jac=False)
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-8.963531105243506)) < 1e-5
    assert np.allclose(ap1rog.params[:-1], np.array([-4.18979612e-03, -1.71684359e-03,
                                                     -1.71684359e-03, -1.16341258e-03,
                                                     -9.55342486e-03, -2.90031933e-02,
                                                     -2.90031933e-02, -1.17012605e-01]),
                       rtol=1e-3, atol=1e-6)
