""" Tests wfns.proj.geminals.ap1rog.AP1roG
"""
from __future__ import absolute_import, division, print_function
from nose.tools import assert_raises
import numpy as np
import scipy
from wfns.proj.solver import solve
from wfns.proj.geminals.ap1rog import AP1roG
from wfns.tools import find_datafile

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
    assert np.allclose(test.template_coeffs, np.array([0, 0, 0]))
    # ngem 2
    test = AP1roG(4, np.ones((4, 4)), np.ones((4, 4, 4, 4)))
    assert np.allclose(test.template_coeffs, np.array([[0, 0],
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


def answer_ap1rog_h2_sto6g():
    """ Finds the AP1roG/STO-6G wavefunction by scanning through the coefficients for the lowest
    energy
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
    ap1rog_ground = AP1roG(nelec, one_int, two_int, nuc_nuc=nuc_nuc, ref_sds=(0b0101, ))
    ap1rog_excited = AP1roG(nelec, one_int, two_int, nuc_nuc=nuc_nuc, ref_sds=(0b1010, ))

    # plot all possible values (within normalization constraint)
    xs_ground = []
    energies_ground = []
    xs_excited = []
    energies_excited = []
    for i in np.arange(-1, 1, 0.001):
        ap1rog_ground.assign_params(np.array([i, 0.0]))
        if np.allclose(ap1rog_ground.compute_norm(), 0):
            continue
        ap1rog_ground.normalize()
        xs_ground.append(ap1rog_ground.params[0])
        energies_ground.append(ap1rog_ground.compute_energy(ref_sds=(0b0101, 0b1010)))
        ap1rog_excited.assign_params(np.array([i, 0.0]))
        if np.allclose(ap1rog_excited.compute_norm(), 0):
            continue
        ap1rog_excited.normalize()
        xs_excited.append(ap1rog_excited.params[0])
        energies_excited.append(ap1rog_excited.compute_energy(ref_sds=(0b0101, 0b1010)))

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.scatter(xs_ground, energies_ground)
    print(sorted(zip(xs_ground, energies_ground), key=lambda x:x[1])[0])
    ax = fig.add_subplot(212)
    ax.scatter(xs_excited, energies_excited)
    print(sorted(zip(xs_excited, energies_excited), key=lambda x:x[1], reverse=True)[0])
    plt.show()

    # find exact minimum
    ap1rog_ground.assign_params(np.array([-0.114, 0]))
    # manually set reference SD (need to get proper energy)
    res = solve(ap1rog_ground, solver_type='variational', use_jac=True, ref_sds=(0b0101, 0b1010))
    print('Minimum energy')
    print(res)

    # find exact maximum
    def max_energy(params):
        """ Find maximum energy"""
        ap1rog_excited.assign_params(params)
        return -ap1rog_excited.compute_energy(ref_sds=(0b0101, 0b1010))
    res = scipy.optimize.minimize(max_energy, np.array([0.114, 0]))
    print('Maximum energy')
    print(res)

def answer_ap1rog_h2_631gdp():
    """ Finds the AP1roG/6-31G** wavefunction by scanning through the coefficients for the lowest
    energy
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


def answer_ap1rog_lih_sto6g():
    """ Finds the AP1roG/6-31G** wavefunction by scanning through the coefficients for the lowest
    energy
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


def test_ap1rog_h2_sto6g_ground():
    """ Tests ground state AP1roG wavefunction using H2 with HF/STO-6G orbital

    Answers obtained from answer_ap1rog_h2_sto6g

    HF (Electronic) Energy : -1.838434256
    AP1roG Energy : -1.859089844148
    AP1roG Coeffs : [-0.113735924428061]
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

    ap1rog = AP1roG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    # see if we can reproduce HF numbers
    ap1rog.cache = {}
    ap1rog.d_cache = {}
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-1.838434256)) < 1e-7

    # Solve with least squares with Jacobian
    ap1rog = AP1roG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(ap1rog, solver_type='least_squares', use_jac=True)
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-1.859089844148)) < 1e-8
    assert np.allclose(ap1rog.params[:-1], -0.113735924428061)
    # Solve with least squares without Jacobian
    ap1rog = AP1roG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(ap1rog, solver_type='least_squares', use_jac=False)
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-1.859089844148)) < 1e-8
    assert np.allclose(ap1rog.params[:-1], -0.113735924428061)
    # Solve with root with Jacobian
    ap1rog = AP1roG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(ap1rog, solver_type='root', use_jac=True)
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-1.859089844148)) < 1e-8
    assert np.allclose(ap1rog.params[:-1], -0.113735924428061)
    # Solve with root without Jacobian
    ap1rog = AP1roG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(ap1rog, solver_type='root', use_jac=False)
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-1.859089844148)) < 1e-8
    assert np.allclose(ap1rog.params[:-1], -0.113735924428061)
    # # Solve with CMA
    # # FIXME: CMA is quite terrible here
    # ap1rog = AP1roG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    # solve(ap1rog, solver_type='cma', use_jac=False)
    # assert abs(ap1rog.compute_energy(include_nuc=False) - (-1.859089844148)) < 1e-4


def test_ap1rog_h2_sto6g_excited():
    """ Tests excited state AP1roG wavefunction using H2 with HF/STO-6G orbital

    Answers obtained from answer_ap1rog_h2_sto6g

    HF (Electronic) Energy : -1.838434256
    AP1roG Energy : -0.24166486974216012
    AP1roG Coeffs : [0.113735939809]
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

    ap1rog = AP1roG(nelec, one_int, two_int, nuc_nuc=nuc_nuc, ref_sds=(0b1010,))

    # Solve with least squares with Jacobian
    ap1rog = AP1roG(nelec, one_int, two_int, nuc_nuc=nuc_nuc, ref_sds=(0b1010,))
    solve(ap1rog, solver_type='least_squares', use_jac=True)
    print(ap1rog.compute_energy(include_nuc=False))
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-0.24166486974216012)) < 1e-7
    print(ap1rog.params)
    assert np.allclose(ap1rog.params[:-1], 0.113735939809)
    # Solve with least squares without Jacobian
    ap1rog = AP1roG(nelec, one_int, two_int, nuc_nuc=nuc_nuc, ref_sds=(0b1010,))
    solve(ap1rog, solver_type='least_squares', use_jac=False)
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-0.24166486974216012)) < 1e-7
    assert np.allclose(ap1rog.params[:-1], 0.113735939809)
    # Solve with root with Jacobian
    ap1rog = AP1roG(nelec, one_int, two_int, nuc_nuc=nuc_nuc, ref_sds=(0b1010,))
    solve(ap1rog, solver_type='root', use_jac=True)
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-0.24166486974216012)) < 1e-7
    assert np.allclose(ap1rog.params[:-1], 0.113735939809)
    # Solve with root without Jacobian
    ap1rog = AP1roG(nelec, one_int, two_int, nuc_nuc=nuc_nuc, ref_sds=(0b1010,))
    solve(ap1rog, solver_type='root', use_jac=False)
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-0.24166486974216012)) < 1e-7
    assert np.allclose(ap1rog.params[:-1], 0.113735939809)
    # # Solve with CMA
    # # FIXME: CMA is quite terrible here
    # ap1rog = AP1roG(nelec, one_int, two_int, nuc_nuc=nuc_nuc, ref_sds=(0b1010,))
    # solve(ap1rog, solver_type='cma', use_jac=False)
    # assert abs(ap1rog.compute_energy(include_nuc=False) - (-0.24166486974216012)) < 1e-4


def test_ap1rog_h2_631gdp():
    """ Tests ground state AP1roG wavefunction using H2 with HF/6-31G** orbital

    Answers obtained from answer_ap1rog_h2_631gdp

    HF (Electronic) Energy : -1.838434256
    AP1roG Energy : -1.8696828608304892
    AP1roG Coeffs : [-0.05949796, -0.05454253, -0.03709503, -0.02899231, -0.02899231, -0.01317386,
                     -0.00852702, -0.00852702, -0.00517996]
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
    ap1rog = AP1roG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    # see if we can reproduce HF numbers
    ap1rog.cache = {}
    ap1rog.d_cache = {}
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-1.84444667247)) < 1e-7

    # Solve with least squares with Jacobian
    ap1rog = AP1roG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(ap1rog, solver_type='least_squares', use_jac=True)
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-1.8696828608304892)) < 1e-8
    assert np.allclose(ap1rog.params[:-1], np.array([-0.05949796, -0.05454253, -0.03709503,
                                                     -0.02899231, -0.02899231, -0.01317386,
                                                     -0.00852702, -0.00852702, -0.00517996]))
    # Solve with least squares without Jacobian
    ap1rog = AP1roG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(ap1rog, solver_type='least_squares', use_jac=False)
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-1.8696828608304892)) < 1e-8
    # FIXME: get_energy is quite a bit different than the compute_energy for some reason
    # assert abs(ap1rog.get_energy(include_nuc=False) - (-1.8696828608304892)) < 1e-8
    assert np.allclose(ap1rog.params[:-1], np.array([-0.05949796, -0.05454253, -0.03709503,
                                                     -0.02899231, -0.02899231, -0.01317386,
                                                     -0.00852702, -0.00852702, -0.00517996]))
    # Solve with root with Jacobian
    ap1rog = AP1roG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(ap1rog, solver_type='root', use_jac=True)
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-1.8696828608304892)) < 1e-8
    assert abs(ap1rog.get_energy(include_nuc=False) - (-1.8696828608304892)) < 1e-8
    assert np.allclose(ap1rog.params[:-1], np.array([-0.05949796, -0.05454253, -0.03709503,
                                                     -0.02899231, -0.02899231, -0.01317386,
                                                     -0.00852702, -0.00852702, -0.00517996]))
    # Solve with root without Jacobian
    ap1rog = AP1roG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(ap1rog, solver_type='root', use_jac=False)
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-1.8696828608304892)) < 1e-8
    assert abs(ap1rog.get_energy(include_nuc=False) - (-1.8696828608304892)) < 1e-8
    assert np.allclose(ap1rog.params[:-1], np.array([-0.05949796, -0.05454253, -0.03709503,
                                                     -0.02899231, -0.02899231, -0.01317386,
                                                     -0.00852702, -0.00852702, -0.00517996]))
    # # Solve with cma
    # # FIXME: CMA answer is quite bad (doesn't always pass)
    # ap1rog = AP1roG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    # solve(ap1rog, solver_type='cma', use_jac=False)
    # print(ap1rog.compute_energy(include_nuc=False), ap1rog.get_energy(include_nuc=False))
    # assert abs(ap1rog.compute_energy(include_nuc=False) - (-1.8696828608304892)) < 1e-2
    # assert abs(ap1rog.get_energy(include_nuc=False) - (-1.8696828608304892)) < 1e-2
    # assert np.allclose(ap1rog.params[:-1], np.array([-0.05949796, -0.05454253, -0.03709503,
    #                                                  -0.02899231, -0.02899231, -0.01317386,
    #                                                  -0.00852702, -0.00852702, -0.00517996]),
    #                    atol=1)


def test_ap1rog_lih_sto6g():
    """ Tests ground state AP1roG wavefunction using LiH with HF/STO-6G orbital

    Answers obtained from answer_ap1rog_lih_sto6g

    HF (Electronic) Energy : -8.9472891719
    AP1roG Energy : -8.963531105243506
    AP1roG Coeffs : [-4.18979612e-03, -1.71684359e-03, -1.71684359e-03, -1.16341258e-03,
                     -9.55342486e-03, -2.90031933e-02, -2.90031933e-02, -1.17012605e-01]
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
    ap1rog = AP1roG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    ap1rog.cache = {}
    ap1rog.d_cache = {}
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-8.9472891719)) < 1e-7

    # Solve with least squares with Jacobian
    ap1rog = AP1roG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(ap1rog, solver_type='least_squares', use_jac=True)
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-8.963531105243506)) < 2e-8
    assert np.allclose(ap1rog.params[:-1], np.array([-4.18979612e-03, -1.71684359e-03,
                                                     -1.71684359e-03, -1.16341258e-03,
                                                     -9.55342486e-03, -2.90031933e-02,
                                                     -2.90031933e-02, -1.17012605e-01]),
                       rtol=1e-4, atol=1e-7)
    # Solve with least squares without Jacobian
    ap1rog = AP1roG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(ap1rog, solver_type='least_squares', use_jac=False)
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-8.963531105243506)) < 2e-8
    assert np.allclose(ap1rog.params[:-1], np.array([-4.18979612e-03, -1.71684359e-03,
                                                     -1.71684359e-03, -1.16341258e-03,
                                                     -9.55342486e-03, -2.90031933e-02,
                                                     -2.90031933e-02, -1.17012605e-01]),
                       rtol=1e-4, atol=1e-7)
    # Solve with root with Jacobian
    ap1rog = AP1roG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(ap1rog, solver_type='root', use_jac=True)
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-8.963531105243506)) < 1e-7
    assert np.allclose(ap1rog.params[:-1], np.array([-4.18979612e-03, -1.71684359e-03,
                                                     -1.71684359e-03, -1.16341258e-03,
                                                     -9.55342486e-03, -2.90031933e-02,
                                                     -2.90031933e-02, -1.17012605e-01]),
                       rtol=1e-3, atol=1e-6)
    # Solve with root without Jacobian
    ap1rog = AP1roG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(ap1rog, solver_type='root', use_jac=False)
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-8.963531105243506)) < 1e-7
    assert np.allclose(ap1rog.params[:-1], np.array([-4.18979612e-03, -1.71684359e-03,
                                                     -1.71684359e-03, -1.16341258e-03,
                                                     -9.55342486e-03, -2.90031933e-02,
                                                     -2.90031933e-02, -1.17012605e-01]),
                       rtol=1e-3, atol=1e-6)
    # Solve with cma
    ap1rog = AP1roG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(ap1rog, solver_type='cma', use_jac=False)
    assert abs(ap1rog.compute_energy(include_nuc=False) - (-8.963531105243506)) < 1e-5
    assert np.allclose(ap1rog.params[:-1], np.array([-4.18979612e-03, -1.71684359e-03,
                                                     -1.71684359e-03, -1.16341258e-03,
                                                     -9.55342486e-03, -2.90031933e-02,
                                                     -2.90031933e-02, -1.17012605e-01]),
                       rtol=1e-3, atol=1e-6)
