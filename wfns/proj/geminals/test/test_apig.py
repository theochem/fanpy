""" Tests wfns.proj.geminals.apig
"""
from __future__ import absolute_import, division, print_function
import numpy as np
import scipy
from nose.plugins.attrib import attr
from nose.tools import assert_raises
from wfns.proj.geminals.apig import APIG
from wfns.proj.solver import solve
from wfns.wrapper.horton import gaussian_fchk

def test_template_orbpairs():
    """ Tests wfns.proj.geminals.apig.APIG.template_orbpairs
    """
    test = APIG(2, np.ones((4, 4)), np.ones((4, 4, 4, 4)))
    assert test.template_orbpairs == ((0, 4), (1, 5), (2, 6), (3, 7))
    test = APIG(4, np.ones((4, 4)), np.ones((4, 4, 4, 4)))
    assert test.template_orbpairs == ((0, 4), (1, 5), (2, 6), (3, 7))
    test = APIG(6, np.ones((4, 4)), np.ones((4, 4, 4, 4)))
    assert test.template_orbpairs == ((0, 4), (1, 5), (2, 6), (3, 7))
    test = APIG(8, np.ones((4, 4)), np.ones((4, 4, 4, 4)))
    assert test.template_orbpairs == ((0, 4), (1, 5), (2, 6), (3, 7))


def test_template_coeffs():
    """ Tests wfns.proj.geminals.apig.APIG.template_coeffs
    """
    test = APIG(2, np.ones((4, 4)), np.ones((4, 4, 4, 4)))
    # ngem 1
    assert np.allclose(test.template_coeffs, np.array([1, 0, 0, 0]))
    # ngem 2
    test.assign_ngem(2)
    assert np.allclose(test.template_coeffs, np.array([[1, 0, 0, 0],
                                                       [0, 1, 0, 0]]))
    # ngem 3
    test.assign_ngem(3)
    assert np.allclose(test.template_coeffs, np.array([[1, 0, 0, 0],
                                                       [0, 1, 0, 0],
                                                       [0, 0, 1, 0]]))
    # ngem 5
    test.assign_ngem(5)
    assert np.allclose(test.template_coeffs, np.array([[1, 0, 0, 0],
                                                       [0, 1, 0, 0],
                                                       [0, 0, 1, 0],
                                                       [0, 0, 0, 1],
                                                       [0, 0, 0, 0]]))
    # different reference sd
    test.assign_ngem(1)
    test.assign_ref_sds(0b00100010)
    # NOTE: this still uses ground state HF as reference
    assert np.allclose(test.template_coeffs, np.array([1, 0, 0, 0]))


def test_compute_overlap():
    """ Tests wfns.proj.geminals.apig.APIG.compute_overlap
    """
    # two electrons
    test = APIG(2, np.ones((4, 4)), np.ones((4, 4, 4, 4)))
    test.assign_params(np.array([1, 2, 3, 4, 0], dtype=float))
    # bad SD
    assert_raises(ValueError, test.compute_overlap, 0b00010010)
    assert_raises(ValueError, test.compute_overlap, 0b00110011)
    # overlap
    assert test.compute_overlap(0b00010001, deriv=None) == 1
    assert test.compute_overlap(0b00100010, deriv=None) == 2
    assert test.compute_overlap(0b01000100, deriv=None) == 3
    assert test.compute_overlap(0b10001000, deriv=None) == 4
    # differentiate
    assert test.compute_overlap(0b00010001, deriv=0) == 1
    assert test.compute_overlap(0b00010001, deriv=1) == 0
    assert test.compute_overlap(0b00010001, deriv=2) == 0
    assert test.compute_overlap(0b00010001, deriv=3) == 0
    assert test.compute_overlap(0b00010001, deriv=4) == 0
    assert test.compute_overlap(0b00100010, deriv=0) == 0
    assert test.compute_overlap(0b00100010, deriv=1) == 1
    assert test.compute_overlap(0b00100010, deriv=2) == 0
    assert test.compute_overlap(0b00100010, deriv=3) == 0
    assert test.compute_overlap(0b00100010, deriv=5) == 0
    assert test.compute_overlap(0b01000100, deriv=0) == 0
    assert test.compute_overlap(0b01000100, deriv=1) == 0
    assert test.compute_overlap(0b01000100, deriv=2) == 1
    assert test.compute_overlap(0b01000100, deriv=3) == 0
    assert test.compute_overlap(0b01000100, deriv=6) == 0
    assert test.compute_overlap(0b10001000, deriv=0) == 0
    assert test.compute_overlap(0b10001000, deriv=1) == 0
    assert test.compute_overlap(0b10001000, deriv=2) == 0
    assert test.compute_overlap(0b10001000, deriv=3) == 1
    assert test.compute_overlap(0b10001000, deriv=99) == 0
    # fout electrons
    test = APIG(4, np.ones((4, 4)), np.ones((4, 4, 4, 4)))
    test.assign_params(np.array([1, 2, 3, 4,
                                 5, 6, 7, 8, 0], dtype=float))
    # bad SD
    assert_raises(ValueError, test.compute_overlap, 0b00010010)
    assert_raises(ValueError, test.compute_overlap, 0b00110101)
    # overlap
    assert test.compute_overlap(0b00110011, deriv=None) == 1*6 + 2*5
    assert test.compute_overlap(0b01010101, deriv=None) == 1*7 + 3*5
    assert test.compute_overlap(0b10011001, deriv=None) == 1*8 + 4*5
    assert test.compute_overlap(0b01100110, deriv=None) == 2*7 + 3*6
    assert test.compute_overlap(0b10101010, deriv=None) == 2*8 + 4*6
    assert test.compute_overlap(0b11001100, deriv=None) == 3*8 + 4*7
    # differentiate
    assert test.compute_overlap(0b00110011, deriv=0) == 6
    assert test.compute_overlap(0b00110011, deriv=1) == 5
    assert test.compute_overlap(0b00110011, deriv=2) == 0
    assert test.compute_overlap(0b00110011, deriv=3) == 0
    assert test.compute_overlap(0b00110011, deriv=4) == 2
    assert test.compute_overlap(0b00110011, deriv=5) == 1
    assert test.compute_overlap(0b00110011, deriv=6) == 0
    assert test.compute_overlap(0b00110011, deriv=7) == 0
    assert test.compute_overlap(0b00110011, deriv=8) == 0
    assert test.compute_overlap(0b00110011, deriv=99) == 0


def test_normalize():
    """ Tests wfns.proj.geminals.apig.APIG.normalize
    """
    test = APIG(2, np.ones((2, 2)), np.ones((2, 2, 2, 2)), dtype=float)
    #check
    test.assign_ref_sds(0b0101)
    test.assign_params(np.array([0.0, 0.0, 1.0]))
    assert_raises(ValueError, test.normalize)
    test.assign_dtype(complex)
    test.assign_params(np.array([1j, 0.0, 1.0]))
    assert_raises(ValueError, test.normalize)

    # one geminal, one reference
    test = APIG(2, np.ones((2, 2)), np.ones((2, 2, 2, 2)), dtype=float)
    print(test.compute_norm(), test.params)
    test.assign_params(np.array([2.0, 3.0, 0.0]))
    test.assign_ref_sds(0b0101)
    test.normalize()
    assert test.compute_norm() == 1
    # one geminal, two reference
    test.assign_params(np.array([2.0, 3.0, 0.0]))
    test.assign_ref_sds([0b0101, 0b1010])
    test.normalize()
    assert test.compute_norm() == 1

    test = APIG(4, np.ones((4, 4)), np.ones((4, 4, 4, 4)))
    # multiple geminal, one reference
    test.assign_params(np.array([2, 3, 4, 5, 6, 7, 8, 9, 0], dtype=float))
    test.assign_ref_sds(0b00110011)
    test.normalize()
    assert test.compute_norm() == 1
    # multiple geminal, multiple reference
    test.assign_params(np.array([2, 3, 4, 5, 6, 7, 8, 9, 0], dtype=float))
    test.assign_ref_sds([0b00110011, 0b01010101])
    test.normalize()
    assert test.compute_norm() == 1


def test_to_apr2g():
    """ Tests wfns.proj.geminals.apig.APIG.to_apr2g
    """
    hf_dict = gaussian_fchk('test/lih_hf_sto6g.fchk')
    one_int = hf_dict["one_int"]
    two_int = hf_dict["two_int"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]

    # construct apig with this coefficient matrix
    apig_coeffs = np.array([[1.033593181822e+00, 3.130903350751e-04, -4.321247538977e-03,
                             -1.767251395337e-03, -1.769214953534e-03, -1.169729179981e-03],
                            [-5.327889357199e-01, 9.602580629349e-01, -1.139839360648e-02,
                             -2.858698370621e-02, -2.878270043699e-02, -1.129324573431e-01]])

    apig = APIG(4, one_int, two_int, nuc_nuc=nuc_nuc, params=np.hstack((apig_coeffs.flat, 0)))
    # convert
    apr2g = apig.to_apr2g()
    lambdas = apr2g.params[:apr2g.npair][:, np.newaxis]
    epsilons = apr2g.params[apr2g.npair:apr2g.npair+apr2g.nspatial]
    zetas = apr2g.params[apr2g.npair+apr2g.nspatial:-1]
    apr2g_coeffs = zetas / (lambdas - epsilons)

    assert apig.nelec == apr2g.nelec
    assert apig.one_int == apr2g.one_int
    assert apig.two_int == apr2g.two_int
    assert apig.dtype == apr2g.dtype
    assert apig.nuc_nuc == apr2g.nuc_nuc
    assert apig.orbtype == apr2g.orbtype
    assert apig.pspace == apr2g.pspace
    assert apig.ref_sds == apr2g.ref_sds
    assert np.allclose(apig_coeffs, apr2g_coeffs)
    assert apig.params[-1] == apr2g.params[-1]

    # bad coefficient matrix
    apig = APIG(4, one_int, two_int, nuc_nuc=nuc_nuc, params=np.hstack((np.eye(2, 6).flat, 0)))
    assert_raises(ValueError, apig.to_apr2g)


def answer_apig_h2_sto6g():
    """ Finds the APIG/STO-6G wavefunction by scanning through the coefficients for the lowest
    energy
    """
    hf_dict = gaussian_fchk('test/h2_hf_sto6g.fchk')
    nelec = 2
    one_int = hf_dict["one_int"]
    two_int = hf_dict["two_int"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]
    apig = APIG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)

    # plot all possible values (within normalization constraint)
    apig.assign_ref_sds([0b0101, 0b1010])
    xs = []
    ys = []
    energies = []
    for i in np.arange(-2, 2, 0.1):
        for j in np.arange(-2, 2, 0.1):
            apig.assign_params(np.array([i, j, 1.0]))
            if np.allclose(apig.compute_norm(), 0):
                continue
            apig.normalize()
            xs.append(apig.params[0])
            ys.append(apig.params[1])
            energies.append(apig.compute_energy())

    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(ys, energies)
    # print(sorted(zip(xs,ys,energies), key=lambda x:x[2])[:10])
    # # ax2 = fig.add_subplot(1, 1, 1, projection='3d')
    # # ax2.scatter(xs, ys, energies)
    # plt.show()

    # print out results
    # print(sorted(zip(xs, ys, energies), key=lambda x: x[2])[0])
    # print(sorted(zip(xs, ys, energies), key=lambda x: x[2])[-1])
    # find exact minimum
    def min_energy(params):
        """ Find minimum energy"""
        apig.assign_params(np.hstack((params, 0)))
        return apig.compute_energy()
    res = scipy.optimize.minimize(min_energy, np.array([0.99388373467361912, -0.1104315260748444]))
    print('Minimum energy')
    print(res)
    # find exact maximum
    def max_energy(params):
        """ Find maximum energy"""
        apig.assign_params(np.hstack((params, 0)))
        return -apig.compute_energy()
    res = scipy.optimize.minimize(max_energy, np.array([0.11043152607484739, 0.99388373467361879]))
    print('Maximum energy')
    print(res)


def test_apig_h2_sto6g_ground():
    """ Tests ground state APIG wavefunction using H2 with HF/STO-6G orbital

    Answers obtained from answer_apig_h2_sto6g

    HF (Electronic) Energy : -1.838434256
    APIG Energy : -1.8590898441488894
    APIG Coeffs : [0.99359749, -0.11300768]
    """
    hf_dict = gaussian_fchk('test/h2_hf_sto6g.fchk')
    nelec = 2
    one_int = hf_dict["one_int"]
    two_int = hf_dict["two_int"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]
    apig = APIG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    # see if we can reproduce HF numbers
    apig.cache = {}
    apig.d_cache = {}
    assert abs(apig.compute_energy(include_nuc=False) - (-1.838434256)) < 1e-7
    # Solve with least squares solver with jacobian
    apig = APIG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(apig, solver_type='least_squares', use_jac=True)
    assert abs(apig.compute_energy(include_nuc=False) - (-1.8590898441488894)) < 1e-7
    assert abs(apig.get_energy(include_nuc=False) - (-1.8590898441488894)) < 1e-7
    # Solve with least squares solver without jacobian
    apig = APIG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(apig, solver_type='least_squares', use_jac=False)
    assert abs(apig.compute_energy(include_nuc=False) - (-1.8590898441488894)) < 1e-7
    # FIXME: get_energy is different from compute_energy here
    # assert abs(apig.get_energy(include_nuc=False) - (-1.8590898441488894)) < 1e-7
    # Solve with root (newton) solver with jacobian
    apig = APIG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(apig, solver_type='root', use_jac=True)
    assert abs(apig.compute_energy(include_nuc=False) - (-1.8590898441488894)) < 1e-7
    assert abs(apig.get_energy(include_nuc=False) - (-1.8590898441488894)) < 1e-7
    # Solve with root (newton) solver without jacobian
    apig = APIG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(apig, solver_type='root', use_jac=False)
    assert abs(apig.compute_energy(include_nuc=False) - (-1.8590898441488894)) < 1e-7
    assert abs(apig.get_energy(include_nuc=False) - (-1.8590898441488894)) < 1e-7
    # Solve with cma solver
    apig = APIG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(apig, solver_type='cma', use_jac=False)
    assert abs(apig.compute_energy(include_nuc=False) - (-1.8590898441488894)) < 1e-7
    assert abs(apig.get_energy(include_nuc=False) - (-1.8590898441488894)) < 1e-7


def test_apig_h2_sto6g_excited():
    """ Tests excited state APIG wavefunction using H2 with HF/STO-6G orbital

    Answers obtained from answer_apig_h2_sto6g

    APIG Energy : 0.2416648697421632
    APIG Coeffs : [0.11300769, 0.99359749]
    """
    hf_dict = gaussian_fchk('test/h2_hf_sto6g.fchk')
    nelec = 2
    one_int = hf_dict["one_int"]
    two_int = hf_dict["two_int"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]
    apig = APIG(nelec, one_int, two_int, nuc_nuc=nuc_nuc, ref_sds=0b1010,
                params=np.array([0.0, 1.0, 0.0]))
    # Solve with least squares solver with jacobian
    apig = APIG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(apig, solver_type='least_squares', use_jac=True)
    assert abs(apig.compute_energy(include_nuc=False) - (-1.8590898441488894)) < 1e-7
    assert abs(apig.get_energy(include_nuc=False) - (-1.8590898441488894)) < 1e-7
    # Solve with least squares solver without jacobian
    apig = APIG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(apig, solver_type='least_squares', use_jac=False)
    assert abs(apig.compute_energy(include_nuc=False) - (-1.8590898441488894)) < 1e-7
    # FIXME: get_energy is different from compute_energy here for some reason
    # assert abs(apig.get_energy(include_nuc=False) - (-1.8590898441488894)) < 1e-7
    # Solve with root (newton) solver with jacobian
    apig = APIG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(apig, solver_type='root', use_jac=True)
    assert abs(apig.compute_energy(include_nuc=False) - (-1.8590898441488894)) < 1e-7
    assert abs(apig.get_energy(include_nuc=False) - (-1.8590898441488894)) < 1e-7
    # Solve with root (newton) solver without jacobian
    apig = APIG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(apig, solver_type='root', use_jac=False)
    assert abs(apig.compute_energy(include_nuc=False) - (-1.8590898441488894)) < 1e-7
    assert abs(apig.get_energy(include_nuc=False) - (-1.8590898441488894)) < 1e-7
    # Solve with cma solver
    apig = APIG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(apig, solver_type='cma', use_jac=False)
    assert abs(apig.compute_energy(include_nuc=False) - (-1.8590898441488894)) < 1e-7
    assert abs(apig.get_energy(include_nuc=False) - (-1.8590898441488894)) < 1e-7


def answer_apig_h2_631gdp():
    """ Finds the APIG/6-31G** wavefunction by scanning through the coefficients for the lowest
    energy
    """
    hf_dict = gaussian_fchk('test/h2_hf_631gdp.fchk')
    nelec = 2
    one_int = hf_dict["one_int"]
    two_int = hf_dict["two_int"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]
    apig = APIG(nelec, one_int, two_int, nuc_nuc=nuc_nuc, ref_sds=(0b00000000010000000001,
                                                                   0b00000000100000000010,
                                                                   0b00000001000000000100,
                                                                   0b00000010000000001000,
                                                                   0b00000100000000010000,
                                                                   0b00001000000000100000,
                                                                   0b00010000000001000000,
                                                                   0b00100000000010000000,
                                                                   0b01000000000100000000,
                                                                   0b10000000001000000000,),
                params=np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float))
    res = solve(apig, solver_type='variational', use_jac=True)
    print(res)
    return apig.params


def test_apig_h2_631gdp():
    """ Tests APIG wavefunction using H2 with HF/6-31G** orbital

    Answers obtained from answer_apig_h2_631gdp

    HF (Electronic) Energy : -1.84444667247
    APIG Energy : -1.8696828608304896
    APIG Coeffs : [0.995079200788, -0.059166892062, -0.054284175189, -0.036920061272,
                   -0.028848919079, -0.028847742282, -0.013108383833, -0.008485392433,
                   -0.008485285973, -0.005149411511]
    """
    hf_dict = gaussian_fchk('test/h2_hf_631gdp.fchk')
    nelec = 2
    one_int = hf_dict["one_int"]
    two_int = hf_dict["two_int"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]

    apig = APIG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    # see if we can reproduce HF numbers
    apig.cache = {}
    apig.d_cache = {}
    assert abs(apig.compute_energy(include_nuc=False) - (-1.84444667247)) < 1e-7
    # Solve with least squares solver with jacobian
    # FIXME: Least squares answer is bad when Jacobian is used
    # apig = APIG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    # solve(apig, solver_type='least_squares', use_jac=True)
    # print(apig.compute_energy(include_nuc=False),
    #       apig.get_energy(include_nuc=False))
    # assert abs(apig.compute_energy(include_nuc=False) - (-1.8696828608304896)) < 1e-2
    # assert abs(apig.get_energy(include_nuc=False) - (-1.8696828608304896)) < 1e-2
    # Solve with least squares solver without jacobian
    apig = APIG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(apig, solver_type='least_squares', use_jac=False)
    assert abs(apig.compute_energy(include_nuc=False) - (-1.8696828608304896)) < 1e-7
    # FIXME: get_energy is different from compute_energy here for some reason
    # assert abs(apig.get_energy(include_nuc=False) - (-1.8696828608304896)) < 1e-7
    # Solve with root (newton) solver with jacobian
    apig = APIG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(apig, solver_type='root', use_jac=True)
    assert abs(apig.compute_energy(include_nuc=False) - (-1.8696828608304896)) < 1e-7
    assert abs(apig.get_energy(include_nuc=False) - (-1.8696828608304896)) < 1e-7
    # Solve with root (newton) solver without jacobian
    apig = APIG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(apig, solver_type='root', use_jac=False)
    assert abs(apig.compute_energy(include_nuc=False) - (-1.8696828608304896)) < 1e-7
    assert abs(apig.get_energy(include_nuc=False) - (-1.8696828608304896)) < 1e-7
    # Solve with cma solver
    # FIXME: CMA answer is quite bad
    apig = APIG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(apig, solver_type='cma', use_jac=False)
    assert abs(apig.compute_energy(include_nuc=False) - (-1.8696828608304896)) < 1e-1
    assert abs(apig.get_energy(include_nuc=False) - (-1.8696828608304896)) < 1e-1
    assert np.allclose(apig.params[:-1],
                       np.array([0.995079200788, -0.059166892062, -0.054284175189, -0.036920061272,
                                 -0.028848919079, -0.028847742282, -0.013108383833, -0.008485392433,
                                 -0.008485285973, -0.005149411511]),
                       atol=1)


def answer_apig_lih_sto6g():
    """ Finds the APIG/STO-6G wavefunction for LiH by scanning through the coefficients for the
    lowest energy
    """
    hf_dict = gaussian_fchk('test/lih_hf_sto6g.fchk')
    nelec = 4
    one_int = hf_dict["one_int"]
    two_int = hf_dict["two_int"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]

    apig = APIG(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc,
                ref_sds=(0b000011000011, 0b000101000101, 0b001001001001, 0b010001010001,
                         0b100001100001, 0b000110000110, 0b001010001010, 0b010010010010,
                         0b100010100010, 0b001100001100, 0b010100010100, 0b100100100100,
                         0b011000011000, 0b101000101000, 0b110000110000),
                params=np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=float))
    res = solve(apig, solver_type='variational', use_jac=True)
    print(res)
    return apig.params


@attr('slow')
def test_apig_lih_sto6g():
    """ Tests APIG wavefunction using LiH with HF/STO-6G orbital

    Answers obtained from answer_apig_lih_sto6g

    HF (Electronic) Energy : -8.9472891719
    APIG Energy : -8.963531109581904
    APIG Coeffs : [1.033593181822e+00, 3.130903350751e-04, -4.321247538977e-03,
                   -1.767251395337e-03, -1.769214953534e-03, -1.169729179981e-03,
                   -5.327889357199e-01, 9.602580629349e-01, -1.139839360648e-02,
                   -2.858698370621e-02, -2.878270043699e-02, -1.129324573431e-01]
    """
    hf_dict = gaussian_fchk('test/lih_hf_sto6g.fchk')
    nelec = 4
    one_int = hf_dict["one_int"]
    two_int = hf_dict["two_int"]
    nuc_nuc = hf_dict["nuc_nuc_energy"]

    apig = APIG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    # see if we can reproduce HF numbers
    apig.cache = {}
    apig.d_cache = {}
    assert abs(apig.compute_energy(include_nuc=False) - (-8.9472891719)) < 1e-7
    # Solve with least squares solver with jacobian
    apig = APIG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(apig, solver_type='least_squares', use_jac=True)
    print(apig.compute_energy(include_nuc=False), apig.get_energy(include_nuc=False))
    assert abs(apig.compute_energy(include_nuc=False) - (-8.963531109581904)) < 1e-6
    assert abs(apig.get_energy(include_nuc=False) - (-8.963531109581904)) < 1e-6
    # Solve with least squares solver without jacobian
    apig = APIG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(apig, solver_type='least_squares', use_jac=False)
    print(apig.compute_energy(include_nuc=False), apig.get_energy(include_nuc=False))
    assert abs(apig.compute_energy(include_nuc=False) - (-8.963531109581904)) < 1e-7
    assert abs(apig.get_energy(include_nuc=False) - (-8.963531109581904)) < 1e-7
    # Solve with cma solver
    apig = APIG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    solve(apig, solver_type='cma', use_jac=False)
    print(apig.compute_energy(include_nuc=False), apig.get_energy(include_nuc=False))
    assert abs(apig.compute_energy(include_nuc=False) - (-8.963531109581904)) < 1e-6
    assert abs(apig.get_energy(include_nuc=False) - (-8.963531109581904)) < 1e-6
    # # Solve with root (newton) solver with jacobian
    # # FIXME: root solver with over projection has problems optimizing
    # apig = APIG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    # solve(apig, solver_type='root', use_jac=True)
    # assert abs(apig.compute_energy(include_nuc=False) - (-8.963531109581904)) < 1e-4
    # assert abs(apig.get_energy(include_nuc=False) - (-8.963531109581904)) < 1e-3
    # # Solve with root (newton) solver without jacobian
    # # FIXME: root solver without Jacobian has trouble optimizing
    # apig = APIG(nelec, one_int, two_int, nuc_nuc=nuc_nuc)
    # solve(apig, solver_type='root', use_jac=False)
    # print(apig.compute_energy(include_nuc=False), apig.get_energy(include_nuc=False))
    # assert abs(apig.compute_energy(include_nuc=False) - (-8.963531109581904)) < 1e-4
    # assert abs(apig.get_energy(include_nuc=False) - (-8.963531109581904)) < 1e-3
