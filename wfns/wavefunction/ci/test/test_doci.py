""" Tests wfns.wavefunction.doci
"""
from __future__ import absolute_import, division, print_function
from nose.tools import assert_raises
import numpy as np
# from wfns.solver.solver_ci import solve
from wfns.wavefunction.ci.doci import DOCI
from wfns.tools import find_datafile


class TestDOCI(DOCI):
    """DOCI instance that skips initialization."""
    def __init__(self):
        pass


def test_assign_nelec():
    """Tests DOCI.assign_nelec."""
    test = TestDOCI()
    # int
    test.assign_nelec(2)
    assert test.nelec == 2
    # check errors
    assert_raises(TypeError, test.assign_nelec, None)
    assert_raises(TypeError, test.assign_nelec, 2.0)
    assert_raises(TypeError, test.assign_nelec, '2')
    assert_raises(ValueError, test.assign_nelec, 0)
    assert_raises(ValueError, test.assign_nelec, -2)
    assert_raises(ValueError, test.assign_nelec, 1)
    assert_raises(ValueError, test.assign_nelec, 3)


def test_assign_spin():
    """Test DOCI.assign_spin."""
    test = TestDOCI()
    test.assign_spin()
    assert test.spin == 0
    test.assign_spin(0)
    assert test.spin == 0
    assert_raises(ValueError, test.assign_spin, 0.5)
    assert_raises(ValueError, test.assign_spin, 1)
    assert_raises(ValueError, test.assign_spin, True)


def test_assign_seniority():
    """Test DOCI.ssign_seniority."""
    test = TestDOCI()
    test.assign_seniority()
    assert test.seniority == 0
    test.assign_seniority(0)
    assert test.seniority == 0
    assert_raises(ValueError, test.assign_seniority, 1)
    assert_raises(ValueError, test.assign_seniority, True)


# FIXME: implement after solver is implemented
def test_doci_h4_hf_sto6g():
    """ Tests DOCI wavefunction for H4 (square) (STO-6G) agaisnt Peter's orbital optimized DOCI

    NOTE
    ----
    Optimized orbitals are read in from Peter's code
    """
    #### H2 ####
    # HF energy: -1.13126983927
    # OO DOCI energy: -1.884948574812363
    nelec = 4

    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/h4_square_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile('test/h4_square_hf_sto6g_oneint.npy'))
    two_int = np.load(find_datafile('test/h4_square_hf_sto6g_twoint.npy'))
    nuc_nuc = 2.70710678119

    # compare HF numbers
    doci = DOCI(nelec=nelec, one_int=(one_int,), two_int=(two_int,), nuc_nuc=nuc_nuc)
    assert abs(doci.compute_ci_matrix()[0, 0] + doci.nuc_nuc - (-1.131269841877) < 1e-8)

    orb_rot = np.array([[0.707106752870, -0.000004484084, 0.000006172115, -0.707106809462],
                        [0.707106809472, -0.000004868924, -0.000006704609, 0.707106752852],
                        [0.000004942751, 0.707106849959, 0.707106712365, 0.000006630781],
                        [0.000004410256, 0.707106712383, -0.707106849949, -0.000006245943]])
    one_int = orb_rot.T.dot(one_int).dot(orb_rot)
    two_int = np.einsum('ijkl,ia->ajkl', two_int, orb_rot)
    two_int = np.einsum('ajkl,jb->abkl', two_int, orb_rot)
    two_int = np.einsum('abkl,kc->abcl', two_int, orb_rot)
    two_int = np.einsum('abcl,ld->abcd', two_int, orb_rot)

    doci = DOCI(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc)
    solve(doci)
    assert abs(doci.get_energy() - (-1.884948574812363)) < 1e-7


# FIXME: NEED EITHER ORBITAL OPTIMIZATION OR REORDERING OF PETER'S ORBITALS
# def test_doci_h2_hf_631gdp():
#     """ Tests DOCI wavefunction for H2 (STO-6G) against Peter's orbital optimized DOCI

#     NOTE
#     ----
#     Optimized orbitals are read in from Peter's code
#     """
#     hf_dict = gaussian_fchk('test/h2_hf_631gdp.fchk')

#     nelec = 2
#     one_int = hf_dict["one_int"][0]
#     two_int = hf_dict["two_int"][0]
#     nuc_nuc = hf_dict["nuc_nuc_energy"]

#     # compare HF numbers
#     doci = DOCI(nelec=nelec, one_int=(one_int,), two_int=(two_int,), nuc_nuc=nuc_nuc)
#     assert abs(doci.compute_ci_matrix()[0, 0] + doci.nuc_nuc - (-1.131269841877) < 1e-8)
#     solve(doci)
#     print(doci.get_energy(include_nuc=True))

#     # transform orbitals
#     # Peter's orbitals are reordered
#     orb_rot = np.array([[0.488169047143, -0.543185538076, -0.411601097620, -0.000000403150, 0.000001613245, -0.388739227860, 0.303171079258, 0.000006888161, -0.000003111236, 0.232796259502],
#                         [0.503298667825, -0.293907169278, 0.461587797414, -0.000000054879, -0.000002339911, 0.579287242867, -0.182498320074, -0.000009461656, 0.000004256464, 0.279939967492],
#                         [0.000000002376, -0.000000458104, -0.000000245684, 0.707091981947, -0.003178725269, 0.000005319524, -0.000000256975, -0.001618581793, -0.707112582929, -0.000000446558],
#                         [-0.000000013848, 0.000000722493, -0.000001875001, 0.003178673760, 0.707091982001, -0.000007897018, -0.000000481474, -0.707112583195, 0.001618530225, 0.000001889435],
#                         [0.091549505235, 0.344332883161, -0.342816419640, -0.000000734051, -0.000000876752, -0.117351069059, -0.611827958397, 0.000003721555, -0.000001885537, 0.606156524329],
#                         [0.488169105550, 0.543185764514, -0.411602641767, 0.000000325673, -0.000000886083, 0.387810769711, 0.304363736938, -0.000004400987, 0.000003086321, -0.232785174377],
#                         [0.503298863137, 0.293908060439, 0.461584812911, 0.000000136369, 0.000001903559, -0.578727232405, -0.184262403760, 0.000006801308, -0.000004346795, -0.279946287399],
#                         [0.000000009714, 0.000000469899, 0.000000085206, 0.707107221036, -0.003194171012, -0.000005808918, 0.000000207363, 0.001603067038, 0.707097309689, 0.000001905864],
#                         [-0.000000002638, -0.000000370181, 0.000001927471, 0.003194222521, 0.707107220974, 0.000012676707, -0.000001105440, 0.707097309311, -0.001603118441, -0.000000680451],
#                         [-0.091549435933, 0.344340224027, 0.342813170568, -0.000000606375, 0.000000474065, -0.119224270382, 0.611439523904, 0.000002445211, -0.000002458517, 0.606180602568]])

#     one_int = orb_rot.T.dot(one_int).dot(orb_rot)
#     two_int = np.einsum('ijkl,ia->ajkl', two_int, orb_rot)
#     two_int = np.einsum('ajkl,jb->abkl', two_int, orb_rot)
#     two_int = np.einsum('abkl,kc->abcl', two_int, orb_rot)
#     two_int = np.einsum('abcl,ld->abcd', two_int, orb_rot)

#     doci = DOCI(nelec=nelec, one_int=one_int, two_int=two_int, nuc_nuc=nuc_nuc)
#     solve(doci)
#     print(doci.get_energy(), doci.nuc_nuc)
#     assert abs(doci.get_energy() - (-1.165148755001395)) < 1e-7
