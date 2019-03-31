"""Test wfns.ham.unrestricted_base."""
import itertools as it
import numpy as np
from nose.tools import assert_raises
from wfns.ham.unrestricted_base import BaseUnrestrictedHamiltonian


class Empty:
    """Empty container class."""
    pass


class TestBaseUnrestrictedHamiltonian(BaseUnrestrictedHamiltonian):
    """BaseUnrestrictedHamiltonian class that bypasses the abstract methods."""
    def integrate_wfn_sd(self, wfn, sd, deriv=None):
        """Abstract method."""
        pass

    def integrate_sd_sd(self, sd1, sd2, deriv=None):
        """Abstract method."""
        pass


def test_assign_integrals():
    """Test BaseGeneralizedHamiltonian.assign_integrals."""
    # good input
    one_int = np.random.rand(4, 4)
    two_int = np.random.rand(4, 4, 4, 4)

    test = Empty()
    BaseUnrestrictedHamiltonian.assign_integrals(test, [one_int]*2, [two_int]*3)
    assert np.allclose(test.one_int, one_int)
    assert np.allclose(test.two_int, two_int)

    # bad input
    test = Empty()

    assert_raises(TypeError, BaseUnrestrictedHamiltonian.assign_integrals, test,
                  one_int, [two_int]*3)
    assert_raises(TypeError, BaseUnrestrictedHamiltonian.assign_integrals, test,
                  [one_int], [two_int]*3)
    assert_raises(TypeError, BaseUnrestrictedHamiltonian.assign_integrals, test,
                  [one_int, one_int.astype(int)]*2, [two_int]*3)
    assert_raises(TypeError, BaseUnrestrictedHamiltonian.assign_integrals, test,
                  [one_int.tolist(), one_int]*2, [two_int]*3)

    assert_raises(TypeError, BaseUnrestrictedHamiltonian.assign_integrals, test,
                  [one_int]*2, two_int)
    assert_raises(TypeError, BaseUnrestrictedHamiltonian.assign_integrals, test,
                  [one_int]*2, [two_int])
    assert_raises(TypeError, BaseUnrestrictedHamiltonian.assign_integrals, test,
                  [one_int]*2, [two_int, two_int, two_int.astype(int)])
    assert_raises(TypeError, BaseUnrestrictedHamiltonian.assign_integrals, test,
                  [one_int]*2, [two_int, two_int, two_int.tolist()])

    assert_raises(TypeError, BaseUnrestrictedHamiltonian.assign_integrals, test,
                  [one_int, one_int], [two_int, two_int, two_int.astype(complex)])
    assert_raises(TypeError, BaseUnrestrictedHamiltonian.assign_integrals, test,
                  [one_int, one_int.astype(complex)], [two_int, two_int, two_int])

    assert_raises(ValueError, BaseUnrestrictedHamiltonian.assign_integrals, test,
                  [one_int, one_int.reshape(1, 4, 4)], [two_int]*3)
    assert_raises(ValueError, BaseUnrestrictedHamiltonian.assign_integrals, test,
                  [one_int, one_int[:, :3]], [two_int]*3)

    assert_raises(ValueError, BaseUnrestrictedHamiltonian.assign_integrals, test,
                  [one_int]*2, [two_int, two_int, two_int.reshape(1, 4, 4, 4, 4)])
    assert_raises(ValueError, BaseUnrestrictedHamiltonian.assign_integrals, test,
                  [one_int]*2, [two_int, two_int, two_int[:, :, :, :3]])

    assert_raises(ValueError, BaseUnrestrictedHamiltonian.assign_integrals, test,
                  [one_int[:3, :3], one_int], [two_int]*3)

    assert_raises(ValueError, BaseUnrestrictedHamiltonian.assign_integrals, test,
                  [one_int]*2, [two_int, two_int, two_int[:3, :3, :3, :3]])

    assert_raises(ValueError, BaseUnrestrictedHamiltonian.assign_integrals, test,
                  [one_int]*2, [two_int[:3, :3, :3, :3]]*3)


def test_nspin():
    """Test BaseUnrestrictedHamiltonian.nspin."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    test = TestBaseUnrestrictedHamiltonian(2*[one_int], 3*[two_int])
    assert test.nspin == 4


def test_dtype():
    """Test BaseUnrestrictedHamiltonian.dtype."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    test = TestBaseUnrestrictedHamiltonian(2*[one_int], 3*[two_int])
    assert test.dtype == float


def test_orb_rotate_jacobi():
    """Test BaseUnrestrictedHamiltonian.orb_rotate_jacobi."""
    one_int_alpha = np.arange(16, dtype=float).reshape(4, 4)
    one_int_beta = np.arange(16, 32, dtype=float).reshape(4, 4)
    two_int_aaaa = np.arange(256, dtype=float).reshape(4, 4, 4, 4)
    two_int_abab = np.arange(256, 512, dtype=float).reshape(4, 4, 4, 4)
    two_int_bbbb = np.arange(512, 768, dtype=float).reshape(4, 4, 4, 4)

    theta = 2 * np.pi * (np.random.random() - 0.5)

    ham = TestBaseUnrestrictedHamiltonian([one_int_alpha, one_int_beta],
                                          [two_int_aaaa, two_int_abab, two_int_bbbb])

    assert_raises(TypeError, ham.orb_rotate_jacobi, {0, 1}, 0.0)
    assert_raises(TypeError, ham.orb_rotate_jacobi, (0, 1, 2), 0.0)
    assert_raises(TypeError, ham.orb_rotate_jacobi, (0.0, 1), 0.0)
    assert_raises(TypeError, ham.orb_rotate_jacobi, (0, 1.0), 0.0)

    assert_raises(ValueError, ham.orb_rotate_jacobi, (0, 0), 0.0)

    assert_raises(ValueError, ham.orb_rotate_jacobi, (-1, 1), 0.0)
    assert_raises(ValueError, ham.orb_rotate_jacobi, (0, 4), 0.0)

    assert_raises(ValueError, ham.orb_rotate_jacobi, (0, 4), 0.0)
    assert_raises(ValueError, ham.orb_rotate_jacobi, (0, 4), 0.0)

    assert_raises(TypeError, ham.orb_rotate_jacobi, (0, 1), '0.0')
    assert_raises(TypeError, ham.orb_rotate_jacobi, (0, 1), np.array([0.0, 1.0]))

    for p, q in it.combinations(range(4), 2):
        jacobi_rotation = np.identity(4)
        jacobi_rotation[p, p] = np.cos(theta)
        jacobi_rotation[p, q] = np.sin(theta)
        jacobi_rotation[q, p] = -np.sin(theta)
        jacobi_rotation[q, q] = np.cos(theta)
        answer_alpha = np.einsum('ij,ia,jb->ab', one_int_alpha, jacobi_rotation, jacobi_rotation)
        answer_aaaa = np.einsum('ijkl,ia,jb,kc,ld->abcd', two_int_aaaa, *[jacobi_rotation]*4)
        answer_abab = np.einsum('ijkl,ia,kc->ajcl', two_int_abab, *[jacobi_rotation]*2)

        ham = TestBaseUnrestrictedHamiltonian([np.copy(one_int_alpha), np.copy(one_int_beta)],
                                              [np.copy(two_int_aaaa), np.copy(two_int_abab),
                                               np.copy(two_int_bbbb)])
        ham.orb_rotate_jacobi((p, q), theta)

        assert np.allclose(ham.one_int[0], answer_alpha)
        assert np.allclose(ham.one_int[1], one_int_beta)
        assert np.allclose(ham.two_int[0], answer_aaaa)
        assert np.allclose(ham.two_int[1], answer_abab)
        assert np.allclose(ham.two_int[2], two_int_bbbb)

        ham = TestBaseUnrestrictedHamiltonian([np.copy(one_int_alpha), np.copy(one_int_beta)],
                                              [np.copy(two_int_aaaa), np.copy(two_int_abab),
                                               np.copy(two_int_bbbb)])
        ham.orb_rotate_jacobi((q, p), theta)

        assert np.allclose(ham.one_int[0], answer_alpha)
        assert np.allclose(ham.one_int[1], one_int_beta)
        assert np.allclose(ham.two_int[0], answer_aaaa)
        assert np.allclose(ham.two_int[1], answer_abab)
        assert np.allclose(ham.two_int[2], two_int_bbbb)

    for p, q in it.combinations(range(4, 8), 2):
        jacobi_rotation = np.identity(4)
        jacobi_rotation[p-4, p-4] = np.cos(theta)
        jacobi_rotation[p-4, q-4] = np.sin(theta)
        jacobi_rotation[q-4, p-4] = -np.sin(theta)
        jacobi_rotation[q-4, q-4] = np.cos(theta)

        answer_beta = np.einsum('ij,ia,jb->ab', one_int_beta, jacobi_rotation, jacobi_rotation)
        answer_bbbb = np.einsum('ijkl,ia,jb,kc,ld->abcd', two_int_bbbb, *[jacobi_rotation]*4)
        answer_abab = np.einsum('ijkl,jb,ld->ibkd', two_int_abab, *[jacobi_rotation]*2)

        ham = TestBaseUnrestrictedHamiltonian([np.copy(one_int_alpha), np.copy(one_int_beta)],
                                              [np.copy(two_int_aaaa), np.copy(two_int_abab),
                                               np.copy(two_int_bbbb)])
        ham.orb_rotate_jacobi((p, q), theta)

        assert np.allclose(ham.one_int[0], one_int_alpha)
        assert np.allclose(ham.one_int[1], answer_beta)
        assert np.allclose(ham.two_int[0], two_int_aaaa)
        assert np.allclose(ham.two_int[1], answer_abab)
        assert np.allclose(ham.two_int[2], answer_bbbb)

        ham = TestBaseUnrestrictedHamiltonian([np.copy(one_int_alpha), np.copy(one_int_beta)],
                                              [np.copy(two_int_aaaa), np.copy(two_int_abab),
                                               np.copy(two_int_bbbb)])
        ham.orb_rotate_jacobi((q, p), theta)

        assert np.allclose(ham.one_int[0], one_int_alpha)
        assert np.allclose(ham.one_int[1], answer_beta)
        assert np.allclose(ham.two_int[0], two_int_aaaa)
        assert np.allclose(ham.two_int[1], answer_abab)
        assert np.allclose(ham.two_int[2], answer_bbbb)


def test_orb_rotate_matrix():
    """Test BaseUnrestrictedHamiltonian.orb_rotate_matrix."""
    one_int = np.arange(1, 17, dtype=float).reshape(4, 4)
    two_int = np.arange(1, 257, dtype=float).reshape(4, 4, 4, 4)

    random1 = np.random.rand(4, 4)
    transform1 = np.linalg.eigh(random1 + random1.T)[1]
    random2 = np.random.rand(4, 4)
    transform2 = np.linalg.eigh(random2 + random2.T)[1]

    one_orig = np.copy(one_int)
    two_orig = np.copy(two_int)

    ham = TestBaseUnrestrictedHamiltonian((one_int, one_int), (two_int, two_int, two_int))
    ham.orb_rotate_matrix(transform1)
    assert np.allclose(ham.one_int[0], np.einsum('ij,ia,jb->ab', one_orig, transform1, transform1))
    assert np.allclose(ham.one_int[1], np.einsum('ij,ia,jb->ab', one_orig, transform1, transform1))
    assert np.allclose(ham.two_int[0], np.einsum('ijkl,ia,jb,kc,ld->abcd', two_orig,
                                                 transform1, transform1, transform1, transform1))
    assert np.allclose(ham.two_int[1],  np.einsum('ijkl,ia,jb,kc,ld->abcd', two_orig,
                                                  transform1, transform1, transform1, transform1))
    assert np.allclose(ham.two_int[2],  np.einsum('ijkl,ia,jb,kc,ld->abcd', two_orig,
                                                  transform1, transform1, transform1, transform1))

    ham = TestBaseUnrestrictedHamiltonian((one_int, one_int), (two_int, two_int, two_int))
    ham.orb_rotate_matrix([transform1, transform2])
    assert np.allclose(ham.one_int[0], np.einsum('ij,ia,jb->ab', one_orig, transform1, transform1))
    assert np.allclose(ham.one_int[1], np.einsum('ij,ia,jb->ab', one_orig, transform2, transform2))
    assert np.allclose(ham.two_int[0], np.einsum('ijkl,ia,jb,kc,ld->abcd', two_orig,
                                                 transform1, transform1, transform1, transform1))
    assert np.allclose(ham.two_int[1],  np.einsum('ijkl,ia,jb,kc,ld->abcd', two_orig,
                                                  transform1, transform2, transform1, transform2))
    assert np.allclose(ham.two_int[2],  np.einsum('ijkl,ia,jb,kc,ld->abcd', two_orig,
                                                  transform2, transform2, transform2, transform2))

    assert_raises(TypeError, ham.orb_rotate_matrix, [np.random.rand(4, 4)])
    assert_raises(TypeError, ham.orb_rotate_matrix, [np.random.rand(4, 4).tolist(),
                                                     np.random.rand(4, 4)])
    assert_raises(TypeError, ham.orb_rotate_matrix, [np.random.rand(4, 4),
                                                     np.random.rand(4, 4).tolist()])

    assert_raises(ValueError, ham.orb_rotate_matrix, np.random.rand(4, 4, 4))

    assert_raises(ValueError, ham.orb_rotate_matrix, np.random.rand(3, 4))
