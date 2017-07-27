""" Tests wfns.proj.geminals.geminal
"""
from __future__ import absolute_import, division, print_function
from nose.tools import assert_raises
import numpy as np
from wfns.proj.geminals.geminal import Geminal

class TestGeminal(Geminal):
    """ Child of Geminal used to test Geminal wavefunction

    Because Geminal is an abstract class
    """
    def __init__(self, nelec, one_int, two_int, dtype=None, nuc_nuc=None, orbtype=None, pspace=None,
                 ref_sds=None):
        super(self.__class__, self).__init__(nelec, one_int, two_int, dtype=None, nuc_nuc=None,
                                             orbtype=None, pspace=None, ref_sds=None, params=None)

    @property
    def template_coeffs(self):
        return np.arange(4, dtype=float)

    @property
    def template_orbpairs(self):
        return ((0, 1), )

    def compute_overlap(self, sd, deriv=None):
        if deriv is None:
            if sd == 0b0101:
                return 5
            elif sd == 0b1010:
                return 6
            return 7
        else:
            return 8

    def normalize(self):
        pass


def test_assign_nelec():
    """ Tests Geminal.assign_nelec
    """
    test = TestGeminal(4, np.ones((4, 4)), np.ones((4, 4, 4, 4)))
    # bad input
    assert_raises(TypeError, test.assign_nelec, 2.0)
    assert_raises(ValueError, test.assign_nelec, -2)
    assert_raises(ValueError, test.assign_nelec, 5)
    # int
    test.assign_nelec(4)
    assert test.nelec == 4
    # long
    test.assign_nelec(long(8))
    assert test.nelec == 8


def test_npair():
    """ Tests Geminal.npair
    """
    test = TestGeminal(4, np.ones((4, 4)), np.ones((4, 4, 4, 4)))
    assert test.npair == 2


def test_assign_ngem():
    """ Tests Geminal.assign_ngem
    """
    test = TestGeminal(4, np.ones((4, 4)), np.ones((4, 4, 4, 4)))
    # check error
    assert_raises(TypeError, test.assign_ngem, 4.0)
    assert_raises(ValueError, test.assign_ngem, 1)
    # None
    test.assign_ngem(None)
    assert test.ngem == 2
    # int
    test.assign_ngem(8)
    assert test.ngem == 8
    # long
    test.assign_ngem(long(8))
    assert test.ngem == 8


def test_assign_orbpair():
    """ Tests Geminal.assign_orbpair
    """
    test = TestGeminal(4, np.ones((4, 4)), np.ones((4, 4, 4, 4)))
    # check errors
    assert_raises(TypeError, test.assign_orbpairs, {1:0, 2:0})
    assert_raises(TypeError, test.assign_orbpairs, ((0, 1), [1, 2]))
    assert_raises(TypeError, test.assign_orbpairs, ((0, 1), (1, 2, 3)))
    assert_raises(ValueError, test.assign_orbpairs, ((0, 1), (1, 2.0)))
    assert_raises(ValueError, test.assign_orbpairs, ((0, 1), (1, 1)))
    # None
    test.assign_orbpairs()
    assert test.dict_orbpair_ind == {(0, 1):0}
    test.assign_orbpairs(None)
    assert test.dict_orbpair_ind == {(0, 1):0}
    # sorted
    test.assign_orbpairs(((0, 1), (1, 2), (3, 4)))
    assert test.dict_orbpair_ind == {(0, 1):0, (1, 2):1, (3, 4):2}
    # unsorted
    test.assign_orbpairs(((0, 1), (2, 1), (3, 4)))
    assert test.dict_orbpair_ind == {(0, 1):0, (1, 2):1, (3, 4):2}
    test.assign_orbpairs(((3, 4), (1, 2), (1, 0)))
    assert test.dict_orbpair_ind == {(0, 1):0, (1, 2):1, (3, 4):2}
    test.assign_orbpairs(((3, 4), (1, 2), (1, 0)))
    assert test.dict_orbpair_ind == {(0, 1):0, (1, 2):1, (3, 4):2}
    test.assign_orbpairs(((0, 1), (2, 4), (2, 3)))
    assert test.dict_orbpair_ind == {(0, 1):0, (2, 3):1, (2, 4):2}


def test_abstract():
    """ Check for abstract methods and properties
    """
    # both template_coeffs and compute_overlap not defined
    assert_raises(TypeError, lambda: Geminal(2, np.ones((3, 3)), np.ones((3, 3, 3, 3))))

    # template_coeffs not defined
    class TestWfn(Geminal):
        def compute_overlap(self):
            pass

        @property
        def template_orbpairs(self):
            pass
    assert_raises(TypeError, lambda: TestWfn(2, np.ones((3, 3)), np.ones((3, 3, 3, 3))))

    # compute_overlap not defined
    class TestWfn(Geminal):
        @property
        def template_coeffs(self):
            pass

        @property
        def template_orbpairs(self):
            pass
    assert_raises(TypeError, lambda: TestWfn(2, np.ones((3, 3)), np.ones((3, 3, 3, 3))))

    # template_orbpairs not defined
    class TestWfn(Geminal):
        @property
        def template_coeffs(self):
            pass

        def compute_overlap(self):
            pass
    assert_raises(TypeError, lambda: TestWfn(2, np.ones((3, 3)), np.ones((3, 3, 3, 3))))
