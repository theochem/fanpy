"""Test wfns.objective.constraints.norm."""
import itertools as it
from nose.tools import assert_raises
import numpy as np
from wfns.objective.constraints.norm import NormConstraint
from wfns.wfn.ci.ci_wavefunction import CIWavefunction
from wfns.param import ParamContainer


def test_norm_init():
    """Test NormConstraint.__init__."""
    assert_raises(TypeError, NormConstraint, 2)
    assert_raises(TypeError, NormConstraint, ParamContainer(4))
    wfn = CIWavefunction(2, 4)
    test = NormConstraint(wfn)
    assert test.wfn == wfn


def test_norm_num_eqns():
    """Test NormConstraint.num_eqns."""
    wfn = CIWavefunction(2, 4)
    test = NormConstraint(wfn)
    assert test.num_eqns == 1


def test_norm_objective():
    """Test NormConstraint.objective."""
    wfn = CIWavefunction(2, 4)
    test = NormConstraint(wfn, param_selection=[(wfn, np.arange(6))])

    guess = np.random.rand(6)
    # check assignment
    test.objective(guess)
    assert np.allclose(wfn.params, guess)

    sds = [0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010]
    # single sd
    for sd in sds:
        test.assign_refwfn(sd)
        olp = wfn.get_overlap(sd)
        assert np.allclose(test.objective(guess), olp**2 - 1)
    # multiple sd
    for sd1, sd2 in it.combinations(sds, 2):
        test.assign_refwfn([sd1, sd2])
        olp1 = wfn.get_overlap(sd1)
        olp2 = wfn.get_overlap(sd2)
        assert np.allclose(test.objective(guess), olp1**2 + olp2**2 - 1)
    # ci wavefunction
    ciref = CIWavefunction(2, 4)
    ciref.assign_params(np.random.rand(6))
    test.assign_refwfn(ciref)
    assert np.allclose(test.objective(guess), sum(ciref.get_overlap(sd) * wfn.get_overlap(sd)
                                                  for sd in sds) - 1)
    # active ci wavefunction
    ciref = CIWavefunction(2, 4)
    ciref.assign_params(np.random.rand(6))
    test = NormConstraint(wfn, param_selection=[(wfn, np.arange(6)), (ciref, np.arange(6))],
                          refwfn=ciref)
    assert np.allclose(test.objective(np.random.rand(12)),
                       sum(ciref.get_overlap(sd) * wfn.get_overlap(sd) for sd in sds) - 1)


def test_norm_gradient():
    """Test NormConstraint.gradient."""
    wfn = CIWavefunction(2, 4)
    test = NormConstraint(wfn, param_selection=[(wfn, np.arange(6))])

    guess = np.random.rand(6)
    # check assignment
    test.gradient(guess)
    assert np.allclose(wfn.params, guess)

    sds = [0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010]
    # single sd
    for sd in sds:
        test.assign_refwfn(sd)
        olp = wfn.get_overlap(sd)
        d_olp = np.array([wfn.get_overlap(sd, deriv=i) for i in range(guess.size)])
        assert np.allclose(test.gradient(guess), 2 * olp * d_olp)
    # multiple sd
    for sd1, sd2 in it.combinations(sds, 2):
        test.assign_refwfn([sd1, sd2])
        olp1 = wfn.get_overlap(sd1)
        olp2 = wfn.get_overlap(sd2)
        d_olp1 = np.array([wfn.get_overlap(sd1, deriv=i) for i in range(guess.size)])
        d_olp2 = np.array([wfn.get_overlap(sd2, deriv=i) for i in range(guess.size)])
        assert np.allclose(test.gradient(guess), 2 * olp1 * d_olp1 + 2 * olp2 * d_olp2)
    # ci wavefunction
    ciref = CIWavefunction(2, 4)
    ciref.assign_params(np.random.rand(6))
    test.assign_refwfn(ciref)
    assert np.allclose(test.gradient(guess),
                       [sum(ciref.get_overlap(sd) * wfn.get_overlap(sd, deriv=i) for sd in sds)
                        for i in range(guess.size)])
    # active ci wavefunction
    ciref = CIWavefunction(2, 4)
    ciref.assign_params(np.random.rand(6))
    test = NormConstraint(wfn, param_selection=[(wfn, np.arange(6)), (ciref, np.arange(6))],
                          refwfn=ciref)
    assert np.allclose(test.gradient(np.random.rand(12)),
                       [sum(ciref.get_overlap(sd) * wfn.get_overlap(sd, deriv=i) for sd in sds)
                        for i in range(6)] +
                       [sum(ciref.get_overlap(sd, deriv=i) * wfn.get_overlap(sd) for sd in sds)
                        for i in range(6)])
