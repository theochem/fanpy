"""Test wfns.wavefunction.ci_pairs."""
from nose.tools import assert_raises
import numpy as np
from wfns.tools import find_datafile
from wfns.wfn.ci.ci_pairs import CIPairs
from wfns.ham.senzero import SeniorityZeroHamiltonian
from wfns.solver.ci import brute


class TestCIPairs(CIPairs):
    """CIPairs class without initializer."""
    def __init__(self):
        pass


def test_assign_sd_vec():
    """Test CIPairs.assign_sd_vec."""
    test = TestCIPairs()
    test.assign_nelec(6)
    test.assign_nspin(10)
    test.assign_spin(0)
    test.assign_seniority(0)
    test.assign_sd_vec()
    assert test.sd_vec == (0b0011100111, 0b0101101011, 0b1001110011, 0b0110101101, 0b1010110101,
                           0b0111001110, 0b1011010110)
    assert_raises(ValueError, test.assign_sd_vec, (0b0011100111, ))


def test_to_ap1rog():
    """Test CIPairs.to_ap1rog."""
    test = CIPairs(2, 6, dtype=float)
    params = np.arange(9, 0, -1, dtype=float).reshape(3, 3)
    test.assign_params(params[:, 0].flatten())
    ap1rog = test.to_ap1rog()
    assert np.allclose(ap1rog.params, np.array([6/9, 3/9]))
    test.assign_params(params[:, 1].flatten())
    ap1rog = test.to_ap1rog()
    assert np.allclose(ap1rog.params, np.array([5/8, 2/8]))
    test.assign_params(params[:, 2].flatten())
    ap1rog = test.to_ap1rog()
    assert np.allclose(ap1rog.params, np.array([4/7, 1/7]))


@np.testing.dec.skipif(True, 'Cannot find reference for comparison.')
def test_to_ap1rog_h2_sto6g_ground():
    """Test wfns.wavefunction.ci_pairs.CIPairs.to_ap1rog using H2 with HF/STO6G orbitals."""
    nelec = 2
    nspin = 4
    cipairs = CIPairs(nelec, nspin)

    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/h2_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile('test/h2_hf_sto6g_oneint.npy'))
    two_int = np.load(find_datafile('test/h2_hf_sto6g_twoint.npy'))
    nuc_nuc = 0.71317683129
    ham = SeniorityZeroHamiltonian(one_int, two_int, energy_nuc_nuc=nuc_nuc)

    energies, coeffs = brute(cipairs, ham)


@np.testing.dec.skipif(True, 'Cannot find reference for comparison.')
def test_to_ap1rog_lih_sto6g():
    """Test wfns.wavefunction.ci_pairs.CIPairs.to_ap1rog with LiH with HF/STO6G orbitals."""
    nelec = 4
    nspin = 12
    cipairs = CIPairs(nelec, nspin)

    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/lih_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = (np.load(find_datafile('test/lih_hf_sto6g_oneint.npy')), )
    two_int = (np.load(find_datafile('test/lih_hf_sto6g_twoint.npy')), )
    nuc_nuc = 0.995317634356
    ham = SeniorityZeroHamiltonian(one_int, two_int, energy_nuc_nuc=nuc_nuc)

    energies, coeffs = brute(cipairs, ham)
    raise AssertionError('No reference for the CIPairs tests.')


@np.testing.dec.skipif(True, 'Cannot find reference for comparison.')
def test_to_ap1rog_h4_sto6g():
    """Test wfns.wavefunction.ci_pairs.CIPairs.to_ap1rog with H4 with HF/STO6G orbitals."""
    nelec = 4
    nspin = 8
    cipairs = CIPairs(nelec, nspin)

    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/h4_square_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile('test/h4_square_hf_sto6g_oneint.npy'))
    two_int = np.load(find_datafile('test/h4_square_hf_sto6g_twoint.npy'))
    nuc_nuc = 2.70710678119
    ham = SeniorityZeroHamiltonian(one_int, two_int, energy_nuc_nuc=nuc_nuc)

    energies, coeffs = brute(cipairs, ham)
    raise AssertionError('No reference for the CIPairs tests.')
