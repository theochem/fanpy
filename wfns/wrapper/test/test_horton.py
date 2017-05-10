""" Tests wfns.wrapper.horton
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from nose.tools import assert_raises
from horton import IOData, PlainSCFSolver
from wfns.tools import find_datafile
from wfns.wrapper.horton import hartreefock, gaussian_fchk


def check_data_h2_rhf_sto6g(data):
    """ Checks data for h2 rhf sto6g calculation
    """
    assert np.allclose(data['el_energy'], -1.838434259892)
    assert np.allclose(data['nuc_nuc_energy'], 0.713176830593)
    assert np.allclose(data['one_int'], np.array([[-1.25637540e+00, 0.0000000000000],
                                                  [0.0000000000000, -4.80588203e-01]]))
    assert np.allclose(data['two_int'], np.array([[[[6.74316543e-01, 0.000000000000],
                                                    [0.000000000000, 1.81610048e-01]],
                                                   [[0.000000000000, 6.64035234e-01],
                                                    [1.81610048e-01, 0.000000000000]]],
                                                  [[[0.000000000000, 1.81610048e-01],
                                                    [6.64035234e-01, 0.000000000000]],
                                                   [[1.81610048e-01, 0.000000000000],
                                                    [0.000000000000, 6.98855952e-01]]]]))

def check_data_h2_uhf_sto6g(data):
    """ Checks data for h2 uhf sto6g calculation
    """
    assert np.allclose(data['el_energy'], -1.838434259892)
    assert np.allclose(data['nuc_nuc_energy'], 0.713176830593)
    assert len(data['one_int']) == 2
    assert np.allclose(data['one_int'][0], np.array([[-1.25637540e+00, 0.0000000000000],
                                                     [0.0000000000000, -4.80588203e-01]]))
    assert np.allclose(data['one_int'][1], np.array([[-1.25637540e+00, 0.0000000000000],
                                                     [0.0000000000000, -4.80588203e-01]]))
    assert len(data['two_int']) == 3
    for two_int in data['two_int']:
        assert np.allclose(two_int, np.array([[[[6.74316543e-01, 0.000000000000],
                                                [0.000000000000, 1.81610048e-01]],
                                               [[0.000000000000, 6.64035234e-01],
                                                [1.81610048e-01, 0.000000000000]]],
                                              [[[0.000000000000, 1.81610048e-01],
                                                [6.64035234e-01, 0.000000000000]],
                                               [[1.81610048e-01, 0.000000000000],
                                                [0.000000000000, 6.98855952e-01]]]]))


def test_hartreefock_h2_rhf_sto6g():
    """ Tests hartreefock against H2 HF STO-6G data from Gaussian
    """
    # file location specified
    hf_dict = hartreefock(fn=find_datafile('test/h2.xyz'), basis="sto-6g", nelec=2)
    check_data_h2_rhf_sto6g(hf_dict)

    # file from data folder
    hf_dict = hartreefock(fn="test/h2.xyz", basis="sto-6g", nelec=2)
    check_data_h2_rhf_sto6g(hf_dict)

    # file in form of IOData (HORTON)
    iodata = IOData.from_file(find_datafile('test/h2.xyz'))
    hf_dict = hartreefock(fn=iodata, basis="sto-6g", nelec=2)
    check_data_h2_rhf_sto6g(hf_dict)

    # using PlainSCFSolver
    hf_dict = hartreefock(fn=find_datafile('test/h2.xyz'),
                          basis="sto-6g", nelec=2, solver=PlainSCFSolver)
    check_data_h2_rhf_sto6g(hf_dict)

    # bad solver
    assert_raises(ValueError,
                  lambda: hartreefock(fn=find_datafile('test/h2.xyz'),
                                      basis="sto-6g", nelec=2, solver='solver'))


def test_gaussian_fchk_h2_rhf_sto6g():
    """ Tests gaussian_fchk against H2 HF STO-6G data from Gaussian
    """
    # file location specified
    fchk_data = gaussian_fchk(find_datafile('test/h2_hf_sto6g.fchk'), horton_internal=True)
    check_data_h2_rhf_sto6g(fchk_data)

    # file from data folder
    fchk_data = gaussian_fchk('test/h2_hf_sto6g.fchk', horton_internal=True)
    check_data_h2_rhf_sto6g(fchk_data)

def test_gaussian_fchk_h2_uhf_sto6g():
    """ Tests gaussian_fchk against H2 UHF STO-6G data from Gaussian
    """
    # file location specified
    fchk_data = gaussian_fchk(find_datafile('test/h2_uhf_sto6g.fchk'), horton_internal=True)
    check_data_h2_uhf_sto6g(fchk_data)

    # file from data folder
    fchk_data = gaussian_fchk('test/h2_uhf_sto6g.fchk', horton_internal=True)
    check_data_h2_uhf_sto6g(fchk_data)
