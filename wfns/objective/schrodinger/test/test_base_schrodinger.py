"""Test wfns.objective.schrodinger.base_schrodinger."""
from nose.tools import assert_raises
import numpy as np
import itertools as it
from wfns.param import ParamContainer
from wfns.objective.schrodinger.base_schrodinger import BaseSchrodinger
from wfns.wavefunction.ci.ci_wavefunction import CIWavefunction
from wfns.hamiltonian.base_hamiltonian import BaseHamiltonian
from wfns.hamiltonian.chemical_hamiltonian import ChemicalHamiltonian


class TestBaseHamiltonian(BaseHamiltonian):
    def integrate_wfn_sd(self, wfn, sd, deriv=None):
        pass

    def integrate_sd_sd(self, sd1, sd2, deriv=None):
        pass


class TestBaseSchrodinger(BaseSchrodinger):
    @property
    def num_eqns(self):
        pass

    def objective(self, params):
        pass


def test_BaseSchrodinger_init():
    """Test BaseSchrodinger.__init__."""
    wfn = CIWavefunction(2, 4)
    ham = TestBaseHamiltonian(np.arange(4, dtype=float).reshape(2, 2),
                              np.arange(16, dtype=float).reshape(2, 2, 2, 2))
    assert_raises(TypeError, TestBaseSchrodinger, ham, ham)
    assert_raises(TypeError, TestBaseSchrodinger, wfn, wfn)
    wfn = CIWavefunction(2, 4, dtype=complex)
    assert_raises(ValueError, TestBaseSchrodinger, wfn, ham)
    wfn = CIWavefunction(2, 6)
    assert_raises(ValueError, TestBaseSchrodinger, wfn, ham)
    wfn = CIWavefunction(2, 4)
    assert_raises(TypeError, TestBaseSchrodinger, wfn, ham, tmpfile=2)

    test = TestBaseSchrodinger(wfn, ham, tmpfile='tmpfile.npy')
    assert test.wfn == wfn
    assert test.ham == ham
    assert test.tmpfile == 'tmpfile.npy'
    assert np.allclose(test.param_selection.all_params, wfn.params)
    assert np.allclose(test.param_selection.active_params, wfn.params)


def test_BaseSchrodinger_wrapped_get_overlap():
    """Test BaseSchrodinger.wrapped_get_overlap."""
    wfn = CIWavefunction(2, 4)
    wfn.assign_params(np.random.rand(wfn.nparams))
    ham = TestBaseHamiltonian(np.arange(4, dtype=float).reshape(2, 2),
                              np.arange(16, dtype=float).reshape(2, 2, 2, 2))
    test = TestBaseSchrodinger(wfn, ham, param_selection=[(wfn, np.array([0, 3, 5])),
                                                        (ParamContainer(3), True)])
    assert test.wrapped_get_overlap(0b0101, deriv=None) == wfn.get_overlap(0b0101, deriv=None)
    assert test.wrapped_get_overlap(0b0101, deriv=0) == wfn.get_overlap(0b0101, deriv=0)
    assert test.wrapped_get_overlap(0b0101, deriv=1) == wfn.get_overlap(0b0101, deriv=3)
    assert test.wrapped_get_overlap(0b0101, deriv=2) == wfn.get_overlap(0b0101, deriv=5)
    assert test.wrapped_get_overlap(0b0101, deriv=3) == 0.0


def test_BaseSchrodinger_wrapped_integrate_wfn_sd():
    """Test BaseSchrodinger.wrapped_integrate_wfn_sd."""
    wfn = CIWavefunction(2, 4)
    wfn.assign_params(np.random.rand(wfn.nparams))
    ham = ChemicalHamiltonian(np.arange(4, dtype=float).reshape(2, 2),
                              np.arange(16, dtype=float).reshape(2, 2, 2, 2))
    test = TestBaseSchrodinger(wfn, ham, param_selection=[(wfn, np.array([0, 3, 5])),
                                                        (ParamContainer(3), True)])
    assert test.wrapped_integrate_wfn_sd(0b0101) == sum(ham.integrate_wfn_sd(wfn, 0b0101))
    assert test.wrapped_integrate_wfn_sd(0b0101,
                                         deriv=0) == sum(ham.integrate_wfn_sd(wfn, 0b0101,
                                                                              wfn_deriv=0))
    assert test.wrapped_integrate_wfn_sd(0b0101,
                                         deriv=1) == sum(ham.integrate_wfn_sd(wfn, 0b0101,
                                                                              wfn_deriv=3))
    assert test.wrapped_integrate_wfn_sd(0b0101,
                                         deriv=2) == sum(ham.integrate_wfn_sd(wfn, 0b0101,
                                                                              wfn_deriv=5))
    # FIXME: no tests for ham_deriv b/c there are no hamiltonians with parameters
    assert test.wrapped_integrate_wfn_sd(0b0101, deriv=3) == 0.0


def test_BaseSchrodinger_wrapped_integrate_sd_sd():
    """Test BaseSchrodinger.wrapped_integrate_sd_sd."""
    wfn = CIWavefunction(2, 4)
    wfn.assign_params(np.random.rand(wfn.nparams))
    ham = ChemicalHamiltonian(np.arange(4, dtype=float).reshape(2, 2),
                              np.arange(16, dtype=float).reshape(2, 2, 2, 2))
    test = TestBaseSchrodinger(wfn, ham, param_selection=[(wfn, np.array([0, 3, 5])),
                                                        (ParamContainer(3), True)])
    assert test.wrapped_integrate_sd_sd(0b0101, 0b0101) == sum(ham.integrate_sd_sd(0b0101, 0b0101))
    assert test.wrapped_integrate_sd_sd(0b0101, 0b0101, deriv=0) == 0.0
    assert test.wrapped_integrate_sd_sd(0b0101, 0b0101, deriv=1) == 0.0
    assert test.wrapped_integrate_sd_sd(0b0101, 0b0101, deriv=2) == 0.0
    assert test.wrapped_integrate_sd_sd(0b0101, 0b0101, deriv=3) == 0.0
    # FIXME: no tests for derivatives wrt hamiltonian b/c there are no hamiltonians with parameters


def test_BaseSchrodinger_get_energy_one_proj():
    """Test BaseSchrodinger.get_energy_one_proj."""
    wfn = CIWavefunction(2, 4)
    wfn.assign_params(np.random.rand(wfn.nparams))
    ham = ChemicalHamiltonian(np.arange(4, dtype=float).reshape(2, 2),
                              np.arange(16, dtype=float).reshape(2, 2, 2, 2))
    test = TestBaseSchrodinger(wfn, ham)

    # sd
    for sd in [0b0101, 0b0110, 0b1001, 0b0110, 0b1010, 0b1100]:
        olp = wfn.get_overlap(sd)
        integral = sum(ham.integrate_wfn_sd(wfn, sd))
        # <SD | H | Psi> = E <SD | Psi>
        # E = <SD | H | Psi> / <SD | Psi>
        assert np.allclose(test.get_energy_one_proj(sd), integral / olp)
        # dE = d<SD | H | Psi> / <SD | Psi> - d<SD | Psi> <SD | H | Psi> / <SD | Psi>^2
        for i in range(4):
            d_olp = wfn.get_overlap(sd, deriv=i)
            d_integral = sum(ham.integrate_wfn_sd(wfn, sd, wfn_deriv=i))
            assert np.allclose(test.get_energy_one_proj(sd, deriv=i),
                               d_integral / olp - d_olp * integral / olp**2)

    # list of sd
    for sd1, sd2 in it.combinations([0b0101, 0b0110, 0b1001, 0b0110, 0b1010, 0b1100], 2):
        olp1 = wfn.get_overlap(sd1)
        olp2 = wfn.get_overlap(sd2)
        integral1 = sum(ham.integrate_wfn_sd(wfn, sd1))
        integral2 = sum(ham.integrate_wfn_sd(wfn, sd2))
        # ( f(SD1) <SD1| + f(SD2) <SD2| ) H |Psi> = E ( f(SD1) <SD1| + f(SD2) <SD2| ) |Psi>
        # f(SD1) <SD1| H |Psi> + f(SD2) <SD2| H |Psi> = E ( f(SD1) <SD1|Psi> + f(SD2) <SD2|Psi> )
        # E = (f(SD1) <SD1| H |Psi> + f(SD2) <SD2| H |Psi>) / (f(SD1) <SD1|Psi> + f(SD2) <SD2|Psi>)
        # where f(SD) = <SD | Psi>
        assert np.allclose(test.get_energy_one_proj([sd1, sd2]),
                           (olp1 * integral1 + olp2 * integral2) / (olp1**2 + olp2**2))
        # dE
        # = d(f(SD1) <SD1| H |Psi> + f(SD2) <SD2| H |Psi>) / (f(SD1) <SD1|Psi> + f(SD2) <SD2|Psi>) -
        #   d(f(SD1) <SD1|Psi> + f(SD2) <SD2|Psi>) (f(SD1) <SD1| H |Psi> + f(SD2) <SD2| H |Psi>) /
        #     (f(SD1) <SD1|Psi> + f(SD2) <SD2|Psi>)**2
        # = (d(f(SD1) <SD1| H |Psi>) + d(f(SD2) <SD2| H |Psi>)) /
        #     (f(SD1) <SD1|Psi> + f(SD2) <SD2|Psi>) -
        #   (d(f(SD1) <SD1|Psi>) + d(f(SD2) <SD2|Psi>)) *
        #     (f(SD1) <SD1| H |Psi> + f(SD2) <SD2| H |Psi>) /
        #       (f(SD1) <SD1|Psi> + f(SD2) <SD2|Psi>)**2
        # = (df(SD1) <SD1| H |Psi> + f(SD1) d<SD1| H |Psi>
        #     + df(SD2) <SD2| H |Psi> + f(SD2) d<SD2| H |Psi>) /
        #       (f(SD1) <SD1|Psi> + f(SD2) <SD2|Psi>) -
        #   (df(SD1) <SD1|Psi> + f(SD1) d<SD1|Psi> + df(SD2) <SD2|Psi> + f(SD2) d <SD2|Psi>) *
        #     (f(SD1) <SD1| H |Psi> + f(SD2) <SD2| H |Psi>) /
        #       (f(SD1) <SD1|Psi> + f(SD2) <SD2|Psi>)**2
        for i in range(4):
            d_olp1 = wfn.get_overlap(sd1, deriv=i)
            d_olp2 = wfn.get_overlap(sd2, deriv=i)
            d_integral1 = sum(ham.integrate_wfn_sd(wfn, sd1, wfn_deriv=i))
            d_integral2 = sum(ham.integrate_wfn_sd(wfn, sd2, wfn_deriv=i))
            assert np.allclose(test.get_energy_one_proj([sd1, sd2], deriv=i),
                               (d_olp1 * integral1 + d_olp2 * integral2
                                + olp1 * d_integral1 + olp2 * d_integral2) / (olp1**2 + olp2**2) -
                               (2 * d_olp1 * olp1 + 2 * d_olp2 * olp2) *
                               (olp1 * integral1 + olp2 * integral2) / (olp1**2 + olp2**2)**2)

    # CI
    for sd1, sd2 in it.combinations([0b0101, 0b0110, 0b1001, 0b0110, 0b1010, 0b1100], 2):
        ciwfn = CIWavefunction(2, 4, sd_vec=[sd1, sd2])
        ciwfn.assign_params(np.random.rand(ciwfn.nparams))
        coeff1 = ciwfn.get_overlap(sd1)
        coeff2 = ciwfn.get_overlap(sd2)
        olp1 = wfn.get_overlap(sd1)
        olp2 = wfn.get_overlap(sd2)
        integral1 = sum(ham.integrate_wfn_sd(wfn, sd1))
        integral2 = sum(ham.integrate_wfn_sd(wfn, sd2))
        # ( c_1 <SD1| + c_2 <SD2| ) H |Psi> = E ( c_1 <SD1| + c_2 <SD2| ) |Psi>
        # c_1 <SD1| H |Psi> + c_2 <SD2| H |Psi> = E ( c_1 <SD1|Psi> + c_2 <SD2|Psi> )
        # E = (c_1 <SD1| H |Psi> + c_2 <SD2| H |Psi>) / (c_1 <SD1|Psi> + c_2 <SD2|Psi>)
        assert np.allclose(test.get_energy_one_proj(ciwfn),
                           (coeff1*integral1 + coeff2*integral2) / (coeff1*olp1 + coeff2*olp2))
        # dE = (dc_1 <SD1| H |Psi> + c_1 d<SD1| H |Psi>
        #        + dc_2 <SD2| H |Psi> + c_2 d<SD2| H |Psi>) /
        #          (c_1 <SD1|Psi> + c_2 <SD2|Psi>) -
        #      (dc_1 <SD1|Psi> + c_1 d<SD1|Psi> + dc_2 <SD2|Psi> + c_2 d <SD2|Psi>) *
        #        (c_1 <SD1| H |Psi> + c_2 <SD2| H |Psi>) /
        #          (c_1 <SD1|Psi> + c_2 <SD2|Psi>)**2
        for i in range(4):
            d_coeff1 = 0.0
            d_coeff2 = 0.0
            d_olp1 = wfn.get_overlap(sd1, deriv=i)
            d_olp2 = wfn.get_overlap(sd2, deriv=i)
            d_integral1 = sum(ham.integrate_wfn_sd(wfn, sd1, wfn_deriv=i))
            d_integral2 = sum(ham.integrate_wfn_sd(wfn, sd2, wfn_deriv=i))
            assert np.allclose(test.get_energy_one_proj(ciwfn, deriv=i),
                               (d_coeff1 * integral1 + d_coeff2 * integral2
                                + coeff1 * d_integral1 + coeff2 * d_integral2)
                               / (coeff1 * olp1 + coeff2 * olp2) -
                               (d_coeff1 * olp1 + coeff1 * d_olp1 +
                                d_coeff2 * olp2 + coeff2 * d_olp2) *
                               (coeff1 * integral1 + coeff2 * integral2)
                               / (coeff1 * olp1 + coeff2 * olp2)**2)

        # others
        assert_raises(TypeError, test.get_energy_one_proj, '0b0101')


def test_BaseSchrodinger_get_energy_two_proj():
    """Test BaseSchrodinger.get_energy_two_proj."""
    wfn = CIWavefunction(2, 4)
    wfn.assign_params(np.random.rand(wfn.nparams))
    ham = ChemicalHamiltonian(np.arange(4, dtype=float).reshape(2, 2),
                              np.arange(16, dtype=float).reshape(2, 2, 2, 2))
    test = TestBaseSchrodinger(wfn, ham)

    # sd
    for sd_l, sd_r, sd_n in it.permutations([0b0101, 0b0110, 0b1001, 0b0110, 0b1010, 0b1100], 3):
        olp_l = wfn.get_overlap(sd_l)
        olp_r = wfn.get_overlap(sd_r)
        olp_n = wfn.get_overlap(sd_n)
        integral = sum(ham.integrate_sd_sd(sd_l, sd_r))
        # <Psi | SDl> <SDl | H | SDr> <SDr | Psi> = E <Psi | SDn> <SDn | Psi>
        # E = <Psi | SDl> <SDl | H | SDr> <SDr | Psi> / <Psi | SDn>^2
        assert np.allclose(test.get_energy_two_proj(sd_l, sd_r, sd_n),
                           olp_l * integral * olp_r / olp_n**2)
        # dE = d<Psi | SDl> <SDl | H | SDr> <SDr | Psi> / <Psi | SDn>^2 +
        #      <Psi | SDl> d<SDl | H | SDr> <SDr | Psi> / <Psi | SDn>^2 +
        #      <Psi | SDl> <SDl | H | SDr> d<SDr | Psi> / <Psi | SDn>^2 -
        #      2 d<Psi | SDn> <Psi | SDl> <SDl | H | SDr> <SDr | Psi> / <Psi | SDn>^3
        for i in range(4):
            d_olp_l = wfn.get_overlap(sd_l, deriv=i)
            d_olp_r = wfn.get_overlap(sd_r, deriv=i)
            d_olp_n = wfn.get_overlap(sd_n, deriv=i)
            # FIXME: hamiltonian does not support parameters right now, so cannot derivatize wrt it
            # d_integral = sum(ham.integrate_sd_sd(sd_l, sd_r, deriv=i))
            d_integral = 0.0
            assert np.allclose(test.get_energy_two_proj(sd_l, sd_r, sd_n, deriv=i),
                               (d_olp_l * integral * olp_r / olp_n**2 +
                                olp_l * d_integral * olp_r / olp_n**2 +
                                olp_l * integral * d_olp_r / olp_n**2 -
                                2 * d_olp_n * olp_l * integral * olp_r / olp_n**3))

    # list of sd
    for sds_l, sds_r, sds_n in it.permutations(it.permutations([0b0101, 0b0110, 0b1001,
                                                                0b0110, 0b1010, 0b1100], 2), 3):
        # randomly skip 9/10 of the tests
        if np.random.random() < 0.9:
            continue

        sd_l1, sd_l2 = sds_l
        sd_r1, sd_r2 = sds_r
        sd_n1, sd_n2 = sds_n
        olp_l1 = wfn.get_overlap(sd_l1)
        olp_l2 = wfn.get_overlap(sd_l2)
        olp_r1 = wfn.get_overlap(sd_r1)
        olp_r2 = wfn.get_overlap(sd_r2)
        olp_n1 = wfn.get_overlap(sd_n1)
        olp_n2 = wfn.get_overlap(sd_n2)
        integral11 = sum(ham.integrate_sd_sd(sd_l1, sd_r1))
        integral12 = sum(ham.integrate_sd_sd(sd_l1, sd_r2))
        integral21 = sum(ham.integrate_sd_sd(sd_l2, sd_r1))
        integral22 = sum(ham.integrate_sd_sd(sd_l2, sd_r2))
        # <Psi| (|SDl1> <SDl1| + |SDl2> <SDl2| ) H (|SDr1> <SDr1| + |SDr2> <SDr2| ) |Psi>
        # = E <Psi| (|SDn1> <SDn1| + |SDn2> <SDn2|) | Psi>
        # <Psi|SDl1> <SDl1|H|SDr1> <SDr1|Psi> + <Psi|SDl2> <SDl2|H|SDr1> <SDr1|Psi> +
        # <Psi|SDl1> <SDl1|H|SDr2> <SDr2|Psi> + <Psi|SDl2> <SDl2|H|SDr2> <SDr2|Psi>
        # = E (<Psi|SDn1> <SDn1|Psi> + <Psi|SDn2> <SDn2|Psi>)
        # E = (<Psi|SDl1> <SDl1|H|SDr1> <SDr1|Psi> + <Psi|SDl2> <SDl2|H|SDr1> <SDr1|Psi> +
        #      <Psi|SDl1> <SDl1|H|SDr2> <SDr2|Psi> + <Psi|SDl2> <SDl2|H|SDr2> <SDr2|Psi>) /
        #     (<Psi|SDn1> <SDn1|Psi> + <Psi|SDn2> <SDn2|Psi>)
        assert np.allclose(test.get_energy_two_proj(sds_l, sds_r, sds_n),
                           ((olp_l1 * integral11 * olp_r1 + olp_l2 * integral21 * olp_r1 +
                             olp_l1 * integral12 * olp_r2 + olp_l2 * integral22 * olp_r2) /
                            (olp_n1**2 + olp_n2**2)))
        # dE = d(<Psi|SDl1> <SDl1|H|SDr1> <SDr1|Psi> + <Psi|SDl2> <SDl2|H|SDr1> <SDr1|Psi> +
        #        <Psi|SDl1> <SDl1|H|SDr2> <SDr2|Psi> + <Psi|SDl2> <SDl2|H|SDr2> <SDr2|Psi>) /
        #       (<Psi|SDn1> <SDn1|Psi> + <Psi|SDn2> <SDn2|Psi>) +
        #      d(<Psi|SDn1> <SDn1|Psi> + <Psi|SDn2> <SDn2|Psi>) *
        #       (<Psi|SDl1> <SDl1|H|SDr1> <SDr1|Psi> + <Psi|SDl2> <SDl2|H|SDr1> <SDr1|Psi> +
        #        <Psi|SDl1> <SDl1|H|SDr2> <SDr2|Psi> + <Psi|SDl2> <SDl2|H|SDr2> <SDr2|Psi>) /
        #       (<Psi|SDn1> <SDn1|Psi> + <Psi|SDn2> <SDn2|Psi>)^2
        #    = (d(<Psi|SDl1> <SDl1|H|SDr1> <SDr1|Psi>) + d(<Psi|SDl2> <SDl2|H|SDr1> <SDr1|Psi>) +
        #       d(<Psi|SDl1> <SDl1|H|SDr2> <SDr2|Psi>) + d(<Psi|SDl2> <SDl2|H|SDr2> <SDr2|Psi>)) /
        #      (<Psi|SDn1> <SDn1|Psi> + <Psi|SDn2> <SDn2|Psi>) +
        #      (d(<Psi|SDn1> <SDn1|Psi>) + d(<Psi|SDn2> <SDn2|Psi>)) *
        #      (<Psi|SDl1> <SDl1|H|SDr1> <SDr1|Psi> + <Psi|SDl2> <SDl2|H|SDr1> <SDr1|Psi> +
        #       <Psi|SDl1> <SDl1|H|SDr2> <SDr2|Psi> + <Psi|SDl2> <SDl2|H|SDr2> <SDr2|Psi>) /
        #      (<Psi|SDn1> <SDn1|Psi> + <Psi|SDn2> <SDn2|Psi>)^2
        #    = (d<Psi|SDl1> <SDl1|H|SDr1> <SDr1|Psi> + <Psi|SDl1> d<SDl1|H|SDr1> <SDr1|Psi> +
        #       <Psi|SDl1> <SDl1|H|SDr1> d<SDr1|Psi> + d<Psi|SDl2> <SDl2|H|SDr1> <SDr1|Psi> +
        #       <Psi|SDl2> d<SDl2|H|SDr1> <SDr1|Psi> + <Psi|SDl2> <SDl2|H|SDr1> d<SDr1|Psi> +
        #       d<Psi|SDl1> <SDl1|H|SDr2> <SDr2|Psi> + <Psi|SDl1> d<SDl1|H|SDr2> <SDr2|Psi> +
        #       <Psi|SDl1> <SDl1|H|SDr2> d<SDr2|Psi> + d<Psi|SDl2> <SDl2|H|SDr2> <SDr2|Psi> +
        #       <Psi|SDl2> d<SDl2|H|SDr2> <SDr2|Psi> + <Psi|SDl2> <SDl2|H|SDr2> d<SDr2|Psi>) /
        #      (<Psi|SDn1> <SDn1|Psi> + <Psi|SDn2> <SDn2|Psi>) +
        #      (2 * d<Psi|SDn1> <SDn1|Psi> + 2 * d<Psi|SDn2> <SDn2|Psi>) *
        #      (<Psi|SDl1> <SDl1|H|SDr1> <SDr1|Psi> + <Psi|SDl2> <SDl2|H|SDr1> <SDr1|Psi> +
        #       <Psi|SDl1> <SDl1|H|SDr2> <SDr2|Psi> + <Psi|SDl2> <SDl2|H|SDr2> <SDr2|Psi>) /
        #      (<Psi|SDn1> <SDn1|Psi> + <Psi|SDn2> <SDn2|Psi>)^2
        for i in range(4):
            d_olp_l1 = wfn.get_overlap(sd_l1, deriv=i)
            d_olp_r1 = wfn.get_overlap(sd_r1, deriv=i)
            d_olp_n1 = wfn.get_overlap(sd_n1, deriv=i)
            d_olp_l2 = wfn.get_overlap(sd_l2, deriv=i)
            d_olp_r2 = wfn.get_overlap(sd_r2, deriv=i)
            d_olp_n2 = wfn.get_overlap(sd_n2, deriv=i)
            # FIXME: hamiltonian does not support parameters right now, so cannot derivatize wrt it
            # d_integral11 = sum(ham.integrate_sd_sd(sd_l1, sd_r1, deriv=i))
            # d_integral12 = sum(ham.integrate_sd_sd(sd_l2, sd_r1, deriv=i))
            # d_integral21 = sum(ham.integrate_sd_sd(sd_l1, sd_r2, deriv=i))
            # d_integral22 = sum(ham.integrate_sd_sd(sd_l2, sd_r2, deriv=i))
            d_integral11 = 0.0
            d_integral21 = 0.0
            d_integral12 = 0.0
            d_integral22 = 0.0
            assert np.allclose(test.get_energy_two_proj(sds_l, sds_r, sds_n, deriv=i),
                               ((d_olp_l1 * integral11 * olp_r1 + olp_l1 * d_integral11 * olp_r1 +
                                 olp_l1 * integral11 * d_olp_r1 + d_olp_l2 * integral21 * olp_r1 +
                                 olp_l2 * d_integral21 * olp_r1 + olp_l2 * integral21 * d_olp_r1 +
                                 d_olp_l1 * integral12 * olp_r2 + olp_l1 * d_integral12 * olp_r2 +
                                 olp_l1 * integral12 * d_olp_r2 + d_olp_l2 * integral22 * olp_r2 +
                                 olp_l2 * d_integral22 * olp_r2 + olp_l2 * integral22 * d_olp_r2) /
                                (olp_n1**2 + olp_n2**2) -
                                (2 * d_olp_n1 * olp_n1 + 2 * d_olp_n2 * olp_n2) *
                                (olp_l1 * integral11 * olp_r1 + olp_l2 * integral21 * olp_r1 +
                                 olp_l1 * integral12 * olp_r2 + olp_l2 * integral22 * olp_r2) /
                                (olp_n1**2 + olp_n2**2)**2))
