"""Test wfns.eqn.base."""
import itertools as it

import numpy as np
import pytest
from utils import disable_abstract, skip_init
from wfns.ham.restricted_chemical import RestrictedChemicalHamiltonian
from wfns.eqn.base import BaseSchrodinger
from wfns.eqn.utils import ParamContainer, ComponentParameterIndices
from wfns.wfn.ci.base import CIWavefunction


def test_baseschrodinger_init():
    """Test BaseSchrodinger.__init__."""
    wfn = CIWavefunction(2, 4)
    ham = RestrictedChemicalHamiltonian(
        np.arange(4, dtype=float).reshape(2, 2), np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    )
    with pytest.raises(TypeError):
        disable_abstract(BaseSchrodinger)(ham, ham)
    with pytest.raises(TypeError):
        disable_abstract(BaseSchrodinger)(wfn, wfn)
    wfn = CIWavefunction(2, 6)
    with pytest.raises(ValueError):
        disable_abstract(BaseSchrodinger)(wfn, ham)
    wfn = CIWavefunction(2, 4)
    with pytest.raises(TypeError):
        disable_abstract(BaseSchrodinger)(wfn, ham, tmpfile=2)

    test = disable_abstract(BaseSchrodinger)(wfn, ham, tmpfile="tmpfile.npy")
    assert test.wfn == wfn
    assert test.ham == ham
    assert test.tmpfile == "tmpfile.npy"

    test = disable_abstract(BaseSchrodinger)(wfn, ham, param_selection=[(wfn, [0, 1])])
    assert np.allclose(test.all_params, wfn.params)
    assert np.allclose(test.active_params, wfn.params[:2])

    test = disable_abstract(BaseSchrodinger)(wfn, ham, param_selection=[(wfn, [1, 3]), (ham, [0])])
    assert isinstance(test.indices_component_params, ComponentParameterIndices)
    answer = ComponentParameterIndices()
    answer[wfn] = np.array([1, 3])
    answer[ham] = np.array([0])
    assert test.indices_component_params == answer


def test_baseschrodinger_active_params():
    """Test BaseSchrodinger.active_params."""
    wfn = CIWavefunction(2, 4)
    ham = RestrictedChemicalHamiltonian(
        np.arange(4, dtype=float).reshape(2, 2), np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    )
    param1 = ParamContainer(1)
    param2 = ParamContainer(np.array([2, 3]))
    param3 = ParamContainer(np.array([4, 5, 6, 7]))
    test = disable_abstract(BaseSchrodinger)(
        wfn, ham, param_selection=[
            (param1, [False]), (param2, np.array([0])), (param3, np.array([True, False, False, True]))
        ]
    )
    assert np.allclose(test.active_params, np.array([2, 4, 7]))


def test_baseschrodinger_assign_params():
    """Test BaseSchrodinger.assign_params."""
    wfn = CIWavefunction(2, 4)
    ham = RestrictedChemicalHamiltonian(
        np.arange(4, dtype=float).reshape(2, 2), np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    )
    param1 = ParamContainer(1)
    param2 = ParamContainer(np.array([2, 3]))
    param3 = ParamContainer(np.array([4, 5, 6, 7]))
    test = disable_abstract(BaseSchrodinger)(
        wfn, ham, param_selection=[
            (param1, [False]), (param2, np.array([0])), (param3, np.array([True, False, False, True]))
        ]
    )
    test.assign_params(np.array([99, 98, 97]))
    assert np.allclose(param1.params, [1])
    assert np.allclose(param2.params, [99, 3])
    assert np.allclose(param3.params, [98, 5, 6, 97])


def test_baseschrodinger_wrapped_get_overlap():
    """Test BaseSchrodinger.wrapped_get_overlap."""
    wfn = CIWavefunction(2, 4)
    wfn.assign_params(np.random.rand(wfn.nparams))
    ham = RestrictedChemicalHamiltonian(
        np.arange(4, dtype=float).reshape(2, 2), np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    )
    test = disable_abstract(BaseSchrodinger)(
        wfn, ham, param_selection=[(wfn, np.array([0, 3, 5])), (ParamContainer(3), [True])]
    )
    assert test.wrapped_get_overlap(0b0101) == wfn.get_overlap(0b0101, deriv=None)
    assert np.allclose(
        test.wrapped_get_overlap(0b0101, deriv=True),
        np.hstack([wfn.get_overlap(0b0101, deriv=np.array([0, 3, 5])), 0]),
    )


def test_baseschrodinger_wrapped_integrate_sd_wfn():
    """Test BaseSchrodinger.wrapped_integrate_sd_wfn."""
    wfn = CIWavefunction(5, 10)
    wfn.assign_params(np.random.rand(wfn.nparams))

    one_int = np.random.rand(5, 5)
    one_int = one_int + one_int.T
    two_int = np.random.rand(5, 5, 5, 5)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int
    ham = RestrictedChemicalHamiltonian(one_int, two_int)

    test = disable_abstract(BaseSchrodinger)(
        wfn, ham, param_selection=[(wfn, np.array([0, 3, 5])), (ham, np.array([1, 2, 4]))]
    )
    assert np.allclose(test.wrapped_integrate_sd_wfn(0b0101), ham.integrate_sd_wfn(0b0101, wfn))
    assert np.allclose(
        test.wrapped_integrate_sd_wfn(0b0101, deriv=True),
        np.hstack(
            [
                ham.integrate_sd_wfn(0b0101, wfn, wfn_deriv=np.array([0, 3, 5])),
                ham.integrate_sd_wfn(0b0101, wfn, ham_deriv=np.array([1, 2, 4])),
            ]
        )
    )


def test_baseschrodinger_wrapped_integrate_sd_sd():
    """Test BaseSchrodinger.wrapped_integrate_sd_sd."""
    wfn = CIWavefunction(2, 20)
    wfn.assign_params(np.random.rand(wfn.nparams))
    ham = RestrictedChemicalHamiltonian(
        np.arange(100, dtype=float).reshape(10, 10),
        np.arange(10000, dtype=float).reshape(10, 10, 10, 10),
    )
    test = disable_abstract(BaseSchrodinger)(
        wfn, ham, param_selection=[(wfn, np.array([0, 3, 5])), (ParamContainer(3), [True])]
    )
    assert test.wrapped_integrate_sd_sd(0b0101, 0b0101) == ham.integrate_sd_sd(0b0101, 0b0101)
    assert np.allclose(test.wrapped_integrate_sd_sd(0b0101, 0b0101, deriv=True), 0.0)

    test = disable_abstract(BaseSchrodinger)(
        wfn, ham, param_selection=[(wfn, np.array([0, 3, 5])), (ham, np.array([0, 1, 2]))]
    )
    assert test.wrapped_integrate_sd_sd(0b0101, 0b0101) == ham.integrate_sd_sd(0b0101, 0b0101)
    assert np.allclose(
        test.wrapped_integrate_sd_sd(0b0101, 0b0101, deriv=True),
        np.hstack([np.zeros(3), ham.integrate_sd_sd(0b0101, 0b0101, deriv=np.array([0, 1, 2]))])
    )


def test_baseschrodinger_get_energy_one_proj():
    """Test BaseSchrodinger.get_energy_one_proj."""
    wfn = CIWavefunction(2, 4)
    wfn.assign_params(np.random.rand(wfn.nparams))

    one_int = np.random.rand(2, 2)
    one_int = one_int + one_int.T
    two_int = np.random.rand(2, 2, 2, 2)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int
    ham = RestrictedChemicalHamiltonian(one_int, two_int)

    test = disable_abstract(BaseSchrodinger)(wfn, ham)

    sds = [0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010]
    # sd
    for sd in sds:
        olp = wfn.get_overlap(sd)
        integral = (ham.integrate_sd_wfn(sd, wfn))
        # <SD | H | Psi> = E <SD | Psi>
        # E = <SD | H | Psi> / <SD | Psi>
        assert np.allclose(test.get_energy_one_proj(sd), integral / olp)
        # dE = d<SD | H | Psi> / <SD | Psi> - d<SD | Psi> <SD | H | Psi> / <SD | Psi>^2
        d_olp = wfn.get_overlap(sd, deriv=np.arange(6))
        d_integral = (ham.integrate_sd_wfn(sd, wfn, wfn_deriv=np.arange(6)))
        assert np.allclose(
            test.get_energy_one_proj(sd, deriv=True),
            d_integral / olp - d_olp * integral / olp ** 2,
        )

    # list of sd
    for sd1, sd2 in it.combinations(sds, 2):
        olp1 = wfn.get_overlap(sd1)
        olp2 = wfn.get_overlap(sd2)
        integral1 = (ham.integrate_sd_wfn(sd1, wfn))
        integral2 = (ham.integrate_sd_wfn(sd2, wfn))
        # ( f(SD1) <SD1| + f(SD2) <SD2| ) H |Psi> = E ( f(SD1) <SD1| + f(SD2) <SD2| ) |Psi>
        # f(SD1) <SD1| H |Psi> + f(SD2) <SD2| H |Psi> = E ( f(SD1) <SD1|Psi> + f(SD2) <SD2|Psi> )
        # E = (f(SD1) <SD1| H |Psi> + f(SD2) <SD2| H |Psi>) / (f(SD1) <SD1|Psi> + f(SD2) <SD2|Psi>)
        # where f(SD) = <SD | Psi>
        assert np.allclose(
            test.get_energy_one_proj([sd1, sd2]),
            (olp1 * integral1 + olp2 * integral2) / (olp1 ** 2 + olp2 ** 2),
        )
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
        d_olp1 = wfn.get_overlap(sd1, deriv=np.arange(6))
        d_olp2 = wfn.get_overlap(sd2, deriv=np.arange(6))
        d_integral1 = (ham.integrate_sd_wfn(sd1, wfn, wfn_deriv=np.arange(6)))
        d_integral2 = (ham.integrate_sd_wfn(sd2, wfn, wfn_deriv=np.arange(6)))
        assert np.allclose(
            test.get_energy_one_proj([sd1, sd2], deriv=True),
            (d_olp1 * integral1 + d_olp2 * integral2 + olp1 * d_integral1 + olp2 * d_integral2)
            / (olp1 ** 2 + olp2 ** 2)
            - (2 * d_olp1 * olp1 + 2 * d_olp2 * olp2)
            * (olp1 * integral1 + olp2 * integral2)
            / (olp1 ** 2 + olp2 ** 2) ** 2,
        )

    # CI
    for sd1, sd2 in it.combinations(sds, 2):
        ciwfn = CIWavefunction(2, 4, sd_vec=[sd1, sd2])
        ciwfn.assign_params(np.random.rand(ciwfn.nparams))
        coeff1 = ciwfn.get_overlap(sd1)
        coeff2 = ciwfn.get_overlap(sd2)
        olp1 = wfn.get_overlap(sd1)
        olp2 = wfn.get_overlap(sd2)
        integral1 = (ham.integrate_sd_wfn(sd1, wfn))
        integral2 = (ham.integrate_sd_wfn(sd2, wfn))
        # ( c_1 <SD1| + c_2 <SD2| ) H |Psi> = E ( c_1 <SD1| + c_2 <SD2| ) |Psi>
        # c_1 <SD1| H |Psi> + c_2 <SD2| H |Psi> = E ( c_1 <SD1|Psi> + c_2 <SD2|Psi> )
        # E = (c_1 <SD1| H |Psi> + c_2 <SD2| H |Psi>) / (c_1 <SD1|Psi> + c_2 <SD2|Psi>)
        assert np.allclose(
            test.get_energy_one_proj(ciwfn),
            (coeff1 * integral1 + coeff2 * integral2) / (coeff1 * olp1 + coeff2 * olp2),
        )
        # dE = (dc_1 <SD1| H |Psi> + c_1 d<SD1| H |Psi>
        #        + dc_2 <SD2| H |Psi> + c_2 d<SD2| H |Psi>) /
        #          (c_1 <SD1|Psi> + c_2 <SD2|Psi>) -
        #      (dc_1 <SD1|Psi> + c_1 d<SD1|Psi> + dc_2 <SD2|Psi> + c_2 d <SD2|Psi>) *
        #        (c_1 <SD1| H |Psi> + c_2 <SD2| H |Psi>) /
        #          (c_1 <SD1|Psi> + c_2 <SD2|Psi>)**2
        d_coeff1 = 0.0
        d_coeff2 = 0.0
        d_olp1 = wfn.get_overlap(sd1, deriv=np.arange(6))
        d_olp2 = wfn.get_overlap(sd2, deriv=np.arange(6))
        d_integral1 = (ham.integrate_sd_wfn(sd1, wfn, wfn_deriv=np.arange(6)))
        d_integral2 = (ham.integrate_sd_wfn(sd2, wfn, wfn_deriv=np.arange(6)))
        assert np.allclose(
            test.get_energy_one_proj(ciwfn, deriv=True),
            (
                d_coeff1 * integral1
                + d_coeff2 * integral2
                + coeff1 * d_integral1
                + coeff2 * d_integral2
            )
            / (coeff1 * olp1 + coeff2 * olp2)
            - (d_coeff1 * olp1 + coeff1 * d_olp1 + d_coeff2 * olp2 + coeff2 * d_olp2)
            * (coeff1 * integral1 + coeff2 * integral2)
            / (coeff1 * olp1 + coeff2 * olp2) ** 2,
        )

        # others
        with pytest.raises(TypeError):
            test.get_energy_one_proj("0b0101")


def test_baseschrodinger_get_energy_two_proj():
    """Test BaseSchrodinger.get_energy_two_proj."""
    wfn = CIWavefunction(2, 4)
    wfn.assign_params(np.random.rand(wfn.nparams))
    ham = RestrictedChemicalHamiltonian(
        np.arange(4, dtype=float).reshape(2, 2), np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    )
    test = disable_abstract(BaseSchrodinger)(wfn, ham)

    sds = [0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010]
    # sd
    for sd_l, sd_r, sd_n in it.permutations(sds, 3):
        olp_l = wfn.get_overlap(sd_l)
        olp_r = wfn.get_overlap(sd_r)
        olp_n = wfn.get_overlap(sd_n)
        integral = (ham.integrate_sd_sd(sd_l, sd_r))
        # <Psi | SDl> <SDl | H | SDr> <SDr | Psi> = E <Psi | SDn> <SDn | Psi>
        # E = <Psi | SDl> <SDl | H | SDr> <SDr | Psi> / <Psi | SDn>^2
        assert np.allclose(
            test.get_energy_two_proj(sd_l, sd_r, sd_n), olp_l * integral * olp_r / olp_n ** 2
        )
        # dE = d<Psi | SDl> <SDl | H | SDr> <SDr | Psi> / <Psi | SDn>^2 +
        #      <Psi | SDl> d<SDl | H | SDr> <SDr | Psi> / <Psi | SDn>^2 +
        #      <Psi | SDl> <SDl | H | SDr> d<SDr | Psi> / <Psi | SDn>^2 -
        #      2 d<Psi | SDn> <Psi | SDl> <SDl | H | SDr> <SDr | Psi> / <Psi | SDn>^3
        d_olp_l = wfn.get_overlap(sd_l, deriv=np.arange(6))
        d_olp_r = wfn.get_overlap(sd_r, deriv=np.arange(6))
        d_olp_n = wfn.get_overlap(sd_n, deriv=np.arange(6))
        # FIXME: hamiltonian does not support parameters right now, so cannot derivatize wrt it
        # d_integral = (ham.integrate_sd_sd(sd_l, sd_r, deriv=np.arange(6)))
        d_integral = 0.0
        assert np.allclose(
            test.get_energy_two_proj(sd_l, sd_r, sd_n, deriv=True),
            (
                d_olp_l * integral * olp_r / olp_n ** 2
                + olp_l * d_integral * olp_r / olp_n ** 2
                + olp_l * integral * d_olp_r / olp_n ** 2
                - 2 * d_olp_n * olp_l * integral * olp_r / olp_n ** 3
            ),
        )

    # list of sd
    for sds_l, sds_r, sds_n in it.permutations(it.permutations(sds, 2), 3):
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
        integral11 = (ham.integrate_sd_sd(sd_l1, sd_r1))
        integral12 = (ham.integrate_sd_sd(sd_l1, sd_r2))
        integral21 = (ham.integrate_sd_sd(sd_l2, sd_r1))
        integral22 = (ham.integrate_sd_sd(sd_l2, sd_r2))
        # <Psi| (|SDl1> <SDl1| + |SDl2> <SDl2| ) H (|SDr1> <SDr1| + |SDr2> <SDr2| ) |Psi>
        # = E <Psi| (|SDn1> <SDn1| + |SDn2> <SDn2|) | Psi>
        # <Psi|SDl1> <SDl1|H|SDr1> <SDr1|Psi> + <Psi|SDl2> <SDl2|H|SDr1> <SDr1|Psi> +
        # <Psi|SDl1> <SDl1|H|SDr2> <SDr2|Psi> + <Psi|SDl2> <SDl2|H|SDr2> <SDr2|Psi>
        # = E (<Psi|SDn1> <SDn1|Psi> + <Psi|SDn2> <SDn2|Psi>)
        # E = (<Psi|SDl1> <SDl1|H|SDr1> <SDr1|Psi> + <Psi|SDl2> <SDl2|H|SDr1> <SDr1|Psi> +
        #      <Psi|SDl1> <SDl1|H|SDr2> <SDr2|Psi> + <Psi|SDl2> <SDl2|H|SDr2> <SDr2|Psi>) /
        #     (<Psi|SDn1> <SDn1|Psi> + <Psi|SDn2> <SDn2|Psi>)
        assert np.allclose(
            test.get_energy_two_proj(sds_l, sds_r, sds_n),
            (
                (
                    olp_l1 * integral11 * olp_r1
                    + olp_l2 * integral21 * olp_r1
                    + olp_l1 * integral12 * olp_r2
                    + olp_l2 * integral22 * olp_r2
                )
                / (olp_n1 ** 2 + olp_n2 ** 2)
            ),
        )
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
        d_olp_l1 = wfn.get_overlap(sd_l1, deriv=np.arange(6))
        d_olp_r1 = wfn.get_overlap(sd_r1, deriv=np.arange(6))
        d_olp_n1 = wfn.get_overlap(sd_n1, deriv=np.arange(6))
        d_olp_l2 = wfn.get_overlap(sd_l2, deriv=np.arange(6))
        d_olp_r2 = wfn.get_overlap(sd_r2, deriv=np.arange(6))
        d_olp_n2 = wfn.get_overlap(sd_n2, deriv=np.arange(6))
        # FIXME: hamiltonian does not support parameters right now, so cannot derivatize wrt it
        # d_integral11 = (ham.integrate_sd_sd(sd_l1, sd_r1, deriv=np.arange(6)))
        # d_integral12 = (ham.integrate_sd_sd(sd_l2, sd_r1, deriv=np.arange(6)))
        # d_integral21 = (ham.integrate_sd_sd(sd_l1, sd_r2, deriv=np.arange(6)))
        # d_integral22 = (ham.integrate_sd_sd(sd_l2, sd_r2, deriv=np.arange(6)))
        d_integral11 = 0.0
        d_integral21 = 0.0
        d_integral12 = 0.0
        d_integral22 = 0.0
        assert np.allclose(
            test.get_energy_two_proj(sds_l, sds_r, sds_n, deriv=True),
            (
                (
                    d_olp_l1 * integral11 * olp_r1
                    + olp_l1 * d_integral11 * olp_r1
                    + olp_l1 * integral11 * d_olp_r1
                    + d_olp_l2 * integral21 * olp_r1
                    + olp_l2 * d_integral21 * olp_r1
                    + olp_l2 * integral21 * d_olp_r1
                    + d_olp_l1 * integral12 * olp_r2
                    + olp_l1 * d_integral12 * olp_r2
                    + olp_l1 * integral12 * d_olp_r2
                    + d_olp_l2 * integral22 * olp_r2
                    + olp_l2 * d_integral22 * olp_r2
                    + olp_l2 * integral22 * d_olp_r2
                )
                / (olp_n1 ** 2 + olp_n2 ** 2)
                - (2 * d_olp_n1 * olp_n1 + 2 * d_olp_n2 * olp_n2)
                * (
                    olp_l1 * integral11 * olp_r1
                    + olp_l2 * integral21 * olp_r1
                    + olp_l1 * integral12 * olp_r2
                    + olp_l2 * integral22 * olp_r2
                )
                / (olp_n1 ** 2 + olp_n2 ** 2) ** 2
            ),
        )


def test_baseschrodinger_get_energy_one_two_proj():
    wfn = CIWavefunction(4, 10)
    wfn.assign_params(np.random.rand(wfn.nparams))

    one_int = np.random.rand(5, 5)
    one_int = one_int + one_int.T
    two_int = np.random.rand(5, 5, 5, 5)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int
    ham = RestrictedChemicalHamiltonian(one_int, two_int)

    test = disable_abstract(BaseSchrodinger)(wfn, ham)

    from wfns.tools.sd_list import sd_list
    sds = sd_list(4, 10)
    assert np.allclose(test.get_energy_one_proj(sds), test.get_energy_two_proj(sds))
