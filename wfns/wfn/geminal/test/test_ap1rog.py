"""Test wfns.wavefunction.geminals.ap1rog.AP1roG."""
from nose.tools import assert_raises
import numpy as np
from wfns.tools import find_datafile
from wfns.wfn.geminal.ap1rog import AP1roG
from wfns.ham.senzero import SeniorityZeroHamiltonian
from wfns.objective.schrodinger.system_nonlinear import SystemEquations
from wfns.solver.system import least_squares
from wfns.objective.schrodinger.onesided_energy import OneSidedEnergy
from wfns.solver.equation import minimize


class TestAP1roG(AP1roG):
    """AP1roG that skips initialization."""
    def __init__(self):
        self._cache_fns = {}


def test_ap1rog_assign_ref_sd():
    """Test AP1roG.assign_ref_sd."""
    test = TestAP1roG()
    test.assign_nelec(4)
    test.assign_nspin(8)
    test.assign_ref_sd(None)
    assert test.ref_sd == 0b00110011
    test.assign_ref_sd(0b00110011)
    assert test.ref_sd == 0b00110011
    assert_raises(ValueError, test.assign_ref_sd, 0b00010001)
    assert_raises(ValueError, test.assign_ref_sd, 0b01110001)
    assert_raises(ValueError, test.assign_ref_sd, 0b01010011)
    # NOTE: multiple references is not supported
    assert_raises(TypeError, test.assign_ref_sd, [0b00110011, 0b01010101])
    # this is equivalent to
    assert_raises(TypeError, test.assign_ref_sd, [51, 85])
    # which means that the 51st and the 85th orbitals are occupied


def test_ap1rog_assign_ngem():
    """Test AP1roG.assign_ngem."""
    test = TestAP1roG()
    test.assign_nelec(4)
    assert_raises(NotImplementedError, test.assign_ngem, 3)


def test_ap1rog_assign_orbpairs():
    """Test AP1roG.assign_orbpairs."""
    test = TestAP1roG()
    test.assign_nelec(4)
    test.assign_nspin(6)
    test.assign_ref_sd()
    test.assign_orbpairs()
    assert test.dict_orbpair_ind == {(2, 5): 0}
    assert test.dict_ind_orbpair == {0: (2, 5)}


def test_ap1rog_template_params():
    """Test AP1roG.template_params."""
    test = TestAP1roG()
    test.assign_dtype(float)
    test.assign_nelec(4)
    test.assign_ngem(2)
    test.assign_nspin(6)
    test.assign_ref_sd()
    test.assign_orbpairs()
    assert np.allclose(test.template_params, np.zeros((2, 1)))
    test = TestAP1roG()
    test.assign_dtype(float)
    test.assign_nelec(6)
    test.assign_ngem(3)
    test.assign_nspin(20)
    test.assign_ref_sd()
    test.assign_orbpairs()
    assert np.allclose(test.template_params, np.zeros((3, 7)))


def test_ap1rog_get_overlap():
    """Test AP1roG.get_overlap."""
    test = TestAP1roG()
    test.assign_dtype(float)
    test.assign_nelec(4)
    test.assign_nspin(10)
    test.assign_memory()
    test.assign_ngem(2)
    test.assign_ref_sd()
    test.assign_orbpairs()
    test.assign_params(np.arange(6, dtype=float).reshape(2, 3))
    test.load_cache()
    assert test.get_overlap(0b0001100011) == 1.0
    assert test.get_overlap(0b0010100101) == 3.0
    assert test.get_overlap(0b0000001111) == 0.0
    assert test.get_overlap(0b0110001100) == 0*4 + 1*3
    # check derivatives
    test.assign_params(np.arange(6, dtype=float).reshape(2, 3))
    assert test.get_overlap(0b0001100011, deriv=0) == 0
    assert test.get_overlap(0b0001100011, deriv=1) == 0
    assert test.get_overlap(0b0001100011, deriv=99) == 0
    assert test.get_overlap(0b0011000110, deriv=0) == 1
    assert test.get_overlap(0b0011000110, deriv=3) == 0
    assert test.get_overlap(0b0011000110, deriv=99) == 0
    assert test.get_overlap(0b0010100101, deriv=0) == 0
    assert test.get_overlap(0b0010100101, deriv=3) == 1
    assert test.get_overlap(0b0110001100, deriv=99) == 0
    assert test.get_overlap(0b0110001100, deriv=0) == 4
    assert test.get_overlap(0b0110001100, deriv=1) == 3
    assert test.get_overlap(0b0110001100, deriv=3) == 1
    assert test.get_overlap(0b0110001100, deriv=4) == 0
    assert test.get_overlap(0b0110001100, deriv=99) == 0


def answer_ap1rog_h2_sto6g():
    """Find the AP1roG/STO-6G wavefunction by scanning through the coefficients."""
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/h2_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile('test/h2_hf_sto6g_oneint.npy'))
    two_int = np.load(find_datafile('test/h2_hf_sto6g_twoint.npy'))
    nuc_nuc = 0.71317683129
    ham = SeniorityZeroHamiltonian(one_int, two_int, orbtype='restricted', energy_nuc_nuc=nuc_nuc)

    ap1rog_ground = AP1roG(2, 4, ref_sd=0b0101)
    objective_ground = OneSidedEnergy(ap1rog_ground, ham)
    ap1rog_excited = AP1roG(2, 4, ref_sd=0b1010)
    objective_excited = OneSidedEnergy(ap1rog_excited, ham)

    # plot all possible values (within normalization constraint)
    xs = np.arange(-1, 1, 0.001)
    energies_ground = []
    energies_excited = []
    for i in xs:
        objective_ground.assign_params(np.array([i]))
        energies_ground.append(objective_ground.get_energy_one_proj([0b0101, 0b1010]))
    for i in xs:
        objective_excited.assign_params(np.array([i]))
        energies_excited.append(objective_excited.get_energy_one_proj([0b0101, 0b1010]))

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.scatter(xs, energies_ground)
    print(sorted(zip(xs, energies_ground), key=lambda x: x[1])[0])
    ax = fig.add_subplot(212)
    ax.scatter(xs, energies_excited)
    print(sorted(zip(xs, energies_excited), key=lambda x: x[1], reverse=True)[0])
    plt.show()

    # find exact minimum
    ap1rog_ground.assign_params(np.array([[-0.114]]))
    # manually set reference SD (need to get proper energy)
    res = minimize(objective_ground)
    print('Minimum energy')
    print(res)

    # find exact maximum
    def max_energy(params):
        """ Find maximum energy"""
        objective_excited.assign_params(params)
        energy = objective_excited.get_energy_one_proj([0b0101, 0b1010])
        return -energy
    import scipy.optimize
    res = scipy.optimize.minimize(max_energy, 0.114)
    print('Maximum energy')
    print(res)


def test_ap1rog_h2_sto6g_ground():
    """Test ground state AP1roG wavefunction using H2 with HF/STO-6G orbitals.

    Answers obtained from answer_ap1rog_h2_sto6g

    HF (Electronic) Energy : -1.838434256
    AP1roG Energy : -1.859089844148
    AP1roG Coeffs : [-0.113735924428061]

    """
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/h2_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile('test/h2_hf_sto6g_oneint.npy'))
    two_int = np.load(find_datafile('test/h2_hf_sto6g_twoint.npy'))
    nuc_nuc = 0.71317683129
    ham = SeniorityZeroHamiltonian(one_int, two_int, orbtype='restricted', energy_nuc_nuc=nuc_nuc)
    ap1rog = AP1roG(2, 4)

    # Solve system of equations
    objective = SystemEquations(ap1rog, ham, refwfn=[0b0101, 0b1010], constraints=[])
    results = least_squares(objective)
    assert np.allclose(results['energy'], -1.8590898441488894)


def test_ap1rog_h2_sto6g_excited():
    """Test excited state AP1roG wavefunction using H2 with HF/STO-6G orbital.

    Answers obtained from answer_ap1rog_h2_sto6g

    HF (Electronic) Energy : -1.838434256
    AP1roG Energy : -0.24166486974216012
    AP1roG Coeffs : [0.113735939809]

    """
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/h2_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile('test/h2_hf_sto6g_oneint.npy'))
    two_int = np.load(find_datafile('test/h2_hf_sto6g_twoint.npy'))
    nuc_nuc = 0.71317683129
    ham = SeniorityZeroHamiltonian(one_int, two_int, orbtype='restricted', energy_nuc_nuc=nuc_nuc)
    ap1rog = AP1roG(2, 4, ref_sd=0b1010)

    # Solve system of equations
    objective = SystemEquations(ap1rog, ham, refwfn=[0b0101, 0b1010], constraints=[])
    results = least_squares(objective)
    assert np.allclose(results['energy'], -0.24166486974216012)


# FIXME: answer should be brute force or external (should not depend on the code)
def answer_ap1rog_h2_631gdp():
    """Find the AP1roG/6-31G** wavefunction by scanning through the coefficients."""
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/h2_hf_631gdp.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile('test/h2_hf_631gdp_oneint.npy'))
    two_int = np.load(find_datafile('test/h2_hf_631gdp_twoint.npy'))
    nuc_nuc = 0.71317683129
    ham = SeniorityZeroHamiltonian(one_int, two_int, orbtype='restricted', energy_nuc_nuc=nuc_nuc)
    ap1rog = AP1roG(2, 20)

    objective = OneSidedEnergy(ap1rog, ham)
    results = minimize(objective)
    print('Minimum energy')
    print(results)


def test_ap1rog_h2_631gdp():
    """Test ground state AP1roG wavefunction using H2 with HF/6-31G** orbitals.

    Answers obtained from answer_ap1rog_h2_631gdp

    HF (Electronic) Energy : -1.838434256
    AP1roG Energy : -1.8696828608304892
    AP1roG Coeffs : [-0.05949796, -0.05454253, -0.03709503, -0.02899231, -0.02899231, -0.01317386,
                     -0.00852702, -0.00852702, -0.00517996]
    """
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/h2_hf_631gdp.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile('test/h2_hf_631gdp_oneint.npy'))
    two_int = np.load(find_datafile('test/h2_hf_631gdp_twoint.npy'))
    nuc_nuc = 0.71317683129
    ham = SeniorityZeroHamiltonian(one_int, two_int, orbtype='restricted', energy_nuc_nuc=nuc_nuc)
    ap1rog = AP1roG(2, 20)
    full_sds = (0b00000000010000000001, 0b00000000100000000010, 0b00000001000000000100,
                0b00000010000000001000, 0b00000100000000010000, 0b00001000000000100000,
                0b00010000000001000000, 0b00100000000010000000, 0b01000000000100000000,
                0b10000000001000000000)

    # Solve system of equations
    objective = SystemEquations(ap1rog, ham, refwfn=full_sds, constraints=[])
    results = least_squares(objective)
    assert np.allclose(results['energy'], -1.8696828608304892)


# FIXME: answer should be brute force or external (should not depend on the code)
def answer_ap1rog_lih_sto6g():
    """Find the AP1roG/6-31G** wavefunction by scanning through the coefficients."""
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/lih_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = (np.load(find_datafile('test/lih_hf_sto6g_oneint.npy')), )
    two_int = (np.load(find_datafile('test/lih_hf_sto6g_twoint.npy')), )
    nuc_nuc = 0.995317634356
    ham = SeniorityZeroHamiltonian(one_int, two_int, orbtype='restricted', energy_nuc_nuc=nuc_nuc)
    ap1rog = AP1roG(4, 12)

    objective = OneSidedEnergy(ap1rog, ham)
    results = minimize(objective)
    print('Minimum energy')
    print(results)


def test_ap1rog_lih_sto6g():
    """Test ground state AP1roG wavefunction using LiH with HF/STO-6G orbitals.

    Answers obtained from answer_ap1rog_lih_sto6g

    HF (Electronic) Energy : -8.9472891719
    AP1roG Energy : -8.963531105243506
    AP1roG Coeffs : [-4.18979612e-03, -1.71684359e-03, -1.71684359e-03, -1.16341258e-03,
                     -9.55342486e-03, -2.90031933e-02, -2.90031933e-02, -1.17012605e-01]
    """
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/lih_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile('test/lih_hf_sto6g_oneint.npy'))
    two_int = np.load(find_datafile('test/lih_hf_sto6g_twoint.npy'))
    nuc_nuc = 0.995317634356
    ham = SeniorityZeroHamiltonian(one_int, two_int, orbtype='restricted', energy_nuc_nuc=nuc_nuc)
    ap1rog = AP1roG(4, 12)
    full_sds = (0b000011000011, 0b000101000101, 0b001001001001, 0b010001010001, 0b100001100001,
                0b000110000110, 0b001010001010, 0b010010010010, 0b100010100010, 0b001100001100,
                0b010100010100, 0b100100100100, 0b011000011000, 0b101000101000, 0b110000110000)

    # Solve system of equations
    objective = SystemEquations(ap1rog, ham, refwfn=full_sds, constraints=[])
    results = least_squares(objective)
    assert np.allclose(results['energy'], -8.963531105243506)
