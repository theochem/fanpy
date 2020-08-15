"""Code generating script."""
import textwrap

from fanpy.scripts.utils import check_inputs, parser


def make_script(
    nelec,
    one_int_file,
    two_int_file,
    wfn_type,
    nuc_nuc=0.0,
    optimize_orbs=False,
    pspace_exc=(1, 2),
    objective="projected",
    solver="least_squares",
    solver_kwargs=None,
    wfn_kwargs=None,
    ham_noise=0.0,
    wfn_noise=0.0,
    load_orbs=None,
    load_ham=None,
    load_ham_um=None,
    load_wfn=None,
    save_chk="",
    filename=None,
    memory=None,
):
    """Make a script for running calculations.

    Parameters
    ----------
    nelec : int
        Number of electrons.
    one_int_file : str
        Path to the one electron integrals (for restricted orbitals).
        One electron integrals should be stored as a numpy array of dimension (nspin/2, nspin/2).
    two_int_file : str
        Path to the two electron integrals (for restricted orbitals).
        Two electron integrals should be stored as a numpy array of dimension
        (nspin/2, nspin/2, nspin/2, nspin/2).
    wfn_type : str
        Type of wavefunction.
        One of `ci_pairs`, `cisd`, `fci`, `doci`, `mps`, `determinant-ratio`, `ap1rog`, `apr2g`,
        `apig`, `apsetg`, or `apg`.
    nuc_nuc : float
        Nuclear-nuclear repulsion energy.
    optimize_orbs : bool
        If True, orbitals are optimized.
        If False, orbitals are not optimized.
        By default, orbitals are not optimized.
        Not compatible with solvers that require a gradient (everything except cma).
    pspace_exc : list of int
        Orders of excitations that will be used to build the projection space.
    objective : str
        Form of the Schrodinger equation that will be solved.
        Use `projected` to solve the Schrodinger equation as a system of equations.
        Use `least_squares` to solve the Schrodinger equation as a squared sum of the system of
        equations.
        Use `variational` to solve the energy variationally.
        Use `one_energy` to solve the energy projected on one side..
        Must be one of `projected`, `least_squares`, `variational`, or `one_energy`.
    solver : str
        Solver that will be used to solve the Schrodinger equation.
        Keyword `cma` uses Covariance Matrix Adaptation - Evolution Strategy (CMA-ES).
        Keyword `diag` results in diagonalizing the CI matrix.
        Keyword `minimize` uses the BFGS algorithm.
        Keyword `least_squares` uses the Trust Region Reflective Algorithm.
        Keyword `root` uses the MINPACK hybrd routine.
        Must be one of `cma`, `diag`, `least_squares`, or `root`.
        Must be compatible with the objective.
    solver_kwargs : {str, None}
        Keyword arguments for the solver.
    wfn_kwargs : {str, None}
        Keyword arguments for the wavefunction.
    ham_noise : float
        Scale of the noise to be applied to the Hamiltonian parameters.
        The noise is generated using a uniform distribution between -1 and 1.
        By default, no noise is added.
    wfn_noise : bool
        Scale of the noise to be applied to the wavefunction parameters.
        The noise is generated using a uniform distribution between -1 and 1.
        By default, no noise is added.
    load_orbs : str
        Numpy file of the orbital transformation matrix that will be applied to the initial
        Hamiltonian.
        If the initial Hamiltonian parameters are provided, the orbitals will be transformed
        afterwards.
    load_ham : str
        Numpy file of the Hamiltonian parameters that will overwrite the parameters of the initial
        Hamiltonian.
    load_ham_um : str
        Numpy file of the Hamiltonian parameters that will overwrite the unitary matrix of the
        initial Hamiltonian.
    load_wfn : str
        Numpy file of the wavefunction parameters that will overwrite the parameters of the initial
        wavefunction.
    save_chk : str
        Name of the Numpy file that will store the chkpoint of the objective.
    filename : str
        Name of the script
        By default, the script is printed.
        If `-1` is given, then the script is returned as a string.
        Otherwise, the given string is treated as the name of the file.
    memory : None
        Memory available to run calculations.

    """
    # check inputs
    check_inputs(
        nelec,
        one_int_file,
        two_int_file,
        wfn_type,
        pspace_exc,
        objective,
        solver,
        nuc_nuc,
        optimize_orbs=optimize_orbs,
        load_orbs=load_orbs,
        load_ham=load_ham,
        load_ham_um=load_ham_um,
        load_wfn=load_wfn,
        save_chk=save_chk,
        filename=filename if filename != -1 else None,
        memory=memory,
        solver_kwargs=solver_kwargs,
        wfn_kwargs=wfn_kwargs,
        ham_noise=ham_noise,
        wfn_noise=wfn_noise,
    )

    imports = ["numpy as np", "os", "sys"]
    from_imports = []

    wfn_type = wfn_type.lower()
    if wfn_type == "ci_pairs":
        from_imports.append(("fanpy.wfn.ci.ci_pairs", "CIPairs"))
        wfn_name = "CIPairs"
        if wfn_kwargs is None:
            wfn_kwargs = ""
    elif wfn_type == "cisd":
        from_imports.append(("fanpy.wfn.ci.cisd", "CISD"))
        wfn_name = "CISD"
        if wfn_kwargs is None:
            wfn_kwargs = ""
    elif wfn_type == "fci":
        from_imports.append(("fanpy.wfn.ci.fci", "FCI"))
        wfn_name = "FCI"
        if wfn_kwargs is None:
            wfn_kwargs = "spin=None"
    elif wfn_type == "doci":
        from_imports.append(("fanpy.wfn.ci.doci", "DOCI"))
        wfn_name = "DOCI"
        if wfn_kwargs is None:
            wfn_kwargs = ""
    elif wfn_type == "mps":
        from_imports.append(("fanpy.wfn.network.mps", "MatrixProductState"))
        wfn_name = "MatrixProductState"
        if wfn_kwargs is None:
            wfn_kwargs = "dimension=None"
    elif wfn_type == "determinant-ratio":
        from_imports.append(("fanpy.wfn.quasiparticle.det_ratio", "DeterminantRatio"))
        wfn_name = "DeterminantRatio"
        if wfn_kwargs is None:
            wfn_kwargs = "numerator_mask=None"
    elif wfn_type == "ap1rog":
        from_imports.append(("fanpy.wfn.geminal.ap1rog", "AP1roG"))
        wfn_name = "AP1roG"
        if wfn_kwargs is None:
            wfn_kwargs = "ref_sd=None, ngem=None"
    elif wfn_type == "apr2g":
        from_imports.append(("fanpy.wfn.geminal.apr2g", "APr2G"))
        wfn_name = "APr2G"
        if wfn_kwargs is None:
            wfn_kwargs = "ngem=None"
    elif wfn_type == "apig":
        from_imports.append(("fanpy.wfn.geminal.apig", "APIG"))
        wfn_name = "APIG"
        if wfn_kwargs is None:
            wfn_kwargs = "ngem=None"
    elif wfn_type == "apsetg":
        from_imports.append(("fanpy.wfn.geminal.apsetg", "BasicAPsetG"))
        wfn_name = "BasicAPsetG"
        if wfn_kwargs is None:
            wfn_kwargs = "ngem=None"
    elif wfn_type == "apg":  # pragma: no branch
        from_imports.append(("fanpy.wfn.geminal.apg", "APG"))
        wfn_name = "APG"
        if wfn_kwargs is None:
            wfn_kwargs = "ngem=None"

    if wfn_name in ["DOCI", "CIPairs"]:
        from_imports.append(("fanpy.ham.senzero", "SeniorityZeroHamiltonian"))
        ham_name = "SeniorityZeroHamiltonian"
    else:
        from_imports.append(("fanpy.ham.restricted_chemical", "RestrictedMolecularHamiltonian"))
        ham_name = "RestrictedMolecularHamiltonian"

    from_imports.append(("fanpy.tools.sd_list", "sd_list"))

    if objective == "projected":
        from_imports.append(("fanpy.eqn.projected", "ProjectedSchrodinger"))
    elif objective == "least_squares":
        from_imports.append(("fanpy.eqn.least_squares", "LeastSquaresEquations"))
    elif objective == "variational":
        from_imports.append(("fanpy.eqn.energy_twoside", "EnergyTwoSideProjection"))
    elif objective == "one_energy":  # pragma: no branch
        from_imports.append(("fanpy.eqn.energy_oneside", "EnergyOneSideProjection"))

    if solver == "cma":
        from_imports.append(("fanpy.solver.equation", "cma"))
        solver_name = "cma"
        if solver_kwargs is None:
            solver_kwargs = (
                "sigma0=0.01, options={'ftarget': None, 'timeout': np.inf, "
                "'tolfun': 1e-11, 'verb_filenameprefix': 'outcmaes', 'verb_log': 1}"
            )
    elif solver == "diag":
        from_imports.append(("fanpy.solver.ci", "brute"))
        solver_name = "brute"
    elif solver == "minimize":
        from_imports.append(("fanpy.solver.equation", "minimize"))
        solver_name = "minimize"
        if solver_kwargs is None:
            solver_kwargs = "method='BFGS', jac=objective.gradient, options={'gtol': 1e-8}"
    elif solver == "least_squares":
        from_imports.append(("fanpy.solver.system", "least_squares"))
        solver_name = "least_squares"
        if solver_kwargs is None:
            solver_kwargs = (
                "xtol=1.0e-15, ftol=1.0e-15, gtol=1.0e-15, "
                "max_nfev=1000*objective.active_params.size, jac=objective.jacobian"
            )
    elif solver == "root":  # pragma: no branch
        from_imports.append(("fanpy.solver.system", "root"))
        solver_name = "root"
        if solver_kwargs is None:
            solver_kwargs = "method='hybr', jac=objective.jacobian, options={'xtol': 1.0e-9}"

    if memory is not None:
        memory = "'{}'".format(memory)

    output = ""
    for i in imports:
        output += "import {}\n".format(i)
    for key, val in from_imports:
        output += "from {} import {}\n".format(key, val)
    output += "\n\n"

    output += "# Number of electrons\n"
    output += "nelec = {:d}\n".format(nelec)
    output += "print('Number of Electrons: {}'.format(nelec))\n"
    output += "\n"

    output += "# One-electron integrals\n"
    output += "one_int_file = '{}'\n".format(one_int_file)
    output += "one_int = np.load(one_int_file)\n"
    output += (
        "print('One-Electron Integrals: {{}}'.format(os.path.abspath(one_int_file)))"  # noqa: F523
        "\n".format(one_int_file)
    )
    output += "\n"

    output += "# Two-electron integrals\n"
    output += "two_int_file = '{}'\n".format(two_int_file)
    output += "two_int = np.load(two_int_file)\n"
    output += (
        "print('Two-Electron Integrals: {{}}'.format(os.path.abspath(two_int_file)))"  # noqa: F523
        "\n".format(two_int_file)
    )
    output += "\n"

    output += "# Number of spin orbitals\n"
    output += "nspin = one_int.shape[0] * 2\n"
    output += "print('Number of Spin Orbitals: {}'.format(nspin))\n"
    output += "\n"

    output += "# Nuclear-nuclear repulsion\n"
    output += "nuc_nuc = {}\n".format(nuc_nuc)
    output += "print('Nuclear-nuclear repulsion: {}'.format(nuc_nuc))\n"
    output += "\n"

    if load_wfn is not None:
        output += "# Load wavefunction parameters\n"
        output += "wfn_params_file = '{}'\n".format(load_wfn)
        output += "wfn_params = np.load(wfn_params_file)\n"
        output += "print('Load wavefunction parameters: {}'"
        output += ".format(os.path.abspath(wfn_params_file)))\n"
        output += "\n"
        wfn_params = "wfn_params"
    else:
        wfn_params = "None"

    output += "# Initialize wavefunction\n"
    wfn_init1 = "wfn = {}(".format(wfn_name)
    wfn_init2 = "nelec, nspin, params={}, memory={}, {})\n".format(wfn_params, memory, wfn_kwargs)
    output += "\n".join(
        textwrap.wrap(wfn_init1 + wfn_init2, width=100, subsequent_indent=" " * len(wfn_init1))
    )
    output += "\n"
    if wfn_noise not in [0, None]:
        output += (
            "wfn.assign_params(wfn.params + "
            "{} * 2 * (np.random.rand(*wfn.params.shape) - 0.5))\n".format(wfn_noise)
        )
    output += "print('Wavefunction: {}')\n".format(wfn_name)
    output += "\n"

    output += "# Initialize Hamiltonian\n"
    ham_init1 = "ham = {}(".format(ham_name)
    ham_init2 = "one_int, two_int)\n"
    output += "\n".join(
        textwrap.wrap(ham_init1 + ham_init2, width=100, subsequent_indent=" " * len(ham_init1))
    )

    if load_ham_um is not None:
        output += "# Load unitary matrix of the Hamiltonian\n"
        output += "ham_um_file = '{}'\n".format(load_ham_um)
        output += "ham_um = np.load(ham_um_file)\n"
        output += "print('Load unitary matrix of the Hamiltonian: {}'"
        output += ".format(os.path.abspath(ham_um_file)))\n"
        if ham_name == "UnrestrictedMolecularHamiltonian":  # pragma: no cover
            output += "ham._prev_unitary_alpha = ham_um[0]\n"
            output += "ham._prev_unitary_beta = ham_um[1]\n"
        else:
            output += "ham._prev_unitary = ham_um\n"
        output += "\n"

    if load_ham is not None:
        output += "# Load Hamiltonian parameters (orbitals)\n"
        output += "ham_params_file = '{}'\n".format(load_ham)
        output += "ham_params = np.load(ham_params_file)\n"
        output += "print('Load Hamiltonian parameters: {}'"
        output += ".format(os.path.abspath(ham_params_file)))\n"
        if load_ham_um:
            output += "ham._prev_params = ham_params\n"
        output += "ham.assign_params(ham_params)\n"
        output += "\n"

    output += "\n"
    if ham_noise not in [0, None]:
        output += (
            "ham.assign_params(ham.params + "
            "{} * 2 * (np.random.rand(*ham.params.shape) - 0.5))\n".format(ham_noise)
        )
    output += "print('Hamiltonian: {}')\n".format(ham_name)
    output += "\n"

    if pspace_exc is None:  # pragma: no cover
        pspace = "[1, 2]"
    else:
        pspace = str([int(i) for i in pspace_exc])
    output += "# Projection space\n"
    pspace1 = "pspace = sd_list("
    pspace2 = (
        "nelec, nspin, num_limit=None, exc_orders={}, spin=None, "
        "seniority=wfn.seniority)\n".format(pspace)
    )
    output += "\n".join(
        textwrap.wrap(pspace1 + pspace2, width=100, subsequent_indent=" " * len(pspace1))
    )
    output += "\n"
    output += "print('Projection space (orders of excitations): {}')\n".format(pspace)
    output += "\n"

    output += "# Select parameters that will be optimized\n"
    if optimize_orbs:
        output += (
            "param_selection = [(wfn, np.ones(wfn.nparams, dtype=bool)), "
            "(ham, np.ones(ham.nparams, dtype=bool))]\n"
        )
    else:
        output += "param_selection = [(wfn, np.ones(wfn.nparams, dtype=bool))]\n"
    output += "\n"

    output += "# Initialize objective\n"
    if objective == "projected":
        objective1 = "objective = ProjectedSchrodinger("
        objective2 = (
            "wfn, ham, param_selection=param_selection, "
            "pspace=pspace, refwfn=None, energy_type='compute', "
            "energy=None, constraints=None, eqn_weights=None)\n"
        )
    elif objective == "least_squares":
        objective1 = "objective = LeastSquaresEquations("
        objective2 = (
            "wfn, ham, param_selection=param_selection, "
            "pspace=pspace, refwfn=None, energy_type='compute', "
            "energy=None, constraints=None, eqn_weights=None)\n"
        )
    elif objective == "variational":
        objective1 = "objective = EnergyTwoSideProjection("
        objective2 = (
            "wfn, ham, param_selection=param_selection, "
            "pspace_l=pspace, pspace_r=pspace, pspace_n=pspace)\n"
        )
    elif objective == "one_energy":  # pragma: no branch
        objective1 = "objective = EnergyOneSideProjection("
        objective2 = "wfn, ham, param_selection=param_selection, " "refwfn=pspace)\n"
    output += "\n".join(
        textwrap.wrap(objective1 + objective2, width=100, subsequent_indent=" " * len(objective1))
    )
    output += "\n"
    output += "objective.tmpfile = '{}'".format(save_chk)
    output += "\n\n"

    output += "# Solve\n"
    if solver_name == "brute":
        output += "results = brute(wfn, ham, save_file='')\n"
        output += "print('Optimizing wavefunction: brute force diagonalization of CI matrix')\n"
    else:
        results1 = "results = {}(".format(solver_name)
        results2 = "objective, {})\n".format(solver_kwargs)
        output += "print('Optimizing wavefunction: {} solver')\n".format(solver_name)
        output += "\n".join(
            textwrap.wrap(results1 + results2, width=100, subsequent_indent=" " * len(results1))
        )
        output += "\n"
    output += "\n"

    output += "# Results\n"
    output += "if results['success']:\n"
    output += "    print('Optimization was successful')\n"
    output += "else:\n"
    output += "    print('Optimization was not successful: {}'.format(results['message']))\n"
    output += "print('Final Electronic Energy: {}'.format(results['energy']))\n"
    output += "print('Final Total Energy: {}'.format(results['energy'] + nuc_nuc))\n"

    if filename is None:  # pragma: no cover
        print(output)
    # NOTE: number was used instead of string (eg. 'return') to prevent problems arising from
    #       accidentally using the reserved string/keyword.
    elif filename == -1:
        return output
    else:
        with open(filename, "w") as f:
            f.write(output)


def main():  # pragma: no cover
    """Run script for run_calc using arguments obtained via argparse."""
    parser.description = "Optimize a wavefunction and/or Hamiltonian."
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        required=False,
        help="Name of the file that contains the output of the script.",
    )
    args = parser.parse_args()
    make_script(
        args.nelec,
        args.one_int_file,
        args.two_int_file,
        args.wfn_type,
        nuc_nuc=args.nuc_nuc,
        optimize_orbs=args.optimize_orbs,
        pspace_exc=args.pspace_exc,
        objective=args.objective,
        solver=args.solver,
        solver_kwargs=args.solver_kwargs,
        wfn_kwargs=args.wfn_kwargs,
        ham_noise=args.ham_noise,
        wfn_noise=args.wfn_noise,
        load_orbs=args.load_orbs,
        load_ham=args.load_ham,
        load_ham_um=args.load_ham_um,
        load_wfn=args.load_wfn,
        save_chk=args.save_chk,
        filename=args.filename,
        memory=args.memory,
    )
