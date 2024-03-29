"""Code generating script."""
import textwrap

from wfns.scripts.utils import check_inputs, parser, parser_add_arguments


# FIXME: not tested
def make_script(
    nelec,
    nspin,
    one_int_file,
    two_int_file,
    wfn_type,
    nuc_nuc=None,
    optimize_orbs=False,
    pspace_exc=None,
    objective=None,
    solver=None,
    solver_kwargs=None,
    wfn_kwargs=None,
    ham_noise=None,
    wfn_noise=None,
    load_orbs=None,
    load_ham=None,
    load_wfn=None,
    load_chk=None,
    save_orbs=None,
    save_ham=None,
    save_wfn=None,
    save_chk=None,
    filename=None,
    memory=None,
):
    """Make a script for running calculations.

    Parameters
    ----------
    nelec : int
        Number of electrons.
    nspin : int
        Number of spin orbitals.
    one_int_file : str
        Path to the one electron integrals (for restricted orbitals).
        One electron integrals should be stored as a numpy array of dimension (nspin/2, nspin/2).
    two_int_file : str
        Path to the two electron integrals (for restricted orbitals).
        Two electron integrals should be stored as a numpy array of dimension
        (nspin/2, nspin/2, nspin/2, nspin/2).
    wfn_type : str
        Type of wavefunction.
        One of `fci`, `doci`, `mps`, `determinant-ratio`, `ap1rog`, `apr2g`, `apig`, `apsetg`, and
        `apg`.
    nuc_nuc : float
        Nuclear-nuclear repulsion energy.
        Default is `0.0`.
    optimize_orbs : bool
        If True, orbitals are optimized.
        If False, orbitals are not optimized.
        By default, orbitals are not optimized.
        Not compatible with solvers that require a gradient (everything except cma).
    pspace_exc : list of int
        Orders of excitations that will be used to build the projection space.
        Default is first and second order excitations of the HF ground state.
    objective : str
        Form of the Schrodinger equation that will be solved.
        Use `system` to solve the Schrodinger equation as a system of equations.
        Use `least_squares` to solve the Schrodinger equation as a squared sum of the system of
        equations.
        Use `variational` to solve the Schrodinger equation variationally.
        Must be one of `system`, `least_squares`, and `variational`.
        By default, the Schrodinger equation is solved as system of equations.
    solver : str
        Solver that will be used to solve the Schrodinger equation.
        Keyword `cma` uses Covariance Matrix Adaptation - Evolution Strategy (CMA-ES).
        Keyword `diag` results in diagonalizing the CI matrix.
        Keyword `minimize` uses the BFGS algorithm.
        Keyword `least_squares` uses the Trust Region Reflective Algorithm.
        Keyword `root` uses the MINPACK hybrd routine.
        Must be one of `cma`, `diag`, `least_squares`, or `root`.
        Must be compatible with the objective.
    solver_kwargs : str
        Keyword arguments for the solver.
    wfn_kwargs : str
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
    load_wfn : str
        Numpy file of the wavefunction parameters that will overwrite the parameters of the initial
        wavefunction.
    load_chk : str
        Numpy file of the chkpoint file for the objective.
    save_orbs : str
        Name of the Numpy file that will store the last orbital transformation matrix that was
        applied to the Hamiltonian (after a successful optimization).
    save_ham : str
        Name of the Numpy file that will store the Hamiltonian parameters after a successful
        optimization.
    save_wfn : str
        Name of the Numpy file that will store the wavefunction parameters after a successful
        optimization.
    save_chk : str
        Name of the Numpy file that will store the chkpoint of the objective.
    filename : {str, -1, None}
        Name of the file that will store the output.
        By default, the script is printed.
        If `-1` is given, then the script is returned as a string.
        Otherwise, the given string is treated as the name of the file.
    memory : str
        Memory available to run the calculation.

    """
    # check inputs
    check_inputs(
        nelec,
        nspin,
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
        load_wfn=load_wfn,
        load_chk=load_chk,
        save_orbs=save_orbs,
        save_ham=save_ham,
        save_wfn=save_wfn,
        save_chk=save_chk,
        filename=filename if filename != -1 else None,
        memory=memory,
        solver_kwargs=solver_kwargs,
        wfn_kwargs=wfn_kwargs,
        ham_noise=ham_noise,
        wfn_noise=wfn_noise,
    )

    imports = ["numpy as np", "os"]
    from_imports = []

    wfn_type = wfn_type.lower()
    if wfn_type == "fci":
        from_imports.append(("wfns.wfn.ci.fci", "FCI"))
        wfn_name = "FCI"
        if wfn_kwargs is None:
            wfn_kwargs = "spin=None"
    elif wfn_type == "doci":
        from_imports.append(("wfns.wfn.ci.doci", "DOCI"))
        wfn_name = "DOCI"
    elif wfn_type == "mps":
        from_imports.append(("wfns.wfn.network.mps", "MatrixProductState"))
        wfn_name = "MatrixProductState"
        if wfn_kwargs is None:
            wfn_kwargs = "dimension=None"
    elif wfn_type == "determinant-ratio":
        from_imports.append(("wfns.wfn.quasiparticle.det_ratio", "DeterminantRatio"))
        wfn_name = "DeterminantRatio"
        if wfn_kwargs is None:
            wfn_kwargs = "numerator_mask=None"
    elif wfn_type == "ap1rog":
        from_imports.append(("wfns.wfn.geminal.ap1rog", "AP1roG"))
        wfn_name = "AP1roG"
        if wfn_kwargs is None:
            wfn_kwargs = "ref_sd=None, ngem=None"
    elif wfn_type == "apr2g":
        from_imports.append(("wfns.wfn.geminal.apr2g", "APr2G"))
        wfn_name = "APr2G"
        if wfn_kwargs is None:
            wfn_kwargs = "ngem=None"
    elif wfn_type == "apig":
        from_imports.append(("wfns.wfn.geminal.apig", "APIG"))
        wfn_name = "APIG"
        if wfn_kwargs is None:
            wfn_kwargs = "ngem=None"
    elif wfn_type == "apsetg":
        from_imports.append(("wfns.wfn.geminal.apsetg", "BasicAPsetG"))
        wfn_name = "BasicAPsetG"
        if wfn_kwargs is None:
            wfn_kwargs = "ngem=None"
    elif wfn_type == "apg":
        from_imports.append(("wfns.wfn.geminal.apg", "APG"))
        wfn_name = "APG"
        if wfn_kwargs is None:
            wfn_kwargs = "ngem=None"

    if wfn_name == "DOCI":
        from_imports.append(("wfns.ham.senzero", "SeniorityZeroHamiltonian"))
        ham_name = "SeniorityZeroHamiltonian"
    else:
        from_imports.append(("wfns.ham.restricted_chemical", "RestrictedChemicalHamiltonian"))
        ham_name = "RestrictedChemicalHamiltonian"

    from_imports.append(("wfns.backend.sd_list", "sd_list"))

    if objective == "system":
        from_imports.append(("wfns.objective.schrodinger.system_nonlinear", "SystemEquations"))
    elif objective == "least_squares":
        from_imports.append(("wfns.objective.schrodinger.least_squares", "LeastSquaresEquations"))
    elif objective == "variational":
        from_imports.append(("wfns.objective.schrodinger.twosided_energy", "TwoSidedEnergy"))

    if solver == "cma":
        from_imports.append(("wfns.solver.equation", "cma"))
        solver_name = "cma"
        if solver_kwargs is None:
            solver_kwargs = (
                "sigma0=0.01, options={'ftarget': None, 'timeout': np.inf, "
                "'tolfun': 1e-11, 'verb_filenameprefix': 'outcmaes', 'verb_log': 0}"
            )
    elif solver == "diag":
        from_imports.append(("wfns.solver.ci", "brute"))
        solver_name = "brute"
    elif solver == "minimize":
        from_imports.append(("wfns.solver.equation", "minimize"))
        solver_name = "minimize"
        if solver_kwargs is None:
            solver_kwargs = "method='BFGS', jac=objective.gradient, options={'gtol': 1e-8}"
    elif solver == "least_squares":
        from_imports.append(("wfns.solver.system", "least_squares"))
        solver_name = "least_squares"
        if solver_kwargs is None:
            solver_kwargs = (
                "xtol=1.0e-15, ftol=1.0e-15, gtol=1.0e-15, "
                "max_nfev=1000*objective.params.size, jac=objective.jacobian"
            )
    elif solver == "root":
        from_imports.append(("wfns.solver.system", "root"))
        solver_name = "root"
        if solver_kwargs is None:
            solver_kwargs = "method='hybr', jac=objective.jacobian, options={'xtol': 1.0e-9}"

    if save_orbs is not None:
        from_imports.append(("wfns.backend.math_tools", "unitary_matrix"))

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

    output += "# Number of spin orbitals\n"
    output += "nspin = {:d}\n".format(nspin)
    output += "print('Number of Spin Orbitals: {}'.format(nspin))\n"
    output += "\n"

    output += "# One-electron integrals\n"
    output += "one_int_file = '{}'\n".format(one_int_file)
    output += "one_int = np.load(one_int_file)\n"
    output += (
        "print('One-Electron Integrals: {{}}'.format(os.path.abspath(one_int_file)))\n"
        "".format(one_int_file)
    )
    output += "\n"

    output += "# Two-electron integrals\n"
    output += "two_int_file = '{}'\n".format(two_int_file)
    output += "two_int = np.load(two_int_file)\n"
    output += (
        "print('Two-Electron Integrals: {{}}'.format(os.path.abspath(two_int_file)))\n"
        "".format(two_int_file)
    )
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

    if load_ham is not None:
        output += "# Load Hamiltonian parameters (orbitals)\n"
        output += "ham_params_file = '{}'\n".format(load_ham)
        output += "ham_params = np.load(ham_params_file)\n"
        output += "print('Load Hamiltonian parameters: {}'"
        output += ".format(os.path.abspath(ham_params_file)))\n"
        output += "\n"
        ham_params = "ham_params"
    else:
        ham_params = "None"

    output += "# Initialize Hamiltonian\n"
    ham_init1 = "ham = {}(".format(ham_name)
    ham_init2 = "one_int, two_int, energy_nuc_nuc=nuc_nuc, params={})\n".format(ham_params)
    output += "\n".join(
        textwrap.wrap(ham_init1 + ham_init2, width=100, subsequent_indent=" " * len(ham_init1))
    )
    output += "\n"
    if ham_noise not in [0, None]:
        output += (
            "ham.assign_params(ham.params + "
            "{} * 2 * (np.random.rand(*ham.params.shape) - 0.5))\n".format(ham_noise)
        )
    output += "print('Hamiltonian: {}')\n".format(ham_name)
    output += "\n"

    if load_orbs:
        output += "# Rotate orbitals\n"
        output += "orb_matrix_file = '{}'\n".format(load_orbs)
        output += "orb_matrix = np.load(orb_matrix_file)\n"
        output += "ham.orb_rotate_matrix(orb_matrix)\n"
        output += "print('Rotate orbitals from {}'.format(os.path.abspath(orb_matrix_file)))\n"
        output += "\n"

    if pspace_exc is None:
        pspace = "[1, 2]"
    else:
        pspace = str([int(i) for i in pspace_exc])
    output += "# Projection space\n"
    pspace1 = "pspace = sd_list("
    pspace2 = (
        "nelec, nspin//2, num_limit=None, exc_orders={}, spin=None, "
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

    if save_chk is None:
        save_chk = ""

    output += "# Initialize objective\n"
    if objective == "system":
        objective1 = "objective = SystemEquations("
        objective2 = (
            "wfn, ham, param_selection=param_selection, "
            "tmpfile='{}', pspace=pspace, refwfn=None, energy_type='compute', "
            "energy=None, constraints=None, eqn_weights=None)\n".format(save_chk)
        )
    elif objective == "least_squares":
        objective1 = "objective = LeastSquaresEquations("
        objective2 = (
            "wfn, ham, param_selection=param_selection, "
            "tmpfile='{}', pspace=pspace, refwfn=None, energy_type='compute', "
            "energy=None, constraints=None, eqn_weights=None)\n".format(save_chk)
        )
    elif objective == "variational":
        objective1 = "objective = TwoSidedEnergy("
        objective2 = (
            "wfn, ham, param_selection=param_selection, "
            "tmpfile='{}', pspace_l=pspace, pspace_r=pspace, pspace_n=pspace)\n"
            "".format(save_chk)
        )
    output += "\n".join(
        textwrap.wrap(objective1 + objective2, width=100, subsequent_indent=" " * len(objective1))
    )
    output += "\n\n"

    if load_chk is not None:
        output += "# Load checkpoint\n"
        output += "chk_point_file = '{}'\n".format(load_chk)
        output += "chk_point = np.load(chk_point_file)\n"
        output += "objective.assign_params(chk_point)\n"
        output += "print('Load checkpoint file: {}'.format(os.path.abspath(chk_point_file)))\n"
        output += "\n"

    if save_chk is None:
        save_chk = ""
    output += "# Solve\n"
    if solver_name == "brute":
        output += "results = brute(wfn, ham, save_file='')\n"
        output += "print('Optimizing wavefunction: brute force diagonalization of CI matrix')\n"
    else:
        results1 = "results = {}(".format(solver_name)
        results2 = "objective, save_file='{}', {})\n".format(save_chk, solver_kwargs)
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
    if objective == "system":
        output += "print('Cost: {}'.format(results['cost']))\n"

    if not all(save is None for save in [save_orbs, save_ham, save_wfn]):
        output += "\n"
        output += "# Save results\n"
        output += "if results['success']:"
    if save_orbs is not None:
        output += "\n"
        output += "    unitary = unitary_matrix(ham.params)\n"
        output += "    np.save('{}', unitary)".format(save_orbs)
    if save_ham is not None:
        output += "\n"
        output += "    np.save('{}', ham.params)".format(save_ham)
    if save_wfn is not None:
        output += "\n"
        output += "    np.save('{}', wfn.params)".format(save_wfn)

    if filename is None:
        print(output)
    # NOTE: number was used instead of string (eg. 'return') to prevent problems arising from
    #       accidentally using the reserved string/keyword.
    elif filename == -1:
        return output
    else:
        with open(filename, "w") as f:
            f.write(output)


def main():
    """Run script for run_calc using arguments obtained via argparse."""
    parser.description = "Optimize a wavefunction and/or Hamiltonian."
    parser.add_argument("--nspin", type=int, required=True, help="Number of spin orbitals.")
    parser_add_arguments()
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
        args.nspin,
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
        load_wfn=args.load_wfn,
        load_chk=args.load_chk,
        save_orbs=args.save_orbs,
        save_ham=args.save_ham,
        save_wfn=args.save_wfn,
        save_chk=args.save_chk,
        filename=args.filename,
        memory=args.memory,
    )
