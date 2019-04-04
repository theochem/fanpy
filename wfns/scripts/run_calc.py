"""Script for running calculations."""
import numpy as np
from wfns.scripts.make_script import make_script
from wfns.scripts.utils import check_inputs, parser, parser_add_arguments


# FIXME: not tested
def run_calc(
    nelec,
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
    load_orbs=None,
    load_ham=None,
    load_wfn=None,
    load_chk=None,
    save_orbs=None,
    save_ham=None,
    save_wfn=None,
    save_chk="",
    filename=None,
    memory=None,
):
    """Script for running some basic calculations.

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
    memory : None
        Memory available for the calculation.

    """
    # check inputs
    check_inputs(
        nelec, 0, one_int_file, two_int_file, wfn_type, pspace_exc, objective, solver, nuc_nuc
    )
    nspin = np.load(one_int_file)[0].shape[0] * 2

    # make script
    script = make_script(
        nelec,
        nspin,
        one_int_file,
        two_int_file,
        wfn_type,
        nuc_nuc=nuc_nuc,
        wfn_kwargs=wfn_kwargs,
        optimize_orbs=optimize_orbs,
        pspace_exc=pspace_exc,
        objective=objective,
        solver=solver,
        load_orbs=load_orbs,
        load_ham=load_ham,
        load_wfn=load_wfn,
        load_chk=load_chk,
        save_orbs=save_orbs,
        save_ham=save_ham,
        save_wfn=save_wfn,
        save_chk=save_chk,
        filename=-1,
        memory=memory,
    )
    # run script
    exec(script)


def main():
    """Run script for run_calc using arguments obtained via argparse."""
    parser.description = "Optimize a wavefunction and/or Hamiltonian."
    parser_add_arguments()
    args = parser.parse_args()
    run_calc(
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
        load_orbs=args.load_orbs,
        load_ham=args.load_ham,
        load_wfn=args.load_wfn,
        load_chk=args.load_chk,
        save_orbs=args.save_orbs,
        save_ham=args.save_ham,
        save_wfn=args.save_wfn,
        save_chk=args.save_chk,
        memory=args.memory,
    )
