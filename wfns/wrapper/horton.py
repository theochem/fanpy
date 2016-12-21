""" Wrapper of HORTON

At this moment, we mainly need the integrals (in MO basis) and the energies (electron, nuclear
nuclear repulsion) to construct a wavefunction. Here, we use HORTON to get these values.

Functions
---------
hartreefock(fn=None, basis=None, nelec=None, solver=EDIIS2SCFSolver, tol=1.0e-12,
            horton_internal=False, **kwargs)
    Runs a HF in HORTON
gaussian_fchk(fchk_file, horton_internal=False, compute_nuc=True)
    Extracts appropriate information from a Gaussian FCHK file
"""
from __future__ import absolute_import, division, print_function
import os
import numpy as np
from horton import (IOData, get_gobasis,
                    PlainSCFSolver, EDIIS2SCFSolver,
                    DenseLinalgFactory,
                    compute_nucnuc, guess_core_hamiltonian, transform_integrals,
                    AufbauOccModel, RTwoIndexTerm, RDirectTerm, RExchangeTerm, REffHam)
from wfns import __file__ as package_path


def hartreefock(fn=None, basis=None, nelec=None, solver=EDIIS2SCFSolver, tol=1.0e-12,
                horton_internal=False, **kwargs):
    """ Runs a HF calculation using HORTON

    Parameters
    ----------
    fn : str
        File name
        Supports XYZ file (in Angstrom)
    basis : str
        Basis set name
    nelec : int
        Number of electrons
    solver : PlainSCFSolver, ODASSCFSolver, CDIISSCFSolver, EDIISSCFSolver,  EDIIS2SCFSolver
        HORTON's SCF solver
    tol : float
        The convergence threshold for the wavefunction
    horton_internal : bool
        Flag for returning HORTON's internal data
    kwargs
        Other optional arguments for the SCF Solver

    Returns
    -------
    result : dict
        "el_energy"
            Electronic energy
        "nuc_nuc_energy"
            Nuclear repulsion energy
        "one_int"
            Tuple of the one-electron Hamiltonian;
        "two_int"
            Tuple of the two-electron Hamiltonian;
        "horton_internal"
            Dictionary that  contains horton's internal object
            "mol"
                horton.io.iodata.IOData object that contains fchk data
            "lf"
                horton.matrix.dense.LinalgFactory that creates horton matrices
            "occ_model"
                horton.meanfield.occ.OccModel that describe the occupation of the orbitals
            "one"
                horton.matrix.dense.DenseTwoIndex that contain the one electron integrals
            "two"
                horton.matrix.dense.DenseTwoIndex that contain the two electron integrals
            "orb"
                horton.matrix.dense.DenseExpansion that contains the MO info
            "olp"
                horton.matrix.dense.DenseTwoIndex that contains the overlap matrix of the atomic
                orbitals

    Raises
    ------
    ValueError
        If solver is not callable

    Note
    ----
    While HORTON does support unrestricted calculations, the wrapper does not support it.
    """
    # Initialize molecule and basis set from specified file
    if isinstance(fn, IOData):
        mol = fn
    else:
        try:
            mol = IOData.from_file(fn)
        except IOError:
            data_dir = os.path.join(os.path.dirname(package_path), '../data')
            file_path = os.path.join(data_dir, fn)
            mol = IOData.from_file(file_path)
    obasis = get_gobasis(mol.coordinates, mol.numbers, basis)
    npair = nelec // 2

    # Fill in orbital expansion and overlap
    occ_model = AufbauOccModel(npair)
    lf = DenseLinalgFactory(obasis.nbasis)
    orb = lf.create_expansion(obasis.nbasis)
    olp = obasis.compute_overlap(lf)

    # Construct Hamiltonian and density matrix
    kin = obasis.compute_kinetic(lf)
    na = obasis.compute_nuclear_attraction(mol.coordinates, mol.pseudo_numbers, lf)
    two = obasis.compute_electron_repulsion(lf)
    external = {"nn": compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}
    terms = [
        RTwoIndexTerm(kin, "kin"),
        RDirectTerm(two, "hartree"),
        RExchangeTerm(two, "x_hf"),
        RTwoIndexTerm(na, "ne"),
    ]
    ham = REffHam(terms, external)
    guess_core_hamiltonian(olp, kin, na, orb)

    # Do Hartree-Fock SCF
    if solver is PlainSCFSolver:
        PlainSCFSolver(tol)(ham, lf, olp, occ_model, orb, **kwargs)
    elif hasattr(solver, "__call__"):
        occ_model.assign(orb)
        dm = orb.to_dm()
        solver(tol)(ham, lf, olp, occ_model, dm, **kwargs)
        # transform back to orbitals
        fock_alpha = lf.create_two_index()
        ham.compute_fock(fock_alpha)
        orb.from_fock_and_dm(fock_alpha, dm, olp)
    else:
        raise ValueError('Given solver, {0}, is not callable'.format(solver))
    energy = ham.cache["energy"]

    # Transform one- and two- electron integrals into MO basis
    one = kin
    one.iadd(na)
    one_mo, two_mo = transform_integrals(one, two, "tensordot", orb)
    one_mo = tuple(i._array for i in one_mo)
    two_mo = tuple(i._array for i in two_mo)

    output = {
        # Collect HF energy and integrals
        "el_energy": energy - external["nn"],
        "nuc_nuc_energy": external["nn"],
        "one_int": one_mo,
        "two_int": two_mo,
    }

    if horton_internal:
        output["horton_internal"] = {
            "lf": lf,
            "occ_model": occ_model,
            "one": one,
            "two": two,
            "orb": [orb,],
            "olp": olp,
        }

    return output


def gaussian_fchk(fchk_file, horton_internal=False):
    """ Extracts the appropriate data from Gaussian fchk file (using HORTON)

    Parameters
    ----------
    fchk_file : str
        Formatted chk file
    horton_internal : bool
        Flag to return horton_internal variables

    Returns
    -------
    result : dict
        "el_energy"
            Electronic energy
        "nuc_nuc_energy"
            Nuclear nuclear repulsion energy
        "one_int"
            Tuple of the one-electron Hamiltonian;
        "two_int"
            Tuple of the two-electron Hamiltonian;
        "horton_internal"
            Dictionary that  contains horton's internal object
            "mol"
                horton.io.iodata.IOData object that contains fchk data
            "lf"
                horton.matrix.dense.LinalgFactory that creates horton matrices
            "one"
                horton.matrix.dense.DenseTwoIndex that contain the one electron integrals
            "two"
                horton.matrix.dense.DenseTwoIndex that contain the two electron integrals
            "orb"
                horton.matrix.dense.DenseExpansion that contains the MO info
    """
    try:
        mol = IOData.from_file(fchk_file)
    except IOError:
        data_dir = os.path.join(os.path.dirname(package_path), '../data')
        file_path = os.path.join(data_dir, fchk_file)
        mol = IOData.from_file(file_path)

    # for spin orbitals
    exps = [mol.exp_alpha]
    if hasattr(mol, 'exp_beta'):
        exps.append(mol.exp_beta)

    obasis = mol.obasis
    kin = obasis.compute_kinetic(mol.lf)._array
    na = obasis.compute_nuclear_attraction(mol.coordinates, mol.pseudo_numbers, mol.lf)._array

    one_ab = kin
    one_ab += na
    two_ab = obasis.compute_electron_repulsion(mol.lf)._array

    # compute nuclear nuclear repulsion
    nuc_nuc = compute_nucnuc(mol.coordinates, mol.pseudo_numbers)

    # for spin orbitals
    one_mo = []
    two_mo = []
    for i, exp_i in enumerate(exps):
        for j, exp_j in enumerate(exps[i:]):
            j += i
            temp = np.einsum('sd,pqrs->pqrd', exp_j.coeffs, two_ab, casting='no', order='C')
            temp = np.einsum('rc,pqrd->pqcd', exp_i.coeffs, temp, casting='no', order='C')
            temp = np.einsum('qb,pqcd->pbcd', exp_j.coeffs, temp, casting='no', order='C')
            temp = np.einsum('pa,pbcd->abcd', exp_i.coeffs, temp, casting='no', order='C')
            two_mo.append(temp)
        one_mo.append(exp_i.coeffs.T.dot(one_ab).dot(exp_i.coeffs))

    output = {
        "el_energy": mol.energy - nuc_nuc,
        "nuc_nuc_energy": nuc_nuc,
        "one_int": tuple(one_mo),
        "two_int": tuple(two_mo),
    }
    if horton_internal:
        output["horton_internal"] = {
            "mol": mol,
            "lf": mol.lf,
            "one": one_mo,
            "two": two_mo,
            "orb": exps,
        }
    return output
