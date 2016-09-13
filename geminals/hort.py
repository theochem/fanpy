from __future__ import absolute_import, division, print_function

import os
import numpy as np

from horton import *

def hartreefock(fn=None, basis=None, nelec=None,
                solver=EDIIS2SCFSolver, tol=1.0e-12,
                nuc_nuc=True, horton_internal=False,
                **kwargs):
    """ Runs a HF calculation using HORTON

    Parameters
    ----------
    fn :
    basis : str
        Basis set name
    nelec : int
        Number of electrons
    solver : PlainSCFSolver, ODASSCFSolver, CDIISSCFSolver, EDIISSCFSolver,  EDIIS2SCFSolver
        HORTON's SCF solver
    tol : float
        The convergence threshold for the wavefunction
    nuc_nuc : float
        Nuclear nuclear repulsion
    horton_internal : bool
        Flag for returning HORTON's internal data
    kwargs
        Other optional arguments for the SCF Solver

    Returns
    -------
    result : dict
        "energy", electronic energy
        "nuc_nuc", nuclear repulsion energy
        "H", tuple of the one-electron Hamiltonian;
        "G", tuple of the two-electron Hamiltonian;
        "horton_internal", dictionary that  contains horton's internal object
            "mol", horton.io.iodata.IOData object that contains fchk data
            "lf", horton.matrix.dense.LinalgFactory that creates horton matrices
            "occ_model", horton.meanfield.occ.OccModel that describe the occupation of
            the orbitals
            "one", horton.matrix.dense.DenseTwoIndex that contain the one electron
            integrals
            "two", horton.matrix.dense.DenseTwoIndex that contain the two electron
            integrals
            "nuc_nuc", nuclear repulsion energy
            "orb", horton.matrix.dense.DenseExpansion that contains the MO info
            "olp", horton.matrix.dense.DenseTwoIndex that contains the overlap matrix of
            the atomic orbitals
    """
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    file_path = os.path.join(data_dir, fn)
    # Initialize molecule and basis set from specified file
    if isinstance(fn, IOData):
        mol = fn
    else:
        try:
            mol = IOData.from_file(file_path)
        except IOError:
            mol = IOData.from_file(fn)
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
        raise_invalid_type("solver", solver, "callable")
    energy = ham.cache["energy"]

    # Transform one- and two- electron integrals into MO basis
    one = kin
    one.iadd(na)
    one_mo, two_mo = transform_integrals(one, two, "tensordot", orb)
    one_mo = tuple(i._array for i in one_mo)
    two_mo = tuple(i._array for i in two_mo)

    output = {
        # Collect HF energy and integrals
        "energy": energy if nuc_nuc else energy - external["nn"],
        "nuc_nuc": external["nn"] if nuc_nuc else 0.0,
        "H": one_mo,
        "G": two_mo,
    }

    if horton_internal:
        output["horton_internal"] = {
            # Collect input to HORTON's RAp1rog module
            "lf": lf,
            "occ_model": occ_model,
            "one": one,
            "two": two,
            "nuc_nuc": external["nn"] if nuc_nuc else 0.0,
            "orb": [orb,],
            "olp": olp,
        }

    return output


def ap1rog(fn=None,
           basis=None, nelec=None, nuc_nuc=True,
           solver=EDIIS2SCFSolver, tol=1.0e-12,
           opt=False,
           **kwargs):
    """ Runs an AP1roG calculation using HORTON

    Parameters
    ----------
    fn :
    basis : str
        Basis set name
    nelec : int
        Number of electrons
    nuc_nuc : float
        Nuclear nuclear repulsion
    solver : PlainSCFSolver, ODASSCFSolver, CDIISSCFSolver, EDIISSCFSolver,  EDIIS2SCFSolver
        HORTON's SCF solver
    tol : float
        The convergence threshold for the wavefunction
    opt : bool
    kwargs
        "hf_kwargs", dictionary of the keyword arguments for HF solver
        "ap1rog_kwargs", dictionary of the keyword arguments for AP1roG solver

    Returns
    -------
    result : dict
        "energy", electronic energy
        "H", one-electron Hamiltonian;
        "G", two-electron Hamiltonian;
        "x", AP1roG geminal coefficients;
        "mol", horton.io.iodata.IOData object that contains fchk data
        "basis", horton.gbasis.GOBasis that describes the atomic basis
        "orb", horton.matrix.dense.DenseExpansion that contains the MO info
    """
    #FIXME: add orbital rotation option

    hf_kwargs = {
        "fn": fn,
        "basis": basis,
        "nelec": nelec,
        "solver": solver,
        "nuc_nuc": nuc_nuc,
        "tol": tol,
        "horton_internal": True,
    }
    if "hf_kwargs" in kwargs:
        hf_kwargs.update(kwargs["hf_kwargs"])

    hf_result = hartreefock(**hf_kwargs)

    ap1rog_kwargs = hf_result["horton_internal"]
    ap1rog_kwargs["opt"] = opt
    if "ap1rog_kwargs" in kwargs:
        ap1rog_kwargs["ap1rog_kwargs"] = kwargs["ap1rog_kwargs"]
    else:
        ap1rog_kwargs["ap1rog_kwargs"] = {}

    geminal = RAp1rog(ap1rog_kwargs["lf"], ap1rog_kwargs["occ_model"])
    ap1rog_result = geminal(
        ap1rog_kwargs["one"],
        ap1rog_kwargs["two"],
        ap1rog_kwargs["nuc_nuc"],
        ap1rog_kwargs["orb"][0],
        ap1rog_kwargs["olp"],
        scf=ap1rog_kwargs["opt"],
        **ap1rog_kwargs["ap1rog_kwargs"])

    output = {
        "energy": ap1rog_result[0],
        "x": ap1rog_result[1]._array.ravel(),
        "C": ap1rog_result[1]._array,
    }
    if opt:
        output["lagrange"] = ap1rog_result[2]
    return output


def gaussian_fchk(fchk_file, horton_internal=False):
    """ Extracts the appropriate data from Gaussian fchk file (using HORTON)

    Parameters
    ----------
    fchk_file : str
        Formatted chk file
    horton_internal : bool


    Returns
    -------
    result : dict
        "energy", electronic energy including nuclear nuclear repulsion
        "nuc_nuc", zero
        "H", tuple of the one-electron Hamiltonian;
        "G", tuple of the two-electron Hamiltonian;
        "horton_internal", dictionary that  contains horton's internal object
            "mol", horton.io.iodata.IOData object that contains fchk data
            "lf", horton.matrix.dense.LinalgFactory that creates horton matrices
            "one", horton.matrix.dense.DenseTwoIndex that contain the one electron
            integrals
            "two", horton.matrix.dense.DenseTwoIndex that contain the two electron
            integrals
            "orb", horton.matrix.dense.DenseExpansion that contains the MO info
    """
    mol = IOData.from_file(fchk_file)
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

    # for spin orbitals
    one_mo = []
    two_mo = []
    for i, exp_i in enumerate(exps):
        for j, exp_j in enumerate(exps):
            if i <= j:
                temp = np.einsum('sd,pqrs->pqrd', exp_j.coeffs, two_ab, casting='no', order='C')
                temp = np.einsum('rc,pqrd->pqcd', exp_i.coeffs, temp, casting='no', order='C')
                temp = np.einsum('qb,pqcd->pbcd', exp_j.coeffs, temp, casting='no', order='C')
                temp = np.einsum('pa,pbcd->abcd', exp_i.coeffs, temp, casting='no', order='C')
                two_mo.append(temp)
        one_mo.append(exp_i.coeffs.T.dot(one_ab).dot(exp_i.coeffs))

    # FIXME: need to get the nuclear nuclear repulsion
    # energy includes the nuclear nuclear repulsion
    output = {
        "energy": mol.energy,
        "nuc_nuc": None,
        "H": tuple(one_mo),
        "G": tuple(two_mo),
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
