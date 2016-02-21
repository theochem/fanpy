from __future__ import absolute_import, division, print_function

import numpy as np
from horton import AufbauOccModel, DenseLinalgFactory, IOData, PlainSCFSolver
from horton import REffHam, RDirectTerm, RExchangeTerm, RTwoIndexTerm, RAp1rog
from horton import context, compute_nucnuc, get_gobasis, guess_core_hamiltonian, transform_integrals


def ap1rog(fn=None, basis=None, n=None, scf=True, opt=False, x=None):
    """
    Compute information about a molecule's AP1roG wavefunction with HORTON.

    Parameters
    ----------
    fn : str
        The file containing the molecular geometry.
    basis: str
        The basis set to use for the orbitals.
    n : int
        The number of electrons.
    scf : bool
        Whether to do a Hartree-Fock SCF.
    opt : bool
        Whether to optimize the orbitals using the vOO-AP1roG method.
    x : str, optional
        The type of coefficient vector to return.  One of "apig", "ap1rog".

    Returns
    -------
    result : dict
        Contains "mol", a `horton.IOData` instance;
        "basis", a `horton.GOBasis` instance;
        "orb", the orbital coefficients;
        "H", the one-electron Hamiltonian;
        "G", the two-electron Hamiltonian;
        "x", the coefficient vector.
        "energy", the energy of the system.

    """

    # Initialize the molecule and basis set from the specified file
    try:
        mol = IOData.from_file(fn)
    except IOError:
        mol = IOData.from_file(context.get_fn(fn))
    obasis = get_gobasis(mol.coordinates, mol.numbers, basis)
    p = n // 2

    # Fill in the orbital expansion and overlap
    occ_model = AufbauOccModel(p)
    lf = DenseLinalgFactory(obasis.nbasis)
    orb = lf.create_expansion(obasis.nbasis)
    olp = obasis.compute_overlap(lf)

    # Construct Hamiltonian
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
    if scf:
        PlainSCFSolver(1.0e-6)(ham, lf, olp, occ_model, orb)

    # Get initial guess at energy, coefficients from RAp1rog
    one = kin
    one.iadd(na)
    if x:
        ap1rog = RAp1rog(lf, occ_model)
        ap1rog_result = ap1rog(one, two, external["nn"], orb, olp, opt)
        energy, c = ap1rog_result[:2]
    else:
        energy = None
    energy -= external["nn"]

    # Transform the one- and two- electron integrals into the MO basis
    one_mo, two_mo = transform_integrals(one, two, "tensordot", orb)

    # Return the results
    if x is "apig":
        x = np.zeros((p, 2 * obasis.nbasis))
        x[:, 0::2] = np.eye(p, obasis.nbasis)
        x[:, (2 * p)::2] = c._array
        x = x.ravel()
    elif x is "ap1rog":
        x = c._array
        x = x.ravel()
    else:
        x = None

    return {
        "mol": mol,
        "basis": obasis,
        "orb": orb.coeffs,
        "H": one_mo[0]._array,
        "G": two_mo[0]._array,
        "x": x,
        "energy": energy,
    }

def gaussian_fchk(fchk_file):
    """ Extracts the hamiltonian from gaussian fchk file

    Parameters
    ----------
    fchk_file : str
        Formatted chk file

    Returns
    -------
    result : dict
        Contains "mol", a `horton.IOData` instance;
        "basis", a `horton.GOBasis` instance;
        "orb", the spatial orbital coefficients;
        "H", the one-electron Hamiltonian;
        "G", the two-electron Hamiltonian;

    Raises
    ------
    Assertion Error
        If beta orbitals exist (orbitals are not spatial)
    """
    mol = IOData.from_file(fchk_file)
    # for spin orbitals
    exps = [mol.exp_alpha]
    if hasattr(mol, 'exp_beta'):
        exps.append(mol.exp_beta)
    #assert not hasattr(mol, 'exp_beta')

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

    return {
        "mol": mol,
        "basis": obasis,
        "orb": [exp.coeffs for exp in exps],
        "H": one_mo[0],
        "G": two_mo[0]
    }
