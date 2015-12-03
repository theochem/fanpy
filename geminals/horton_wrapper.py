#!/usr/bin/env python2

from __future__ import absolute_import, division, print_function

import numpy as np
from horton import context, IOData
from horton.cext import compute_nucnuc
from horton.correlatedwfn import RAp1rog
from horton.gbasis import get_gobasis
from horton.matrix import DenseLinalgFactory
from horton.meanfield import AufbauOccModel
from horton.meanfield import guess_core_hamiltonian, PlainSCFSolver, REffHam
from horton.meanfield.observable import RDirectTerm, RExchangeTerm, RTwoIndexTerm
from horton.orbital_utils import transform_integrals

def from_horton(fn=None, basis=None, nocc=None, guess='apig'):
    """Computes Geminal-class-compatible information about a molecule's wavefunction from
    HORTON.

    Parameters
    ----------
    fn : str
        The file containing the molecule's information.
    basis: str
        The basis set to use for the orbitals.
    nocc :
        The number of occupied orbitals.
    guess : str, optional
        The type of guess for the coefficients to make.  One of `apig` or 'ap1rog'.

    Returns
    -------
    result : dict
        Contains `mol`, an IOData instance; `basis`, a GOBasis instance; and `ham`, a
        tuple containing the terms of the Hamiltonian matrix.

    """

    # Load the molecule and basis set from file
    mol = IOData.from_file(context.get_fn(fn))
    obasis = get_gobasis(mol.coordinates, mol.numbers, basis)

    # Fill in the orbital expansion and overlap
    occ_model = AufbauOccModel(nocc)
    lf = DenseLinalgFactory(obasis.nbasis)
    orb = lf.create_expansion(obasis.nbasis)
    olp = obasis.compute_overlap(lf)

    # Construct Hamiltonian
    kin = obasis.compute_kinetic(lf)
    na = obasis.compute_nuclear_attraction(mol.coordinates, mol.pseudo_numbers, lf)
    two = obasis.compute_electron_repulsion(lf)
    external = {'nn': compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}
    terms = [ RTwoIndexTerm(kin, 'kin'),
              RDirectTerm(two, 'hartree'),
              RExchangeTerm(two, 'x_hf'),
              RTwoIndexTerm(na, 'ne'),
            ]
    ham = REffHam(terms, external)
    guess_core_hamiltonian(olp, kin, na, orb)

    # Do Hartree-Fock SCF
    PlainSCFSolver(1.0e-6)(ham, lf, olp, occ_model, orb)

    # Get initial guess at energy, coefficients from AP1roG
    one = kin
    one.iadd(na)
    ap1rog = RAp1rog(lf, occ_model)
    energy, cblock = ap1rog(one, two, external['nn'], orb, olp, False)

    # Transform the one- and two- index integrals into the MO basis
    one_mo, two_mo = transform_integrals(one, two, 'tensordot', orb)

    #RAp1rog only returns the 'A' block from the [I|A]-shaped coefficient matrix
    if guess is 'apig':
        coeffs = np.eye(nocc, obasis.nbasis)
        coeffs[:,nocc:] += cblock._array
    elif guess is 'ap1rog':
        coeffs = cblock._array
    else:
        raise NotImplementedError

    # Return 
    return { 'mol': mol,
             'basis': obasis,
             'ham': (one_mo[0]._array, two_mo[0]._array, external['nn']),
             'energy': energy,
             'coeffs': coeffs,
           }

# vim: set textwidth=90 :
