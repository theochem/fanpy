from __future__ import absolute_import, division, print_function

from horton import *


def hartreefock(fn=None, basis=None, nelec=None, solver=EDIIS2SCFSolver,
    nuc_nuc=True, tol=1.0e-12, horton_internal=False, **kwargs):
    # FIXME: doesn't support unrestricted
    # Initialize molecule and basis set from specified file
    if isinstance(fn, IOData):
        mol = fn
    else:
        try:
            mol = IOData.from_file(context.get_fn(fn))
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
    one_mo, two_mo =  one_mo[0]._array, two_mo[0]._array

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
            "orb": orb,
            "olp": olp,
        }

    return output

def ap1rog(fn=None, basis=None, nelec=None, solver=EDIIS2SCFSolver,
    nuc_nuc=True, tol=1.0e-12, opt=False, **kwargs):

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
        ap1rog_kwargs["orb"],
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
