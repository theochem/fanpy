from itertools import combinations
import numpy as np

from . import slater
from .ci_matrix import spatial_index, get_H_value, get_G_value

def hamiltonian(self, sd, orb_type, deriv=None):
    """ Compute the Hamiltonian of the wavefunction projected against `sd`.

    ..math::
        \big< \Phi_i \big| H \big| \Psi \big>

    Parameters
    ----------
    self : Wavefunction Instance
        Instance of Wavefunction class
        Needs to have the following in __dict__:
            nspin, H, G, overlap
    sd : int
        The Slater Determinant against which to project.
    orb_type : {'restricted', 'unrestricted', 'generalized'}
        Flag that indicates the type of the orbital

    Returns
    -------
    energy : tuple of floats
        Tuple of floats corresponding to the one electron, coulomb, and exchange
        energies.

    """
    H = self.H
    G = self.G

    occ_indices = slater.occ_indices(sd)
    vir_indices = slater.vir_indices(sd, self.nspin)

    one_electron = 0.0
    coulomb = 0.0
    exchange = 0.0
    # sum over zeroth order excitation
    coeff = self.overlap(sd, deriv=deriv)
    for ic, i in enumerate(occ_indices):
        one_electron += coeff*get_H_value(H, i, i, orb_type=orb_type)
        for j in occ_indices[ic+1:]:
            coulomb += coeff*get_G_value(G, i, j, i, j, orb_type=orb_type)
            exchange -= coeff*get_G_value(G, i, j, j, i, orb_type=orb_type)
    # sum over one electron excitation
    for i in occ_indices:
        for a in vir_indices:
            ion_sd = slater.annihilate(sd, i)
            exc_sd = slater.create(sd, a)
            coeff = self.overlap(exc_sd, deriv=deriv)
            one_electron += coeff*get_H_value(H, i, a, orb_type=orb_type)
            for j in slater.occ_indices(ion_sd):
                if j != i:
                    coulomb += coeff*get_G_value(G, i, j, a, j, orb_type=orb_type)
                    exchange -= coeff*get_G_value(G, i, j, j, a, orb_type=orb_type)
    # sum over two electron excitation
    for i,j in combinations(occ_indices, 2):
        for a, b in combinations(vir_indices, 2):
            exc_sd = slater.excite(sd, i, j, a, b)
            coeff = self.overlap(exc_sd, deriv=deriv)
            coulomb += coeff*get_G_value(G, i, j, k, l, orb_type=orb_type)
            exchange -= coeff*get_G_value(G, i, j, l, k, orb_type=orb_type)
    return one_electron, coulomb, exchange

def doci_hamiltonian(self, sd, orb_type, deriv=None):
    """ Compute the DOCI Hamiltonian of the wavefunction projected against `sd`.

    ..math::
        \big< \Phi_i \big| H \big| \Psi \big>

    Parameters
    ----------
    self : Wavefunction Instance
        Instance of Wavefunction class
        Needs to have the following in __dict__:
            nspin, nspatial, H, G, overlap
    sd : int
        The Slater Determinant against which to project.
    orb_type : {'restricted', 'unrestricted', 'generalized'}
        Flag that indicates the type of the orbital

    Returns
    -------
    energy : tuple
        Tuple of floats corresponding to the one electron, coulomb, and exchange
        energies.

    """
    H = self.H
    G = self.G
    ns = self.nspatial

    occ_indices = slater.occ_indices(sd)
    alpha_sd, beta_sd = slater.split_spin(sd, ns)
    occ_alpha_indices = slater.occ_indices(alpha_sd)
    occ_beta_indices = slater.occ_indices(beta_sd)
    if occ_alpha_indices != occ_beta_indices:
        raise ValueError('Given Slater determinant, {0}, does not belong'
                         ' to the DOCI Slater determinants'.format(bin(sd)))
    vir_alpha_indices = slater.vir_indices(alpha_sd, ns)

    one_electron = 0.0
    coulomb = 0.0
    exchange = 0.0
    # FIXME: restricted orbitals hard coded in
    # sum over zeroth order excitation
    coeff = self.overlap(sd, deriv=deriv)
    for ic, i in enumerate(occ_alpha_indices):
        one_electron += 2*coeff*get_H_value(H, i, i, orb_type=orb_type)
        coulomb += coeff*get_G_value(G, i, i, i, i, orb_type=orb_type)
        for j in occ_alpha_indices[ic+1:]:
            coulomb += 4*coeff*get_G_value(G, i, j, i, j, orb_type=orb_type)
            exchange -= 2*coeff*get_G_value(G, i, j, j, i, orb_type=orb_type)
    # sum over two electron excitation
    for i in occ_alpha_indices:
        for a in vir_alpha_indices:
            j = i + ns
            b = a + ns
            exc_sd = slater.excite(sd, i, j, a, b)
            # This shouldn't happen
            if exc_sd is None:
                raise ValueError('Given Slater determinant, {0}, does not belong'
                                 ' to the DOCI Slater determinants'.format(bin(sd)))
            coeff = self.overlap(exc_sd, deriv=deriv)
            coulomb += coeff*get_G_value(G, i, j, a, b, orb_type=orb_type)
            exchange -= coeff*get_G_value(G, i, j, b, a, orb_type=orb_type)
    return one_electron, coulomb, exchange
