"""Methods for calculating the energy from the wavefunction and hamiltonian.

Serve as the objective for single equation solvers.

Methods
-------
get_norm(wfn, pspace=None, return_grad=False)
    Return the norm of the wavefunction after projecting out some Slater determinants
get_energy_one_proj(wfn, ham, pspace_energy=None, pspace_norm=None, use_norm=True,
                    return_grad=False)
    Return the energy of the Schrodinger equation after projecting out one of the sides.
    Integrates each Slater determinant of the projection space with the Hamiltonian and the
    wavefunction.
get_energy_two_proj(wfn, ham, pspace_energy=None, pspace_norm=None, use_norm=True,
                    return_grad=False)
    Return the energy of the Schrodinger equation after projecting out both of the sides.
    Constructs the CI matrix and transforms it using the overlap between the wavefunction and the
    Slater determinants in the projection space.
"""
import numpy as np
from ..backend import sd_list, slater


def get_norm(wfn, pspace=None, return_grad=False):
    """Return the norm of the wavefunction.

    Parameters
    ----------
    wfn : BaseWavefunction
        Wavefunction that defines the state of the system (number of electrons and excited
        state).
    pspace : None, list/tuple of int
        Projection space used to truncate norm evaluation
        By default, all Slater determinants for a given wavefunction is used

    Returns
    -------
    norm : float
        Norm of the wavefunction
        If get_grad is False
    d_norm : np.ndarray
        Gradient of the norm of the wavefunction
        If get_grad is True
    """
    if not return_grad:
        return sum(wfn.get_overlap(sd)**2 for sd in pspace)
    else:
        return np.array([sum(2 * wfn.get_overlap(sd) * wfn.get_overlap(sd, deriv=j)
                             for sd in pspace)
                         for j in range(wfn.nparams)])


def get_energy_one_proj(wfn, ham, pspace_energy=None, pspace_norm=None, use_norm=True,
                        return_grad=False):
    """Return energy of the Schrodinger equation after projecting out one of the sides.

    ..math::
        E = \frac{\braket{\Psi | \hat{H} | \Psi}}{\braket{\Psi | \Psi}}

    Then, the numerator can be approximated by inserting a projection operator:
    ..math:
        \braket{\Psi | \hat{H} | \Psi} &\approx \bra{\Psi}
        \sum_{\mathbf{m} \in S} \ket{\mathbf{m}} \bra{\mathbf{m}}
        \hat{H} \ket{\Psi}\\
        &\approx \sum_{\mathbf{m} \in S} \braket{\Psi | \mathbf{m}}
        \braket{\mathbf{m} | \hat{H} | \Psi}\\

    Likewise, the denominator can be approximated by inserting a projection operator:
    ..math::
        \braket{\Psi | \Psi} &\approx \bra{\Psi}
        \sum_{\mathbf{m} \in S_{norm}} \ket{\mathbf{m}} \bra{\mathbf{m}}
        \ket{\Psi}\\
        &\approx \sum_{\mathbf{m} \in S_{norm}} \braket{\Psi | \mathbf{m}}^2


    Parameters
    ----------
    wfn : BaseWavefunction
        Wavefunction that defines the state of the system (number of electrons and excited
        state).
    ham : ChemicalHamiltonian
        Hamiltonian that defines the system under study.
    pspace_energy : None, list/tuple of int
        Projection space used to truncate the numerator of the energy evaluation.
        By default, all Slater determinants for a given wavefunction is used.
    pspace_norm : None, list/tuple of int
        Projection space used to truncate the denominator of the energy evaluation.
        By default, same space as pspace_energy is used.
    use_norm : bool
        Flag to use norm in the norm calculation.
        By default, norm is used.
    return_grad : bool
        Flag to return gradient of the energy.
        By default, the energy is returned.

    Returns
    -------
    energy : float
        Energy of the wavefunction with the given Hamiltonian
        If get_grad is False
    d_energy : np.ndarray
        Gradient of the energy of the wavefunction with the given Hamiltonian
        If get_grad is True
    """
    # get projection spaces
    if pspace_energy is None:
        pspace_energy = set(sd_list.sd_list(wfn.nelec, wfn.nspatial, spin=wfn.spin,
                                            seniority=wfn.seniority))
    else:
        pspace_energy = set(slater.internal_sd(sd) for sd in pspace_energy)
    if pspace_norm is None:
        pspace_norm = pspace_energy
    else:
        pspace_norm = set(slater.internal_sd(sd) for sd in pspace_norm)

    # energy
    energy = sum(wfn.get_overlap(sd) * sum(ham.integrate_wfn_sd(wfn, sd)) for sd in pspace_energy)
    d_energy = None
    if return_grad:
        # NOTE: hopefully, with caching, it's not too expensive to compute integrate_wfn_sd multiple
        #       times
        d_energy = np.array([sum(wfn.get_overlap(sd, deriv=j) * sum(ham.integrate_wfn_sd(wfn, sd)) +
                                 wfn.get_overlap(sd) * sum(ham.integrate_wfn_sd(wfn, sd, deriv=j))
                                 for sd in pspace_energy)
                             for j in range(wfn.nparams)])

    # norm
    if use_norm:
        norm = get_norm(wfn, pspace=pspace_norm, get_grad=False)
        if not return_grad:
            return energy / norm
        else:
            d_norm = get_norm(wfn, pspace=pspace_norm, get_grad=True)
            return (d_energy * norm - energy * d_norm) / norm**2
    else:
        if not return_grad:
            return energy
        else:
            return d_energy


def get_energy_two_proj(wfn, ham, l_pspace_energy=None, r_pspace_energy=None,
                        pspace_norm=None, use_norm=True, return_grad=False):
    """Energy of the Schrodinger equation after projecting out both of the sides.

    ..math::
        E = \frac{\braket{\Psi | \hat{H} | \Psi}}{\braket{\Psi | \Psi}}

    Then, the numerator can be approximated by inserting a projection operator:
    ..math:
        \braket{\Psi | \hat{H} | \Psi} &\approx \bra{\Psi}
        \sum_{\mathbf{m} \in S_l} \ket{\mathbf{m}} \bra{\mathbf{m}}
        \hat{H}
        \sum_{\mathbf{n} \in S_r} \ket{\mathbf{n}} \braket{\mathbf{n} | \Psi_\mathbf{n}}\\
        &\approx \sum_{\mathbf{m} \in S_l} \sum_{\mathbf{n} \in S_r} \braket{\Psi | \mathbf{m}}
        \braket{\mathbf{m} | \hat{H} | \mathbf{n}} \braket{\mathbf{n} | \Psi}\\

    Likewise, the denominator can be approximated by inserting a projection operator:
    ..math::
        \braket{\Psi | \Psi} &\approx \bra{\Psi}
        \sum_{\mathbf{m} \in S_{ref}} \ket{\mathbf{m}} \bra{\mathbf{m}}
        \ket{\Psi}\\
        &\approx \sum_{\mathbf{m} \in S} \braket{\Psi | \mathbf{m}}^2

    Parameters
    ----------
    wfn : BaseWavefunction
        Wavefunction that defines the state of the system (number of electrons and excited
        state)
    ham : ChemicalHamiltonian
        Hamiltonian that defines the system under study
    l_pspace_energy : None, list/tuple of int
        Projection space used to truncate the numerator of the energy evaluation from the left
    r_pspace_energy : None, list/tuple of int
        Projection space used to truncate the numerator of the energy evaluation from the right
    pspace_norm : None, list/tuple of int
        Projection space used to truncate the denominator of the energy evaluation
    use_norm : bool
        Flag to use norm in the norm calculation.
        By default, norm is used.
    return_grad : bool
        Flag to return gradient of the energy.
        By default, the energy is returned.

    Returns
    -------
    energy : float
        Energy of the wavefunction with the given Hamiltonian
        If get_grad is False
    d_energy : np.ndarray
        Gradient of the energy of the wavefunction with the given Hamiltonian
        If get_grad is True
    """
    # get projection spaces
    if l_pspace_energy is None:
        l_pspace_energy = set(sd_list.sd_list(wfn.nelec, wfn.nspatial, spin=wfn.spin,
                                              seniority=wfn.seniority))
    else:
        l_pspace_energy = set(slater.internal_sd(sd) for sd in l_pspace_energy)

    if r_pspace_energy is None:
        r_pspace_energy = set(sd_list.sd_list(wfn.nelec, wfn.nspatial, spin=wfn.spin,
                                              seniority=wfn.seniority))
    else:
        r_pspace_energy = set(slater.internal_sd(sd) for sd in r_pspace_energy)

    if pspace_norm is None:
        pspace_norm = l_pspace_energy | r_pspace_energy
    else:
        pspace_norm = set(slater.internal_sd(sd) for sd in pspace_norm)

    # energy
    energy = 0.0
    d_energy = np.zeros(wfn.nparams, dtype=wfn.dtype)
    for lsd in l_pspace_energy:
        for rsd in r_pspace_energy:
            ham_term = sum(ham.integrate_sd_sd(lsd, rsd))
            if ham_term == 0:
                continue
            energy += (wfn.get_overlap(lsd) * sum(ham.integrate_sd_sd(lsd, rsd))
                       * wfn.get_overlap(rsd))
            if return_grad:
                d_energy += np.array([wfn.get_overlap(lsd, deriv=j)*ham_term*wfn.get_overlap(rsd) +
                                      wfn.get_overlap(lsd)*ham_term*wfn.get_overlap(rsd, deriv=j)
                                      for j in range(d_energy.size)])

    # norm
    if use_norm:
        norm = get_norm(wfn, pspace=pspace_norm, get_grad=False)
        if not return_grad:
            return energy / norm
        else:
            d_norm = get_norm(wfn, pspace=pspace_norm, get_grad=True)
            return (d_energy * norm - energy * d_norm) / norm**2
    else:
        if not return_grad:
            return energy
        else:
            return d_energy
