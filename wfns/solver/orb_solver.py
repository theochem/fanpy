"""Orbital optimization.

Orbital optimization essentially minimizes

..math::
    \braket{\Psi | \hat{U} \hat{H} \hat{U}^\dagger | \Psi}

where :math:`\hat{U}^\dagger` is orbital rotation operator (usually unitary).
This equation can be approached as finding the best wavefunction with rotated orbitals, i.e.
:math:`\hat{U}^\dagger \ket{\Psi}`, or as finding the best hamiltonian with rotated orbitals,
i.e. :math:`\hat{U} \hat{H} \hat{U}^\dagger` or some mix of the two.
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from ..wavefunction.base_wavefunction import BaseWavefunction
from ..hamiltonian.chemical_hamiltonian import ChemicalHamiltonian


def optimize_wfn_orbitals_jacobi(wfn, ham, wfn_solver=None):
    """Optimize orbitals of the given wavefunction to minimize energy using Jacobi rotations.

    The Jacobi rotated wavefunction, :math:`\hat{\mathbf{J}}^\dagger_{pq} \ket{\Psi}`, is
    iteratively optimized, where :math:`\hat{\mathbf{J}}^\dagger_{pq}` is the Jacobi orbital
    rotation operator.

    ..math::
        \hat{\mathbf{J}}_{pq} \hat{H} \hat{\mathbf{J}}_{pq}^\dagger
        = \sum_{ab}
        \left( \sum_{ij} (J_{pq})_{ai} h_{ij} (J_{pq})^\dagger_{jb} \right)
        a^\dagger_a a_b
        + \sum_{abcd}
        \left( \sum_{ijkl} (J_{pq})_{ai} (J_{pq})_{bj} g_{ijkl}
        (J_{pq})^\dagger_{kc} (J_{pq})^\dagger_{ld} \right)
        a^\dagger_a a^\dagger_b a_c a_d

    where :math:`J_{pq}` is the Jacobi rotation matrix.

    ..math::
        (J_{pq})_{ij}
        &= \delta_{ij} \mbox{if $(i,j) \not\in \{(p,p), (p,q), (q,p), (q,q)\}$}
        &= \cos \theta \mbox {if $(i,j) \in \{(p,p), (q,q)\}$}
        &= \sin \theta \mbox {if $(i,j) = (p,q)$}
        &= -\sin \theta \mbox {if $(i,j) = (q,p)$}

    When the operator is applied to the wavefunction,

    ..math::
        \hat{\mathbf{J}}_{pq}^\dagger \ket{\Psi}
        &= \sum_{\mathbf{n} \not\ni p,q} f(\mathbf{n}) \ket{\mathbf{n}}
        + \sum_{\mathbf{n} \ni p,q} f(\mathbf{n}) \ket{\mathbf{n}}\\
        &\hspace{2em}
        + \sum_{\mathbf{n} \ni p, \not\ni q}
        f(\mathbf{n}) (\cos\theta + \sin\theta) \ket{\mathbf{n}}
        + \sum_{\mathbf{n} \ni q, \not\ni p}
        f(\mathbf{n}) (\cos\theta - \sin\theta) \ket{\mathbf{n}}

    This wavefunction's parameter, :math:`\theta` , is optimized for the lowest energy in the
    given Hamiltonian.

    Parameters
    ----------
    wfn : BaseWavefunction
        Wavefunction that defines the state of the system (number of electrons and excited
        state)
    ham : ChemicalHamiltonian
        Hamiltonian that defines the system under study
    wfn_solver : function(wfn, ham)
        Function for solving the wavefunction in the given Hamiltonian
        This solver will only have access to its required arguments (`wfn` and `ham`) so all keyword
        arguments need to be added before passing the function. i.e. with
        `lambda wfn, ham: solver(wfn, ham, **kwargs)` where `kwargs` have been defined somewhere.
        Default raises NotImplementedError

    Returns
    -------
    Output of solver

    Raises
    ------
    TypeError
        If wavefunction is not an instance (or instance of a child) of BaseWavefunction
        If Hamiltonian is not an instance (or instance of a child) of ChemicalHamiltonian
    ValueError
        If wavefunction and Hamiltonian do not have the same data type
        If wavefunction and Hamiltonian do not have the same number of spin orbitals
    """
    # Preprocess variables
    if not isinstance(wfn, BaseWavefunction):
        raise TypeError('Given wavefunction is not an instance of BaseWavefunction (or its child).')
    if not isinstance(ham, ChemicalHamiltonian):
        raise TypeError('Given Hamiltonian is not an instance of BaseWavefunction (or its child).')
    if wfn.dtype != ham.dtype:
        raise ValueError('Wavefunction and Hamiltonian do not have the same data type.')
    if wfn.nspin != ham.nspin:
        raise ValueError('Wavefunction and Hamiltonian do not have the same number of '
                         'spin orbitals')
    pass
