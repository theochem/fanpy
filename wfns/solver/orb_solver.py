r"""Orbital optimization.

Orbital optimization essentially minimizes

..math::
    \braket{\Psi | \hat{U} \hat{H} \hat{U}^\dagger | \Psi}

where :math:`\hat{U}^\dagger` is orbital rotation operator (usually unitary).
This equation can be approached as finding the best wavefunction with rotated orbitals, i.e.
:math:`\hat{U}^\dagger \ket{\Psi}`, or as finding the best hamiltonian with rotated orbitals,
i.e. :math:`\hat{U} \hat{H} \hat{U}^\dagger` or some mix of the two.
"""
from __future__ import absolute_import, division, print_function
import itertools as it
import copy
import numpy as np
import scipy.optimize
from .equation_solver import optimize_wfn_variational
from ..backend import slater
from ..backend import sd_list
from ..wavefunction.base_wavefunction import BaseWavefunction
from ..wavefunction.nonorth.jacobi import JacobiWavefunction
from ..hamiltonian.chemical_hamiltonian import ChemicalHamiltonian


def optimize_wfn_orbitals_jacobi(wfn, ham, wfn_solver=None):
    r"""Optimize orbitals of the given wavefunction to minimize energy using Jacobi rotations.

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
        raise ValueError('Wavefunction and Hamiltonian do not have the same number of spin '
                         'orbitals.')

    if not isinstance(wfn, JacobiWavefunction):
        wfn = JacobiWavefunction(wfn.nelec, wfn.nspin, dtype=wfn.dtype, memory=wfn.memory, wfn=wfn,
                                 orbtype='restricted', jacobi_indices=(0, 1))

    if wfn.orbtype == 'restricted':
        pair_generator = it.combinations(range(wfn.nspatial), 2)
        transformation = (np.identity(wfn.nspatial), )
    elif wfn.orbtype == 'unrestricted':
        def pair_generator():
            yield from it.combinations(range(wfn.nspatial), 2)
            yield from it.combinations(range(wfn.nspatial, wfn.nspin), 2)
        transformation = (np.identity(wfn.nspatial), np.identity(wfn.nspatial))
    elif wfn.orbtype == 'generalized':
        pair_generator = it.combinations(range(wfn.nspin), 2)
        transformation = (np.identity(wfn.nspin), )

    num_iterations = 50
    test = np.identity(wfn.nspatial)
    for i in range(num_iterations):
        delta = 0.0
        for orbpair in pair_generator:
            wfn.assign_jacobi_indices(orbpair)
            wfn.clear_cache()
            wfn.assign_params(np.array(2 * np.pi * (np.random.random() - 0.5) / (i+1)))
            # FIXME: use gradient descent or stochastic gradient descent algorithm
            optimize_wfn_variational(wfn, ham, solver=scipy.optimize.brute,
                                     solver_kwargs={'Ns': 100 + int(10 * np.random.random()),
                                                    'ranges': ((-np.pi, np.pi), ),
                                                    'finish': None})
            # if results['success']:
            transformation = tuple(i.dot(j) for i, j in zip(transformation,
                                                            wfn.jacobi_rotation))
            delta += abs(wfn.params)
            ham.orb_rotate_jacobi(orbpair, wfn.params)
            test = test.dot(wfn.jacobi_rotation[0])
        # if last rotation does nothing (i.e. converged)
        if np.isclose(delta, 0.0):
            return test
            break
    # if end of loop is reached (i.e. did not converge)
    else:
        print('Orbital optimization did not converge after {0} iterations'.format(num_iterations))


def optimize_ham_orbitals_jacobi(wfn, ham, ref_sds=None, wfn_solver=None, wfn_solver_kwargs=None):
    r"""Optimize orbitals of the given hamiltonian to minimize energy using Jacobi rotations.

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
        raise ValueError('Wavefunction and Hamiltonian do not have the same number of spin '
                         'orbitals.')

    if ref_sds is None:
        ref_sds = sd_list.sd_list(wfn.nelec, wfn.nspatial, spin=wfn.spin, seniority=wfn.seniority)
    else:
        ref_sds = [slater.internal_sd(sd) for sd in ref_sds]

    # FIXME: not memory efficient
    def _objective(theta, p=0, q=1):
        ham.orb_rotate_jacobi((p, q), theta)
        norm = sum(wfn.get_overlap(sd)**2 for sd in ref_sds)
        energy = sum(wfn.get_overlap(sd) * sum(ham.integrate_wfn_sd(wfn, sd))
                     for sd in ref_sds)
        ham.orb_rotate_jacobi((p, q), -theta)
        return energy / norm

    num_iterations = 50
    orb_rot = np.identity(wfn.nspatial)
    thetas = {orbpair: 0 for orbpair in it.combinations(range(wfn.nspatial), 2)}
    for i in range(num_iterations):
        delta = 0.0

        if wfn_solver is not None:
            if wfn_solver_kwargs is None:
                wfn_solver_kwargs = {}
            wfn_solver(wfn, ham, **wfn_solver_kwargs)

        for p, q in it.combinations(range(wfn.nspatial), 2):
            res = scipy.optimize.minimize_scalar(_objective, args=(p, q), method='brent')
            if res.success:
                delta += res.x
                ham.orb_rotate_jacobi((p, q), res.x)
                p_col = orb_rot[:, p]
                q_col = orb_rot[:, q]
                (orb_rot[:, p], orb_rot[:, q]) = (np.cos(res.x)*p_col - np.sin(res.x)*q_col,
                                                  np.sin(res.x)*p_col + np.cos(res.x)*q_col)
                thetas[(p, q)] += res.x

        # if last rotation does nothing (i.e. converged)
        if np.isclose(delta, 0.0):
            return orb_rot, thetas, _objective(0.0)
    # if end of loop is reached (i.e. did not converge)
    else:
        print('Orbital optimization did not converge after {0} iterations'.format(num_iterations))
