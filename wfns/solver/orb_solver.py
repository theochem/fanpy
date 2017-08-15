r"""Orbital optimization.

Orbital optimization essentially minimizes

.. math::

    \braket{\Psi | \hat{U} \hat{H} \hat{U}^\dagger | \Psi}

where :math:`\hat{U}^\dagger` is orbital rotation operator (usually unitary).

This equation can be approached as finding the best wavefunction with rotated orbitals, i.e.
:math:`\hat{U}^\dagger \ket{\Psi}`, or as finding the best hamiltonian with rotated orbitals, i.e.
:math:`\hat{U} \hat{H} \hat{U}^\dagger` or some mix of the two.

"""
from __future__ import absolute_import, division, print_function
import itertools as it
import numpy as np
import scipy.optimize
import wfns.backend.slater as slater
import wfns.backend.sd_list as sd_list
from wfns.solver.equation_solver import optimize_wfn_variational
from wfns.solver import energies
from wfns.wavefunction.base_wavefunction import BaseWavefunction
from wfns.wavefunction.composite.jacobi import JacobiWavefunction
from wfns.hamiltonian.chemical_hamiltonian import ChemicalHamiltonian
from wfns.wrapper.docstring import docstring


@docstring(indent_level=1)
def optimize_wfn_orbitals_jacobi(wfn, ham, wfn_solver=None, save_file=''):
    r"""Optimize orbitals of the given wavefunction to minimize energy using Jacobi rotations.

    The Jacobi rotated wavefunction, :math:`\hat{\mathbf{J}}^\dagger_{pq} \ket{\Psi}`, is
    iteratively optimized, where :math:`\hat{\mathbf{J}}^\dagger_{pq}` is the Jacobi orbital
    rotation operator.

    .. math::

        \hat{\mathbf{J}}_{pq} \hat{H} \hat{\mathbf{J}}_{pq}^\dagger
        = \sum_{ab}
        \left( \sum_{ij} (J_{pq})_{ai} h_{ij} (J_{pq})^\dagger_{jb} \right)
        a^\dagger_a a_b
        + \sum_{abcd}
        \left( \sum_{ijkl} (J_{pq})_{ai} (J_{pq})_{bj} g_{ijkl}
        (J_{pq})^\dagger_{kc} (J_{pq})^\dagger_{ld} \right)
        a^\dagger_a a^\dagger_b a_c a_d

    where :math:`J_{pq}` is the Jacobi rotation matrix.

    .. math::

        (J_{pq})_{ij}
        &= \delta_{ij} \mbox{if $(i,j) \not\in \{(p,p), (p,q), (q,p), (q,q)\}$}\\
        &= \cos \theta \mbox {if $(i,j) \in \{(p,p), (q,q)\}$}\\
        &= \sin \theta \mbox {if $(i,j) = (p,q)$}\\
        &= -\sin \theta \mbox {if $(i,j) = (q,p)$}\\

    When the operator is applied to the wavefunction,

    .. math::

        \hat{\mathbf{J}}_{pq}^\dagger \ket{\Psi}
        &= \sum_{\mathbf{n} \not\ni p,q} f(\mathbf{n}) \ket{\mathbf{n}}
        + \sum_{\mathbf{n} \ni p,q} f(\mathbf{n}) \ket{\mathbf{n}}\\
        &\hspace{2em}
        + \sum_{\mathbf{n} \ni p, \not\ni q}
        f(\mathbf{n}) (\cos\theta + \sin\theta) \ket{\mathbf{n}}
        + \sum_{\mathbf{n} \ni q, \not\ni p}
        f(\mathbf{n}) (\cos\theta - \sin\theta) \ket{\mathbf{n}}

    This wavefunction's parameter, :math:`\theta` , is optimized for the lowest energy in the given
    Hamiltonian.

    Parameters
    ----------
    wfn : BaseWavefunction
        Wavefunction that defines the state of the system (number of electrons and excited state).
    ham : ChemicalHamiltonian
        Hamiltonian that defines the system under study.
    wfn_solver : function(wfn, ham)
        Function for solving the wavefunction in the given Hamiltonian.
        This solver will only have access to its required arguments (`wfn` and `ham`) so all keyword
        arguments need to be added before passing the function. i.e. with `lambda wfn, ham:
        solver(wfn, ham, **kwargs)` where `kwargs` have been defined somewhere.
        Default raises NotImplementedError.

    Returns
    -------
    output : dict
        Output of solver.

    Raises
    ------
    TypeError
        If wavefunction is not an instance (or instance of a child) of BaseWavefunction.
        If Hamiltonian is not an instance (or instance of a child) of ChemicalHamiltonian.
    ValueError
        If wavefunction and Hamiltonian do not have the same data type.
        If wavefunction and Hamiltonian do not have the same number of spin orbitals.

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
        if save_file != '':
            np.save(save_file, transformation)
        # if last rotation does nothing (i.e. converged)
        if np.isclose(delta, 0.0):
            return test
            break
    # if end of loop is reached (i.e. did not converge)
    else:
        print('Orbital optimization did not converge after {0} iterations'.format(num_iterations))


def optimize_ham_orbitals_jacobi(wfn, ham, pspace_norm=None,
                                 l_pspace_energy=None, r_pspace_energy=None, use_norm=True,
                                 wfn_solver=None, wfn_solver_kwargs=None, save_file=''):
    r"""Optimize orbitals of the given hamiltonian to minimize energy using Jacobi rotations.

    The Jacobi rotated wavefunction, :math:`\hat{\mathbf{J}}^\dagger_{pq} \ket{\Psi}`, is
    iteratively optimized, where :math:`\hat{\mathbf{J}}^\dagger_{pq}` is the Jacobi orbital
    rotation operator.

    .. math::

        \hat{\mathbf{J}}_{pq} \hat{H} \hat{\mathbf{J}}_{pq}^\dagger
        = \sum_{ab}
        \left( \sum_{ij} (J_{pq})_{ai} h_{ij} (J_{pq})^\dagger_{jb} \right)
        a^\dagger_a a_b
        + \sum_{abcd}
        \left( \sum_{ijkl} (J_{pq})_{ai} (J_{pq})_{bj} g_{ijkl}
        (J_{pq})^\dagger_{kc} (J_{pq})^\dagger_{ld} \right)
        a^\dagger_a a^\dagger_b a_c a_d

    where :math:`J_{pq}` is the Jacobi rotation matrix.

    .. math::
        (J_{pq})_{ij}
        &= \delta_{ij} \mbox{if $(i,j) \not\in \{(p,p), (p,q), (q,p), (q,q)\}$}\\
        &= \cos \theta \mbox {if $(i,j) \in \{(p,p), (q,q)\}$}\\
        &= \sin \theta \mbox {if $(i,j) = (p,q)$}\\
        &= -\sin \theta \mbox {if $(i,j) = (q,p)$}

    Parameters
    ----------
    wfn : BaseWavefunction
        Wavefunction that defines the state of the system (number of electrons and excited state).
    ham : ChemicalHamiltonian
        Hamiltonian that defines the system under study.
    pspace_norm : list/tuple of int
        Projection space for calculating the norm.
        Default is the all of the Slater determinants of the wavefunction.
    l_pspace_energy : list/tuple of int
        Projection space for the left side of the energy.
        Default is the all of the Slater determinants of the wavefunction.
    r_pspace_energy : list/tuple of int
        Projection space for the right side of the energy.
        Default is the all of the Slater determinants of the wavefunction.
    use_norm : bool
        Flag for using norm to calculate the energy.
        Default is True.
    wfn_solver : function(wfn, ham)
        Function for solving the wavefunction in the given Hamiltonian.
        This solver will only have access to its required arguments (`wfn` and `ham`) so all keyword
        arguments need to be added before passing the function. i.e. with `lambda wfn, ham:
        solver(wfn, ham, **kwargs)` where `kwargs` have been defined somewhere.
        Default raises NotImplementedError.
    wfn_solver_kwargs : dict
        Optional keyword arguments for the wfn_solver.
    save_file : str
        Name of the file to save the orbitals.
        Default is no save.

    Returns
    -------
    output : dict
        Output of solver.

    Raises
    ------
    TypeError
        If wavefunction is not an instance (or instance of a child) of BaseWavefunction.
        If Hamiltonian is not an instance (or instance of a child) of ChemicalHamiltonian.
    ValueError
        If wavefunction and Hamiltonian do not have the same data type.
        If wavefunction and Hamiltonian do not have the same number of spin orbitals.

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

    # FIXME: not memory efficient
    def _objective(theta, p, q, use_norm=True):
        # rotate hamiltonian
        ham.orb_rotate_jacobi((p, q), theta)
        # calculate energy
        if l_pspace_energy is not None and r_pspace_energy is not None:
            energy = energies.get_energy_two_proj(wfn, ham,
                                                  l_pspace_energy=l_pspace_energy,
                                                  r_pspace_energy=r_pspace_energy,
                                                  pspace_norm=pspace_norm,
                                                  use_norm=use_norm, return_grad=False)
        elif r_pspace_energy is None:
            energy = energies.get_energy_one_proj(wfn, ham,
                                                  pspace_energy=l_pspace_energy,
                                                  pspace_norm=pspace_norm,
                                                  use_norm=use_norm,
                                                  return_grad=False)
        else:
            energy = energies.get_energy_one_proj(wfn, ham,
                                                  pspace_energy=r_pspace_energy,
                                                  pspace_norm=pspace_norm,
                                                  use_norm=use_norm,
                                                  return_grad=False)
        # rotate hamiltonian back
        ham.orb_rotate_jacobi((p, q), -theta)
        return energy

    # initialize (depends on orbital type)
    if ham.orbtype == 'restricted':
        pair_generator = it.combinations(range(ham.nspatial), 2)
        transformation = (np.identity(ham.nspatial), )
    elif ham.orbtype == 'unrestricted':
        def pair_generator():
            yield from it.combinations(range(ham.nspatial), 2)
            yield from it.combinations(range(ham.nspatial, ham.nspin), 2)
        transformation = (np.identity(ham.nspatial), np.identity(ham.nspatial))
    elif ham.orbtype == 'generalized':
        pair_generator = it.combinations(range(ham.nspin), 2)
        transformation = (np.identity(ham.nspin), )

    num_iterations = 50

    print('Orbital optimization')
    print('{0:<5}{1:>11}{2:>15}'.format('Iter', 'wfn energy', 'orb+wfn energy'))
    for i in range(num_iterations):
        delta = 0.0

        # solve wavefunction
        if wfn_solver is not None:
            if wfn_solver_kwargs is None:
                wfn_solver_kwargs = {}
            wfn_solver(wfn, ham, **wfn_solver_kwargs)
        wfn_energy = _objective(0.0, 0, 1)

        # rotate orbitals
        for p, q in pair_generator:
            res = scipy.optimize.minimize_scalar(_objective, args=(p, q), method='brent')
            if res.success:
                delta += res.x
                ham.orb_rotate_jacobi((p, q), res.x)

                # transform
                spin_index = 0
                if ham.orb_type == 'unrestricted':
                    spin_index = p // ham.num_orbs
                    p, q = p % ham.num_orbs, q % ham.num_orbs
                p_col = transformation[spin_index][:, p]
                q_col = transformation[spin_index][:, q]
                (transformation[spin_index][:, p],
                 transformation[spin_index][:, q]) = (np.cos(res.x)*p_col - np.sin(res.x)*q_col,
                                                      np.sin(res.x)*p_col + np.cos(res.x)*q_col)

        rot_energy = _objective(0.0, 0, 1)
        print('{0:<5}{1:>11.8f}{2:>15.8f}'.format(i, wfn_energy, rot_energy))

        # save file
        if save_file != '':
            np.save(save_file, transformation)
        # if last rotation does nothing (i.e. converged)
        if np.isclose(delta, 0.0):
            return transformation, rot_energy
    # if end of loop is reached (i.e. did not converge)
    else:
        print('Orbital optimization did not converge after {0} iterations'.format(num_iterations))
