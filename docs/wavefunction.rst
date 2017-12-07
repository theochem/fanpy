.. _wavefunction:

Wavefunction
============
The wavefunction is the approximation to the eigenfunction of the Hamiltonian. Just like the
Hamiltonian, the wavefunction is often written with respect to the one-electron basis set. Some
build wavefunctions from the vaccuum using creation operators and while build from a reference state
using excitation operators. Regardless of the form of the wavefunction, all wavefunction can be
re-expressed as a linear combination of Slater determinants using the resolution of identity:

.. math::

    \ket{\Psi} &= \sum_{\mathbf{m}} \ket{\mathbf{m}} \left<\mathbf{m} \middle| \Psi \right>\\
    &= \sum_{\mathbf{m}} f(\mathbf{m}) \ket{\mathbf{m}}

where :math:`f(\mathbf{m}) = \left<\mathbf{m} \middle| \Psi \right>` and :math:`\mathbf{m}` is a
Slater determinant. The wavefunction can be entirely described with this function. The :math:`f`
maps a set of parameters and a Slater determinant to coefficient of this Slater determinant in the
wavefunction. The framework of representing the wavefunction as a linear combination of Slater
determinants weighed by an overlap function is called FANCI. (FIXME)

In the FANCI module, all wavefunctions are described as FANCI wavefunctions. Each wavefunction must
define its overlap function. Consequently, the wavefunction parameters must also be defined to
ensure that the overlap function has a consistent input. The number of electrons and spin orbitals
must also be defined as they almost always affect the number and shape of the parameters. This
framework is established in the abstract base class,
:ref:`BaseWavefunction <wfns.wfn.base.BaseWavefunction>`.
