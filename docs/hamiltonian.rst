.. _hamiltonian:

Hamiltonian
===========
..
   The exact nonrelativistic, time-independent chemical Hamiltonian, :math:`\mathscr{H}`, involves the
   interactions between the nuclei (denoted by index :math:`A`) and the electrons (denoted by index
   :math:`i`):

   .. math::

       \mathscf{H} &= - \sum_A \frac{1}{2M_A} \nabla_A^2 + \sum_{A<B} \frac{Z_A Z_B}{R_{AB}}
                      - \sum_i \frac{1}{2} \nabla_i^2 + \sum_{i<j} \frac{1}{r_{ij}}
                      - \sum_A \sum_i \frac{Z_A}{r_{iA}}

   where all units are in atomic units (atomic units will always be used in this module).

   In electronic structure, we often separate out the electronic component from the nuclear component.

   .. math::

       \mathscf{H} &= \mathscf{H_{\mathrm{nuc}} + \mathscf{H_{\mathrm{el}}\\
       \mathscf{H_{\mathrm{el}} &= - \sum_i \frac{1}{2} \nabla_i^2 + \sum_{i<j} \frac{1}{r_{ij}}
                                   - \sum_A \sum_i \frac{Z_A}{r_{iA}}\\
       \mathscf{H_{\mathrm{nuc}} &= - \sum_A \frac{1}{2M_A} \nabla_A^2 + \sum_{A<B} \frac{Z_A Z_B}{R_{AB}}

   Using the Born-Oppenheimer approximation, the solution to the Hamiltonian, :math:`\mathscr{H}`,
   can be decomposed into the nuclear and electronic components.

The Hamiltonian describes the system and the interactions within the system. Within a finite
one-electron basis set, :math:`\{\phi_i\}`, we can represent the Hamiltonian explicitly in terms
of the Slater determinants using a projection operator:

.. math::

    \hat{H}
    &= \sum_{ij} \left| \phi_i \middle> \middle< \phi_i \right| \hat{H}
       \left| \phi_j \middle> \middle< \phi_j \right|\\
    &\hspace{2em}
       + \sum_{i<j} \sum_{k<l} \left| \phi_i \phi_j \middle> \middle< \phi_i \phi_j \right| \hat{H}
         \left| \phi_k \phi_l \middle> \middle< \phi_k \phi_l \right|\\
    &\hspace{2em}
       + \sum_{i<j<k} \sum_{l<m<n}
         \left| \phi_i \phi_j \phi_k \middle> \middle< \phi_i \phi_j \phi_k \right| \hat{H}
         \left| \phi_l \phi_m \phi_n \middle> \middle< \phi_l \phi_m \phi_n \right|\\
    &\hspace{2em}
       + \dots\\
    &= \sum_{n=1}^\infty \sum_{i_1 < i_2 < \dots < i_n} \sum_{j_1 < j_2 < \dots < j_n}
       \left| \phi_{i_1} \dots \phi_{i_n} \middle> \middle< \phi_{i_1} \dots \phi_{i_n} \right|
       \hat{H}
       \left| \phi_{j_1} \dots \phi_{j_n} \middle> \middle< \phi_{j_1} \dots \phi_{j_n} \right|

Though this form of the Hamiltonian is infeasible (it requires a sum over all Fock space), it
demonstrates that the Hamiltonian operator can be described as the integrals of the operator
against different Slater determinants:
:math:`\left< \phi_{i_1} \dots \phi_{i_n} \middle| \hat{H} \middle| \phi_{j_1} \dots \phi_{j_n} \right>`.
With an orthonormal one-electron basis set, the corresponding Slater determinants are orthonormal to
one another. (FIXME) Then, by the Slater-Condon rule, the Hamiltonian only needs to consider as many
orbitals as the number of electrons that are involved in it. For example, a one-electron operator
will only need to consider one-electron components of the Slater determinants. When the Hamiltonian
involves operators of different numbers of electrons, the Hamiltonian can be separated into
different components. In the :class:`MolecularHamiltonian <fanpy.ham.chemical.ChemicalHamiltonian>`,
the Hamiltonian can be decomposoed into the one and two-electron operators:

.. math::

    \hat{H}_{one}
    &= \left(
           \sum_{i} a^\dagger_i \left< \phi_i \right|
       \right) \hat{H}_{one} \left(
           \sum_{j} \left| \phi_j \right> a_j
       \right)\\
    &= \sum_{ij} a^\dagger_i \left< \phi_i \middle| \hat{H}_{one} \middle| \phi_j \right> a_j\\
    &= \sum_{ij} h_{ij} a^\dagger_i a_j

.. math::

    \hat{H}_{two}
    &= \left(
           \sum_{i<j} a^\dagger_i a^\dagger_j \left< \phi_i \phi_j \right|
       \right) \hat{H}_{two} \left(
           \sum_{k<l} \left| \phi_k \phi_l \right> a_l a_k
       \right)\\
    &= \sum_{i<j} \sum_{k<l} a^\dagger_i a^\dagger_j
       \left< \phi_i \phi_j \middle| \hat{H}_{two} \middle| \phi_k \phi_l \right> a_l a_k\\
    &= \sum_{i<j} \sum_{k<l} g_{ijkl} a^\dagger_i a^\dagger_j a_l a_k

.. math::

    \hat{H}
    &= \hat{H}_{one} + \hat{H}_{two}\\
    &= \sum_{ij} h_{ij} a^\dagger_i a_j
       + \sum_{i<j} \sum_{k<l} g_{ijkl} a^\dagger_i a^\dagger_j a_l a_k

where :math:`h_{ij} = \left< \phi_i \right| \hat{H}_{one} \left| \phi_j \right>` and
:math:`g_{ijkl} = \left< \phi_i \phi_j \right| \hat{H}_{two} \left| \phi_k \phi_l \right>`. Note
that the projection operators in the first Equation, i.e.

.. math::

    \sum_{i_1 < i_2 < \dots < i_n}
    \left| \phi_{i_1} \dots \phi_{i_n} \middle> \middle< \phi_{i_1} \dots \phi_{i_n} \right|

and

.. math::

    \sum_{j_1 < j_2 < \dots < j_n}
    \left| \phi_{j_1} \dots \phi_{j_n} \middle> \middle< \phi_{j_1} \dots \phi_{j_n} \right|

can be removed, because they can be interpreted as resolution of identity within the given
one-electron basis set.

Therefore, all Hamiltonians can be expressed within the one-electron basis set in a similar manner
and we can construct a framework from which all possible Hamiltonian can be built. All Hamiltonians
only need to contain the integrals and the method by which these integrals are applied to the
wavefunction. In the FANCI module, the objectives represent the Schrödinger equation with
:math:`\left< \Phi \middle| \hat{H} \middle| \Psi \right>` and
:math:`\left< \Phi \middle| \hat{H} \middle| \Phi \right>`, where
:math:`\Phi` is a Slater determinant and :math:`\Psi` is the wavefunction. This framework is
established in the abstract base class, :class:`BaseHamiltonian <fanpy.ham.base.BaseHamiltonian>`.
