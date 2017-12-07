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
of the basis set using a projection operator:

.. math::

    \hat{H}
    &= \sum_{ij} \ket{\phi_i} \bra{\phi_i} \hat{H} \ket{\phi_j} \bra{\phi_j}\\
    &\hspace{2em}
       + \sum_{i<j} \sum_{k<l} \ket{\phi_i \phi_j} \bra{\phi_i \phi_j} \hat{H} \ket{\phi_k \phi_l}
         \bra{\phi_k \phi_l}\\
    &\hspace{2em}
       + \sum_{i<j<k} \sum_{l<m<n} \ket{\phi_i \phi_j \phi_k}
         \bra{\phi_i \phi_j \phi_k} \hat{H} \ket{\phi_l \phi_m \phi_n} \bra{\phi_l \phi_m \phi_n}
       + \dots\\
    &= \sum_{n=1}^\infty \sum_{i_1 < i_2 < \dots < i_n} \sum_{j_1 < j_2 < \dots < j_n}
       \ket{\phi_{i_1} \dots \phi_{i_n}}
       \bra{\phi_{ \phi_{i_1} \dots \phi_{i_n}}} \hat{H} \ket{\phi_{j_1} \dots \phi_{j_n}}
       \bra{\phi_{j_1} \dots \phi_{j_n}}

Fortunately, most Hamiltonians only contain terms with a small number of bodies (i.e. electrons)
resulting in truncation in the summation. For example, a
:ref:`molecular Hamiltonian <wfns.ham.chemical.ChemicalHamiltonian>` only contains up to bodies, so
the sum truncates at the second order:

.. math::

    \hat{H}
    &= \sum_{ij} \ket{\phi_i} \bra{\phi_i} \hat{H} \ket{\phi_j} \bra{\phi_j}
       + \sum_{i<j} \sum_{k<l} \ket{\phi_i \phi_j} \bra{\phi_i \phi_j} \hat{H} \ket{\phi_k \phi_l}
         \bra{\phi_k \phi_l}\\
    &= \sum_{ij} a^\dagger_i \bra{} a_i \hat{H} a^\dagger_j \ket{} a_j
       + \sum_{i<j} \sum_{k<l} a^\dagger_i a^\dagger_j
       \bra{} a_j a_i \hat{H} a^\dagger_k \phi_l \ket{} a_l a_k\\
    &= \sum_{ij} h_{ij} a^\dagger_i a_j
       + \sum_{i<j} \sum_{k<l} g_{ijkl} a^\dagger_i a^\dagger_j a_l a_k

where :math:`h_{ij} = \bra{} a_i \hat{H} a^\dagger_j \ket{}` and
:math:`g_{ijkl} = \bra{} a_j a_i \hat{H} a^\dagger_k \phi_l \ket{}`.

Therefore, all Hamiltonians can be expressed within the one-electron basis set in a similar manner
and we can construct a framework from which all possible Hamiltonian can be built. All Hamiltonians
only need to contain the integrals and the method by which these integrals are applied to the
wavefunction. In the FANCI module, the objectives represent the Schrodinger equation with
:math:`\bra{\Phi} \hat{H} \ket{\Psi}` and :math:`\bra{\Phi} \hat{H} \ket{\Phi}`, where
:math:`\Phi` is a Slater determinant and :math:`\Psi` is the wavefunction. This framework is
established in the abstract base class, :ref:`BaseHamiltonian <wfns.ham.base.BaseHamiltonian>`.
