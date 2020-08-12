.. _objective:

Objective
=========
Given the Hamiltonian, :math:`\hat{H}`, and the wavefunction, :math:`\Psi`, we seek to optimize the
parameters such that the Schrödinger equation is satisfied:

.. math::

    \hat{H} \left| \Psi \right> = E \left| \Psi \right>

We can convert this equation into other objectives such that the solutions to the objectives
correspond to the solutions to the Schrödinger equation.

Energy
------
The wavefunction is often optimized variationally by solving for the energy as the expectation value
of the Hamiltonian with the given wavefunction. This is equivalent to integrating the Schrödinger
equation with the wavefunction and rearranging for the energy:

.. math::

    \left< \Psi \middle| \hat{H} \middle| \Psi \right> &= E \left< \Psi \middle| \Psi \right>\\

.. math::

    E = \frac{\left< \Psi \middle| \hat{H} \middle| \Psi \right>}{\left< \Psi \middle| \Psi \right>}

The variational method involves minimizing this energy, where the wavefunction that results in the
minimum energy corresponds to the best approximation for the ground state. However, except for some
special cases, :math:`\left< \Psi \middle| \hat{H} \middle| \Psi \right>` is too expensive to
compute. In fact, computing even :math:`\left< \Psi \middle| \Psi \right>` may not be practical.
Unless there is some specialized structure that simplifies
:math:`\left< \Psi \middle| \hat{H} \middle| \Psi \right>` and
:math:`\left< \Psi \middle| \Psi \right>` to something tractible, these terms must be expanded out
in terms of Slater determinants, which would result in intractibly large number of evaluations for a
complete description. At the expense of full accuracy, the cost for evaluation (and optimization)
can be reduced by limiting the number of terms used while integrating. In the FANCI module, the
number of terms in the integration can be limited by (1) integrating the Schrödinger equation with
some reference wavefunction or (2) using subsets of the Slater determinants to project out less
important components of the wavefunction.

.. _integrateref:
Integrating With A Reference Wavefunction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When integrating the Schrödinger equation, we can use a reference wavefunction, :math:`\Phi`,
instead of the original, :math:`\Psi`.

.. math::

    E = \frac{\left< \Phi \middle| \hat{H} \middle| \Psi \right>}{\left< \Phi \middle| \Psi \right>}

In the FANCI module, only the following reference wavefunctions are supported:

* Slater determinant: :math:`\left| \Phi \right> = \left| \mathbf{m} \right>`

.. math::

    E = \frac{\left< \mathbf{m} \middle| \hat{H} \middle| \Psi \right>}
             {\left< \mathbf{m} \middle| \Psi \right>}

* Truncation of the original wavefunction:
  :math:`\left| \Phi \right> = \sum_{\mathbf{m} \in S_{trunc}} f(\mathbf{m}) \left| \mathbf{m} \right>`

.. math::

    E = \frac{
       \sum_{\mathbf{m} \in S_{trunc}}
       f^*(\mathbf{m}) \left< \mathbf{m} \middle| \hat{H} \middle| \Psi \right>
    }{
       \sum_{\mathbf{m} \in S_{trunc}} f^*(\mathbf{m}) \left< \mathbf{m} \middle| \Psi \right>
    }

* CI wavefunction:
  :math:`\left| \Phi \right> = \sum_{\mathbf{m} \in S_{ci}} c_{\mathbf{m}} \left| \mathbf{m} \right>`

.. math::

    E = \frac{
       \sum_{\mathbf{m} \in S_{ci}}
       c^*_{\mathbf{m}} \left< \mathbf{m} \middle| \hat{H} \middle| \Psi \right>
    }{
       \sum_{\mathbf{m} \in S_{ci}} c^*_{\mathbf{m}} \left< \mathbf{m} \middle| \Psi \right>
    }

This objective is implemented in class
:class:`EnergyOneSideProjection <wfns.eqn.energy_oneside.OneSidedEnergy>`.

Projecting Out Slater Determinants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In the integrated Schrödinger equation, we can apply the resolution of identity to every instance of
the wavefunction :math:`\Psi`.

.. math::

    E &= \frac{\left< \Psi \middle| \hat{H} \middle| \Psi \right>}{\left< \Psi \middle| \Psi \right>}\\
    &= \frac{
        \left< \Psi \right|
        \bigg( \sum_{\mathbf{m}}  \left| \mathbf{m} \middle> \middle< \mathbf{m} \right| \bigg)
        \hat{H}
        \bigg( \sum_{\mathbf{n}} \left| \mathbf{n} \middle> \middle< \mathbf{n} \right|  \bigg)
        \left| \Psi \right>
    }{
        \left< \Psi \right|
        \bigg( \sum_{\mathbf{p}}  \left| \mathbf{p} \middle> \middle< \mathbf{p} \right| \bigg)
        \left| \Psi \right>
    }\\
    &= \frac{
        \sum_{\mathbf{m}} \sum_{\mathbf{n}}
        \left< \Psi \middle| \mathbf{m} \middle>
        \middle< \mathbf{m} \middle| \hat{H} \middle| \mathbf{n} \middle>
        \middle< \mathbf{n} \middle| \Psi \right>
    }{
        \sum_{\mathbf{p}}
        \left< \Psi \middle| \mathbf{p} \middle> \middle< \mathbf{p} \middle| \Psi \right>
    }

We can sum over a subset of the Slater determinants to reduce the number of evaluations:

.. math::

    E = \frac{
        \sum_{\mathbf{m} \in S_{left}} \sum_{\mathbf{n} \in S_{right}}
        \left< \Psi \middle| \mathbf{m} \right>
        \left< \mathbf{m} \middle| \hat{H} \middle| \mathbf{n} \right>
        \left< \mathbf{n} \middle| \Psi \right>
    }{
        \sum_{\mathbf{p} \in S_{norm}}
        \left< \Psi \middle| \mathbf{p} \right> \left< \mathbf{p} \middle| \Psi \right>
    }

This objective is implemented in class
:class:`TwoSidedEnergy <wfns.eqn.twosided_energy.TwoSidedEnergy>`.


Projected Schrödinger Equation
------------------------------
Just like above, we can insert a resolution of identity into the Schrödinger equation to decompose
it into smaller components, which can be projected out if they are insignificant.

.. math::

    \hat{H} \left| \Psi \right> &= E \left| \Psi \right>\\
    \sum_{\mathbf{m}} \left| \mathbf{m} \middle> \middle< \mathbf{m} \middle| \hat{H} \middle| \Psi \right>
    &= E \sum_{\mathbf{m}} \left| \mathbf{m} \middle> \middle< \mathbf{m} \middle| \Psi \right>\\

.. math::

    \sum_{\mathbf{m}} \left| \mathbf{m} \right>
    \left(
        \left< \mathbf{m} \middle| \hat{H} \middle| \Psi \right> -
        E \left< \mathbf{m} \middle| \Psi \right>
    \right) = 0

Since the Slater determinants are all orthogonal to one another, we can analytically separate the
Schrödinger equation into a system of equations - one equation for each Slater determinant:

.. math::

    \left< \mathbf{m} \middle| \hat{H} \middle| \Psi \right> - E \left< \mathbf{m} \middle| \Psi \right> = 0
    \; \forall \; \mathbf{m}

If all equation in the system of equations are satisfied, then the Schrödinger equation is
satisfied. Then, we can ignore Slater determinants where both
:math:`\left< \mathbf{m} \middle| \hat{H} \middle| \Psi \right> \approx 0` and
:math:`\left< \mathbf{m} \middle| \Psi \right> \approx 0`, because
:math:`\left< \mathbf{m} \middle| \hat{H} \middle| \Psi \right> - E \left< \mathbf{m} \middle| \Psi \right> \approx 0`.

The objective for the projected Schrödinger equation is implemented in class
:class:`ProjectedSchrodinger <wfns.eqn.projected.SystemEquations>`.

Projection Space
~~~~~~~~~~~~~~~~
In essence, the insignificant ("trivially" satisfied) parts of the Schrödinger equation can be
removed with a projection operator. In the FANCI module, the projection space can include

* Slater determinants
* CI wavefunctions, which is equivalent to linearly combining the equations that correspond to the
  Slater determinants in the CI wavefunction:

.. math::

    \left< \Phi \middle| \hat{H} \middle| \Psi \right> - E \left< \Phi \middle| \Psi \right> &= 0\\
    \sum_{\mathbf{m} \in S_{ci}} c^*_{\mathbf{m}} \left< \mathbf{m} \middle| \hat{H} \middle| \Psi \right>
    - E \sum_{\mathbf{m} \in S_{ci}} c^*_{\mathbf{m}} \left< \mathbf{m} \middle| \Psi \right> &= 0\\
    \sum_{\mathbf{m} \in S_{ci}} c^*_{\mathbf{m}}
    \left(
        \left< \mathbf{m} \middle| \hat{H} \middle| \Psi \right> - E \left< \mathbf{m} \middle| \Psi \right>
    \right)
    &= 0

Energy
~~~~~~
The energy in the projected Schrödinger equation can be treated in different ways. The energy can
be

* a fixed number - it would not change in the course of the optimization.

* a variable - it will be optimizied like all the other parameters in the Schrödinger equation.

* computed by integrating the Schrödinger equation with respect to a reference wavefunction - see
  :ref:`Integrating With A Reference Wavefunction <integrateref>` for details.

Constraints
~~~~~~~~~~~
Since the Schrödinger equation is treated as a system of equations, it is quite easy to put
constraints into the objective - simply add more equations to the system. So far, only the
normalization constraint is implemented:

.. math::

    \left< \Phi \middle| \Psi \right> - 1 = 0

where :math:`\Phi` can be

* a Slater determinant

.. math::

    \left< \mathbf{m} \middle| \Psi \right> - 1 = 0

* a CI wavefunction

.. math::

    \sum_{\mathbf{m} \in S_{ci}} c^*_{\mathbf{m}} \left< \mathbf{m} \middle| \Psi \right> - 1 = 0

* a truncated form of the wavefunction

.. math::

    \sum_{\mathbf{m} \in S_{trunc}} f^*(\mathbf{m}) \left< \mathbf{m} \middle| \Psi \right> - 1 = 0

Though there is no abstract base class for the constraints specifically, they should follow the same
structure as the abstract base class, :class:`BaseObjective <wfns.eqn.base.BaseObjective>`.
