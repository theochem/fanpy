.. _api:

*****************
API Documentation
*****************

Hamiltonians
============

* :class:`Base Hamiltonian <wfns.ham.base.BaseHamiltonian>`
* :class:`Generalized Base Hamiltonian <wfns.ham.generalized_base.BaseGeneralizedHamiltonian>`
* :class:`Unrestricted Base Hamiltonian <wfns.ham.unrestricted_base.BaseUnrestrictedHamiltonian>`
* :class:`Restricted Base Hamiltonian <wfns.ham.restricted_base.BaseRestrictedHamiltonian>`
* :class:`Generalized Chemical Hamiltonian <wfns.ham.generalized_chemical.GeneralizedChemicalHamiltonian>`
* :class:`Unrestricted Chemical Hamiltonian <wfns.ham.unrestricted_chemical.UnrestrictedChemicalHamiltonian>`
* :class:`Restricted Chemical Hamiltonian <wfns.ham.restricted_chemical.RestrictedChemicalHamiltonian>`
* :class:`Seniority Zero Hamiltonian <wfns.ham.senzero.SeniorityZeroHamiltonian>`

Wavefunctions
=============

* :class:`Base Wavefunction <wfns.wfn.base.BaseWavefunction>`
* CI Wavefunction

  * :class:`Base CI Wavefunction <wfns.wfn.ci.base.CIWavefunction>`
  * :class:`FCI Wavefunction <wfns.wfn.ci.fci.FCI>`
  * :class:`DOCI Wavefunction <wfns.wfn.ci.doci.DOCI>`
  * :class:`CISD Wavefunction <wfns.wfn.ci.cisd.CISD>`
  * :class:`CI-Pair Wavefunction <wfns.wfn.ci.ci_pairs.CIPairs>`

* Geminal Wavefunction

  * :class:`Base Geminal Wavefunction <wfns.wfn.geminal.base.BaseGeminal>`
  * :class:`APG Wavefunction <wfns.wfn.geminal.apg.APG>`
  * :class:`APsetG Wavefunction <wfns.wfn.geminal.apsetg.BasicAPsetG>`
  * :class:`APIG Wavefunction <wfns.wfn.geminal.apig.APIG>`
  * :class:`AP1roG Wavefunction <wfns.wfn.geminal.ap1rog.AP1roG>`
  * :class:`APr2g Wavefunction <wfns.wfn.geminal.apr2g.APr2G>`

* Composite Wavefunction

  * Composite of One Wavefunction

    * :class:`Base Composite of One Wavefunction <wfns.wfn.composite.base_one.BaseCompositeOneWavefunction>`
    * :class:`Wavefunction with Nonorthogonal Orbitals <wfns.wfn.composite.nonorth.NonorthWavefunction>`
    * :class:`Wavefunction with Jacobi Rotated Orbitals <wfns.wfn.composite.jacobi.JacobiWavefunction>`

  * :class:`Linear Combination of Wavefunctions <wfns.wfn.composite.lincomb.LinearCombinationWavefunction>`

* Network Wavefunction

  * :class:`KerasNetwork <wfns.wfn.network.keras_network.KerasNetwork>`


Objectives
==========

* :class:`Base Objective <wfns.eqn.base.BaseObjective>`
* Constraints

  * :class:`Normalization Constraint <wfns.eqn.constraints.norm.NormConstraint>`

* Schrödinger Equation

  * :class:`Base Schrodinger Equation <wfns.eqn.base.BaseSchrodinger>`
  * :class:`System of Equations <wfns.eqn.system_nonlinear.SystemEquations>`
  * :class:`Least Squared Sum of Equations <wfns.eqn.least_squares.LeastSquaresEquations>`
  * :class:`One Sided Energy <wfns.eqn.onesided_energy.OneSidedEnergy>`
  * :class:`Two Sided Energy <wfns.eqn.twosided_energy.TwoSidedEnergy>`

Solvers
=======

* :func:`Brute CI Solver <wfns.solver.ci.brute>`
* Single Equation Solver

  * :func:`CMA-ES Solver <wfns.solver.equation.cma>`
  * :func:`scipy.optimize.minimize Solver <wfns.solver.equation.minimize>`

* System of Equations Solver

  * :func:`Least Squares Solver <wfns.solver.system.least_squares>`
  * :func:`Root Solver <wfns.solver.system.root>`

* Wrapper for External Solver

  * :func:`Scipy Solver Wrapper <wfns.solver.wrappers.wrap_scipy>`
  * :func:`skopt Solver Wrapper <wfns.solver.wrappers.wrap_skopt>`

Backend
=======
* General Math Tools

  * :func:`Binomial Coefficient <wfns.tools.math_tools.binomial>`
  * :func:`Adjugate <wfns.tools.math_tools.adjugate>`
  * :func:`Permanent Using Combinatorics <wfns.tools.math_tools.permanent_combinatoric>`
  * :func:`Permanent Using Ryser Algorithm <wfns.tools.math_tools.permanent_ryser>`
  * :func:`Permanent Using Borchardt Theorem <wfns.tools.math_tools.permanent_borchardt>`

* :mod:`Slater Determinant <wfns.tools.slater>`

  * :func:`Check if occupied <wfns.tools.slater.occ>`
  * :func:`Check if alpha <wfns.tools.slater.is_alpha>`
  * :func:`Convert spin to spatial <wfns.tools.slater.spatial_index>`
  * :func:`Get occupation number <wfns.tools.slater.total_occ>`
  * :func:`Annhilation Operator <wfns.tools.slater.annihilate>`
  * :func:`Creation Operator <wfns.tools.slater.create>`
  * :func:`Excitation Operator<wfns.tools.slater.excite>`
  * :func:`Ground state Slater determinant <wfns.tools.slater.ground>`
  * :func:`Check if internal Slater determinant <wfns.tools.slater.is_internal_sd>`
  * :func:`Convert to internal Slater determinant <wfns.tools.slater.internal_sd>`
  * :func:`Get occupied orbital indices <wfns.tools.slater.occ_indices>`
  * :func:`Get virtual orbital indices <wfns.tools.slater.vir_indices>`
  * :func:`Get orbitals shared between Slater determinants <wfns.tools.slater.shared_orbs>`
  * :func:`Get orbitals different between Slater determinants <wfns.tools.slater.diff_orbs>`
  * :func:`Combine alpha and beta parts <wfns.tools.slater.combine_spin>`
  * :func:`Split a Slater determinant into alpha and beta parts <wfns.tools.slater.split_spin>`
  * :func:`Get index after interleaving <wfns.tools.slater.interleave_index>`
  * :func:`Get index after deinterleaving <wfns.tools.slater.deinterleave_index>`
  * :func:`Interleave Slater determinant <wfns.tools.slater.interleave>`
  * :func:`Deinterleave Slater determinant <wfns.tools.slater.deinterleave>`
  * :func:`Get spin of Slater determinant <wfns.tools.slater.get_spin>`
  * :func:`Get seniority of Slater determinant <wfns.tools.slater.get_seniority>`
  * :func:`Get signature of the permutation that sorts a set of annihilators. <wfns.tools.slater.sign_perm>`
  * :func:`Get signature of moving a creation operator to a specific position. <wfns.tools.slater.sign_swap>`
  * :func:`Generate Slater determinants <wfns.tools.sd_list.sd_list>`

* Perfect Matching Generator

  * :func:`Complete Graph Perfect Matching Generator <wfns.tools.graphs.generate_complete_pmatch>`
  * :func:`Bipartite Graph Perfect Matching Generator <wfns.tools.graphs.generate_biclique_pmatch>`

Scripts
=======
* :ref:`Run calculation <script_run_calc>`
* :ref:`Make calculation script <script_make_script>`

.. Silent api generation
    .. autosummary::
      :toctree: modules/generated

      wfns.ham.base.BaseHamiltonian
      wfns.ham.generalized_base.BaseGeneralizedHamiltonian
      wfns.ham.unrestricted_base.BaseUnrestrictedHamiltonian
      wfns.ham.restricted_base.BaseRestrictedHamiltonian
      wfns.ham.generalized_chemical.GeneralizedChemicalHamiltonian
      wfns.ham.unrestricted_chemical.UnrestrictedChemicalHamiltonian
      wfns.ham.restricted_chemical.RestrictedChemicalHamiltonian
      wfns.ham.senzero.SeniorityZeroHamiltonian

      wfns.solver.ci.brute
      wfns.solver.equation.cma
      wfns.solver.equation.minimize
      wfns.solver.system.least_squares
      wfns.solver.system.root
      wfns.solver.wrappers
      wfns.solver.wrappers.wrap_scipy
      wfns.solver.wrappers.wrap_skopt

      wfns.eqn.base.BaseObjective
      wfns.eqn.constraints.norm.NormConstraint
      wfns.eqn.base.BaseSchrodinger
      wfns.eqn.system_nonlinear.SystemEquations
      wfns.eqn.least_squares.LeastSquaresEquations
      wfns.eqn.onesided_energy.OneSidedEnergy
      wfns.eqn.twosided_energy.TwoSidedEnergy

      wfns.wfn.base.BaseWavefunction
      wfns.wfn.ci.base.CIWavefunction
      wfns.wfn.ci.fci.FCI
      wfns.wfn.ci.doci.DOCI
      wfns.wfn.ci.cisd.CISD
      wfns.wfn.ci.ci_pairs.CIPairs
      wfns.wfn.geminal.base.BaseGeminal
      wfns.wfn.geminal.apg.APG
      wfns.wfn.geminal.apsetg.BasicAPsetG
      wfns.wfn.geminal.apig.APIG
      wfns.wfn.geminal.ap1rog.AP1roG
      wfns.wfn.geminal.apr2g.APr2G
      wfns.wfn.composite.base_one.BaseCompositeOneWavefunction
      wfns.wfn.composite.nonorth.NonorthWavefunction
      wfns.wfn.composite.jacobi.JacobiWavefunction
      wfns.wfn.composite.lincomb.LinearCombinationWavefunction
      wfns.wfn.network.keras_network.KerasNetwork

      wfns.tools.math_tools.binomial
      wfns.tools.math_tools.adjugate
      wfns.tools.math_tools.permanent_combinatoric
      wfns.tools.math_tools.permanent_ryser
      wfns.tools.math_tools.permanent_borchardt
      wfns.tools.math_tools.unitary_matrix

      wfns.tools.slater
      wfns.tools.slater.is_internal_sd
      wfns.tools.slater.is_sd_compatible
      wfns.tools.slater.internal_sd
      wfns.tools.slater.occ
      wfns.tools.slater.occ_indices
      wfns.tools.slater.vir_indices
      wfns.tools.slater.total_occ
      wfns.tools.slater.is_alpha
      wfns.tools.slater.spatial_index
      wfns.tools.slater.annihilate
      wfns.tools.slater.create
      wfns.tools.slater.excite
      wfns.tools.slater.ground
      wfns.tools.slater.shared_orbs
      wfns.tools.slater.diff_orbs
      wfns.tools.slater.combine_spin
      wfns.tools.slater.split_spin
      wfns.tools.slater.interleave_index
      wfns.tools.slater.deinterleave_index
      wfns.tools.slater.interleave
      wfns.tools.slater.deinterleave
      wfns.tools.slater.get_spin
      wfns.tools.slater.get_seniority
      wfns.tools.slater.sign_perm
      wfns.tools.slater.sign_swap

      wfns.tools.sd_list.sd_list

      wfns.tools.graphs.generate_complete_pmatch
      wfns.tools.graphs.generate_biclique_pmatch
