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

* :class:`Base Objective <wfns.objective.base.BaseObjective>`
* Constraints

  * :class:`Normalization Constraint <wfns.objective.constraints.norm.NormConstraint>`

* Schrödinger Equation

  * :class:`Base Schrodinger Equation <wfns.objective.schrodinger.base.BaseSchrodinger>`
  * :class:`System of Equations <wfns.objective.schrodinger.system_nonlinear.SystemEquations>`
  * :class:`Least Squared Sum of Equations <wfns.objective.schrodinger.least_squares.LeastSquaresEquations>`
  * :class:`One Sided Energy <wfns.objective.schrodinger.onesided_energy.OneSidedEnergy>`
  * :class:`Two Sided Energy <wfns.objective.schrodinger.twosided_energy.TwoSidedEnergy>`

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

  * :func:`Binomial Coefficient <wfns.backend.math_tools.binomial>`
  * :func:`Adjugate <wfns.backend.math_tools.adjugate>`
  * :func:`Permanent Using Combinatorics <wfns.backend.math_tools.permanent_combinatoric>`
  * :func:`Permanent Using Ryser Algorithm <wfns.backend.math_tools.permanent_ryser>`
  * :func:`Permanent Using Borchardt Theorem <wfns.backend.math_tools.permanent_borchardt>`

* :mod:`Slater Determinant <wfns.backend.slater>`

  * :func:`Check if occupied <wfns.backend.slater.occ>`
  * :func:`Check if alpha <wfns.backend.slater.is_alpha>`
  * :func:`Convert spin to spatial <wfns.backend.slater.spatial_index>`
  * :func:`Get occupation number <wfns.backend.slater.total_occ>`
  * :func:`Annhilation Operator <wfns.backend.slater.annihilate>`
  * :func:`Creation Operator <wfns.backend.slater.create>`
  * :func:`Excitation Operator<wfns.backend.slater.excite>`
  * :func:`Ground state Slater determinant <wfns.backend.slater.ground>`
  * :func:`Check if internal Slater determinant <wfns.backend.slater.is_internal_sd>`
  * :func:`Convert to internal Slater determinant <wfns.backend.slater.internal_sd>`
  * :func:`Get occupied orbital indices <wfns.backend.slater.occ_indices>`
  * :func:`Get virtual orbital indices <wfns.backend.slater.vir_indices>`
  * :func:`Get orbitals shared between Slater determinants <wfns.backend.slater.shared_orbs>`
  * :func:`Get orbitals different between Slater determinants <wfns.backend.slater.diff_orbs>`
  * :func:`Combine alpha and beta parts <wfns.backend.slater.combine_spin>`
  * :func:`Split a Slater determinant into alpha and beta parts <wfns.backend.slater.split_spin>`
  * :func:`Get index after interleaving <wfns.backend.slater.interleave_index>`
  * :func:`Get index after deinterleaving <wfns.backend.slater.deinterleave_index>`
  * :func:`Interleave Slater determinant <wfns.backend.slater.interleave>`
  * :func:`Deinterleave Slater determinant <wfns.backend.slater.deinterleave>`
  * :func:`Get spin of Slater determinant <wfns.backend.slater.get_spin>`
  * :func:`Get seniority of Slater determinant <wfns.backend.slater.get_seniority>`
  * :func:`Get signature of the permutation that sorts a set of annihilators. <wfns.backend.slater.sign_perm>`
  * :func:`Get signature of moving a creation operator to a specific position. <wfns.backend.slater.sign_swap>`
  * :func:`Generate Slater determinants <wfns.backend.sd_list.sd_list>`

* Perfect Matching Generator

  * :func:`Complete Graph Perfect Matching Generator <wfns.backend.graphs.generate_complete_pmatch>`
  * :func:`Bipartite Graph Perfect Matching Generator <wfns.backend.graphs.generate_biclique_pmatch>`

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

      wfns.objective.base.BaseObjective
      wfns.objective.constraints.norm.NormConstraint
      wfns.objective.schrodinger.base.BaseSchrodinger
      wfns.objective.schrodinger.system_nonlinear.SystemEquations
      wfns.objective.schrodinger.least_squares.LeastSquaresEquations
      wfns.objective.schrodinger.onesided_energy.OneSidedEnergy
      wfns.objective.schrodinger.twosided_energy.TwoSidedEnergy

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

      wfns.backend.math_tools.binomial
      wfns.backend.math_tools.adjugate
      wfns.backend.math_tools.permanent_combinatoric
      wfns.backend.math_tools.permanent_ryser
      wfns.backend.math_tools.permanent_borchardt
      wfns.backend.math_tools.unitary_matrix

      wfns.backend.slater
      wfns.backend.slater.is_internal_sd
      wfns.backend.slater.is_sd_compatible
      wfns.backend.slater.internal_sd
      wfns.backend.slater.occ
      wfns.backend.slater.occ_indices
      wfns.backend.slater.vir_indices
      wfns.backend.slater.total_occ
      wfns.backend.slater.is_alpha
      wfns.backend.slater.spatial_index
      wfns.backend.slater.annihilate
      wfns.backend.slater.create
      wfns.backend.slater.excite
      wfns.backend.slater.ground
      wfns.backend.slater.shared_orbs
      wfns.backend.slater.diff_orbs
      wfns.backend.slater.combine_spin
      wfns.backend.slater.split_spin
      wfns.backend.slater.interleave_index
      wfns.backend.slater.deinterleave_index
      wfns.backend.slater.interleave
      wfns.backend.slater.deinterleave
      wfns.backend.slater.get_spin
      wfns.backend.slater.get_seniority
      wfns.backend.slater.sign_perm
      wfns.backend.slater.sign_swap

      wfns.backend.sd_list.sd_list

      wfns.backend.graphs.generate_complete_pmatch
      wfns.backend.graphs.generate_biclique_pmatch
