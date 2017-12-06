.. _api:

*****************
API Documentation
*****************

.. module:: wfns

Hamiltonians
============

* :class:`Base Hamiltonian <ham.base.BaseHamiltonian>`
* :class:`Chemical Hamiltonian <ham.chemical.ChemicalHamiltonian>`
* :class:`Seniority Zero Hamiltonian <ham.senzero.SeniorityZeroHamiltonian>`

Objectives
==========

* :class:`Base Objective <objective.base.BaseObjective>`
* Constraints
    * :class:`Normalization Constraint <objective.constraints.norm.NormConstraint>`
* Schrodinger Equation
    * :class:`Base Schrodinger Equation <objective.schrodinger.base.BaseSchrodinger>`
    * :class:`System of Equations <objective.schrodinger.system_nonlinear.SystemEquations>`
    * :class:`Least Squared Sum of Equations <objective.schrodinger.least_squares.LeastSquaresEquations>`
    * :class:`One Sided Energy <objective.schrodinger.onesided_energy.OneSidedEnergy>`
    * :class:`Two Sided Energy <objective.schrodinger.twosided_energy.TwoSidedEnergy>`

Solvers
=======

* :func:`Brute CI Solver <solver.ci.brute>`
* Single Equation Solver
    * :func:`CMA-ES Solver <solver.equation.cma>`
    * :func:`scipy.optimize.minimize Solver <solver.equation.minimize>`
* System of Equations Solver
    * :func:`Least Squares Solver <solver.system.least_squares>`
    * :func:`Root Solver <solver.system.root>`
* Wrapper for External Solver
    * :func:`Scipy Solver Wrapper <solver.wrapper.wrap_scipy>`
    * :func:`skopt Solver Wrapper <solver.wrapper.wrap_skopt>`

Wavefunctions
=============

* :class:`Base Wavefunction <wfn.base.BaseWavefunction>`
* CI Wavefunction

  * :class:`Base CI Wavefunction <wfn.ci.base.CIWavefunction>`
  * :class:`FCI Wavefunction <wfn.ci.fci.FCI>`
  * :class:`DOCI Wavefunction <wfn.ci.doci.DOCI>`
  * :class:`CISD Wavefunction <wfn.ci.cisd.CISD>`
  * :class:`CI-Pair Wavefunction <wfn.ci.ci_pairs.CIPairs>`

* Geminal Wavefunction

  * :class:`Base Geminal Wavefunction <wfn.geminals.base.BaseGeminal>`
  * :class:`APG Wavefunction <wfn.geminals.apg.APG>`
  * :class:`APsetG Wavefunction <wfn.geminals.apsetg.BasicAPsetG>`
  * :class:`APIG Wavefunction <wfn.geminals.apig.APIG>`
  * :class:`AP1roG Wavefunction <wfn.geminals.ap1rog.AP1roG>`
  * :class:`APr2g Wavefunction <wfn.geminals.apr2g.APr2G>`

* Composite Wavefunction

  * Composite of One Wavefunction
      * :class:`Base Composite of One Wavefunction <wfn.composite.base_one.BaseCompositeOneWavefunction>`
      * :class:`Wavefunction with Nonorthogonal Orbitals <wfn.composite.nonorth.NonorthWavefunction>`
      * :class:`Wavefunction with Jacobi Rotated Orbitals <wfn.composite.jacobi.JacobiWavefunction>`
  * :class:`Linear Combination of Wavefunctions <wfn.composite.lincomb.LinearCombinationWavefunction>`

Backend
=======
* Integrals Storage Classes

  * :class:`Base Integrals <backend.integrals.BaseIntegrals>`
  * :class:`One Electron Integrals <backend.integrals.OneElectronIntegrals>`
  * :class:`Two Electron Integrals <backend.integrals.TwoElectronIntegrals>`

* General Math Tools

  * :func:`Binomial Coefficient <backend.math_tools.binomial>`
  * :func:`Adjugate <backend.math_tools.adjugate>`
  * :func:`Permanent Using Combinatorics <backend.math_tools.permanent_combinatoric>`
  * :func:`Permanent Using Ryser Algorithm <backend.math_tools.permanent_ryser>`
  * :func:`Permanent Using Borchardt Theorem <backend.math_tools.permanent_borchardt>`

* Slater Determinant

  * :func:`Check if occupied <backend.slater.occ>`
  * :func:`Check if alpha <backend.slater.is_alpha>`
  * :func:`Convert spin to spatial <backend.slater.spatial_index>`
  * :func:`Get occupation number <backend.slater.total_occ>`
  * :func:`Annhilation Operator <backend.slater.annihilate>`
  * :func:`Creation Operator <backend.slater.create>`
  * :func:`Excitation Operator<backend.slater.excite>`
  * :func:`Ground state Slater determinant <backend.slater.ground>`
  * :func:`Check if internal Slater determinant <backend.slater.is_internal_sd>`
  * :func:`Convert to internal Slater determinant <backend.slater.internal_sd>`
  * :func:`Get occupied orbital indices <backend.slater.occ_indices>`
  * :func:`Get virtual orbital indices <backend.slater.vir_indices>`
  * :func:`Get orbitals shared between Slater determinants <backend.slater.shared>`
  * :func:`Get orbitals different between Slater determinants <backend.slater.diff>`
  * :func:`Combine alpha and beta parts <backend.slater.combine_spin>`
  * :func:`Split a Slater determinant into alpha and beta parts <backend.slater.split_spin>`
  * :func:`Get index after interleaving <backend.slater.interleave_index>`
  * :func:`Get index after deinterleaving <backend.slater.deinterleave_index>`
  * :func:`Interleave Slater determinant <backend.slater.interleave>`
  * :func:`Deinterleave Slater determinant <backend.slater.deinterleave>`
  * :func:`Get spin of Slater determinant <backend.slater.get_spin>`
  * :func:`Get seniority of Slater determinant <backend.slater.get_seniority>`
  * :func:`Get number of transpositions from one ordering to another <backend.slater.find_num_trans>`
  * :func:`Get number of transpositions from one ordering to move an operator <backend.slater.find_num_trans>`
  * :func:`Generate Slater determinants <backend.sd_list.sd_list>`

* Perfect Matching Generator

  * :func:`Complete Graph Perfect Matching Generator <backend.graphs.generate_complete_pmatch>`
  * :func:`Bipartite Graph Perfect Matching Generator <backend.graphs.generate_biclique_pmatch>`


.. Silent api generation
    .. autosummary::
      :toctree: modules/generated

      ham.base.BaseHamiltonian
      ham.chemical.ChemicalHamiltonian
      ham.senzero.SeniorityZeroHamiltonian

      solver.ci.brute
      solver.equation.cma
      solver.equation.minimize
      solver.system.least_squares
      solver.system.root
      solver.wrappers.wrap_scipy
      solver.wrappers.wrap_skopt

      objective.base.BaseObjective
      objective.constraints.norm.NormConstraint
      objective.schrodinger.base.BaseSchrodinger
      objective.schrodinger.system_nonlinear.SystemEquations
      objective.schrodinger.least_squares.LeastSquaresEquations
      objective.schrodinger.onesided_energy.OneSidedEnergy
      objective.schrodinger.twosided_energy.TwoSidedEnergy

      wfn.base.BaseWavefunction
      wfn.ci.base.CIWavefunction
      wfn.ci.fci.FCI
      wfn.ci.doci.DOCI
      wfn.ci.cisd.CISD
      wfn.ci.ci_pairs.CIPairs
      wfn.geminal.base.BaseGeminal
      wfn.geminal.apg.APG
      wfn.geminal.apsetg.BasicAPsetG
      wfn.geminal.apig.APIG
      wfn.geminal.ap1rog.AP1roG
      wfn.geminal.apr2g.APr2G
      wfn.composite.base_one.BaseCompositeOneWavefunction
      wfn.composite.nonorth.NonorthWavefunction
      wfn.composite.jacobi.JacobiWavefunction
      wfn.composite.lincomb.LinearCombinationWavefunction

      backend.integrals.BaseIntegrals
      backend.integrals.OneElectronIntegrals
      backend.integrals.TwoElectronIntegrals

      backend.math_tools.binomial
      backend.math_tools.adjugate
      backend.math_tools.permanent_combinatoric
      backend.math_tools.permanent_ryser
      backend.math_tools.permanent_borchardt
      backend.math_tools.unitary_matrix

      backend.slater.is_internal_sd
      backend.slater.is_sd_compatible
      backend.slater.internal_sd
      backend.slater.occ
      backend.slater.occ_indices
      backend.slater.vir_indices
      backend.slater.total_occ
      backend.slater.is_alpha
      backend.slater.spatial_index
      backend.slater.annihilate
      backend.slater.create
      backend.slater.excite
      backend.slater.ground
      backend.slater.shared_orbs
      backend.slater.diff_orbs
      backend.slater.combine_spin
      backend.slater.split_spin
      backend.slater.interleave_index
      backend.slater.deinterleave_index
      backend.slater.interleave
      backend.slater.deinterleave
      backend.slater.get_spin
      backend.slater.get_seniority
      backend.slater.sign_perm
      backend.slater.sign_swap

      backend.sd_list.sd_list

      backend.graphs.generate_complete_pmatch
      backend.graphs.generate_biclique_pmatch
