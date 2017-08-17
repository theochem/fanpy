.. _api:

*****************
API Documentation
*****************

.. module:: wfns

Hamiltonians
============

* :class:`Chemical Hamiltonian <hamiltonian.chemical_hamiltonian.ChemicalHamiltonian>`
* :class:`Seniority Zero Hamiltonian <hamiltonian.sen0_hamiltonian.SeniorityZeroHamiltonian>`

Solvers
=======

* :func:`CI Solver <solver.ci_solver.eigen_solve>`
* :func:`System of Nonlinear Equations Solver <solver.system_solver.optimize_wfn_system>`
* :func:`Projected Variational Solver <solver.equation_solver.optimize_wfn_variational>`
* :func:`Orbital Rotated Wavefunction Solver <solver.orb_solver.optimize_wfn_orbitals_jacobi>`
* :func:`Orbital Rotated Hamiltonian Solver <solver.orb_solver.optimize_ham_orbitals_jacobi>`

Wavefunctions
=============

* :class:`Base Wavefunction <wavefunction.base_wavefunction.BaseWavefunction>`
* CI Wavefunction

  * :class:`Base CI Wavefunction <wavefunction.ci.ci_wavefunction.CIWavefunction>`
  * :class:`FCI Wavefunction <wavefunction.ci.fci.FCI>`
  * :class:`DOCI Wavefunction <wavefunction.ci.doci.DOCI>`
  * :class:`CISD Wavefunction <wavefunction.ci.cisd.CISD>`
  * :class:`CI-Pair Wavefunction <wavefunction.ci.ci_pairs.CIPairs>`

* Geminal Wavefunction

  * :class:`Base Geminal Wavefunction <wavefunction.geminals.base_geminal.BaseGeminal>`
  * :class:`APG Wavefunction <wavefunction.geminals.apg.APG>`
  * :class:`APsetG Wavefunction <wavefunction.geminals.apsetg.APsetG>`
  * :class:`APIG Wavefunction <wavefunction.geminals.apig.APIG>`
  * :class:`AP1roG Wavefunction <wavefunction.geminals.ap1rog.AP1roG>`
  * :class:`APr2g Wavefunction <wavefunction.geminals.apr2g.APr2G>`

* Orbital Rotated Wavefunction

  * :class:`Nonorthonormal Orbital Wavefunction <wavefunction.nonorth.nonorth_wavefunction.NonorthWavefunction>`
  * :class:`Jacobi Rotated Orbital Wavefunction <wavefunction.nonorth.jacobi.JacobiWavefunction>`

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

      hamiltonian.chemical_hamiltonian.ChemicalHamiltonian
      hamiltonian.sen0_hamiltonian.SeniorityZeroHamiltonian

      solver.ci_solver.eigen_solve
      solver.system_solver.optimize_wfn_system
      solver.equation_solver.optimize_wfn_variational
      solver.orb_solver.optimize_wfn_orbitals_jacobi
      solver.orb_solver.optimize_ham_orbitals_jacobi

      wavefunction.base_wavefunction.BaseWavefunction
      wavefunction.ci.ci_wavefunction.CIWavefunction
      wavefunction.ci.fci.FCI
      wavefunction.ci.doci.DOCI
      wavefunction.ci.cisd.CISD
      wavefunction.ci.ci_pairs.CIPairs
      wavefunction.geminals.base_geminal.BaseGeminal
      wavefunction.geminals.apg.APG
      wavefunction.geminals.apsetg.APsetG
      wavefunction.geminals.apig.APIG
      wavefunction.geminals.ap1rog.AP1roG
      wavefunction.geminals.apr2g.APr2G
      wavefunction.nonorth.nonorth_wavefunction.NonorthWavefunction
      wavefunction.nonorth.jacobi.JacobiWavefunction

      backend.integrals.BaseIntegrals
      backend.integrals.OneElectronIntegrals
      backend.integrals.TwoElectronIntegrals

      backend.math_tools.binomial
      backend.math_tools.adjugate
      backend.math_tools.permanent_combinatoric
      backend.math_tools.permanent_ryser
      backend.math_tools.permanent_borchardt

      backend.slater.occ
      backend.slater.is_alpha
      backend.slater.spatial_index
      backend.slater.total_occ
      backend.slater.annihilate
      backend.slater.create
      backend.slater.excite
      backend.slater.ground
      backend.slater.is_internal_sd
      backend.slater.internal_sd
      backend.slater.occ_indices
      backend.slater.vir_indices
      backend.slater.shared
      backend.slater.diff
      backend.slater.combine_spin
      backend.slater.split_spin
      backend.slater.interleave_index
      backend.slater.deinterleave_index
      backend.slater.interleave
      backend.slater.deinterleave
      backend.slater.get_spin
      backend.slater.get_seniority
      backend.slater.find_num_trans
      backend.slater.find_num_trans
      backend.sd_list.sd_list

      backend.graphs.generate_complete_pmatch
      backend.graphs.generate_biclique_pmatch
