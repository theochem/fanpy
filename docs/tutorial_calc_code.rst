.. _tutorial_calc_code:

How to run a calculation by making a script
===========================================
In order to run a calculation (i.e. solve a Schrodinger equation), we need to define the
wavefunction, the Hamiltonian, the objective (form of the Schrodinger equation), and the solver that
will be used to solve the objective. To produce a working script, these objects must be instantiated
and passed to the solver in the correct format.

To help guide this process, you can use the script :ref:`wfns_make_script.py <script_make_script>`
(for a tutorial, see :ref:`Using a script that makes a script <tutorial_calc_make_script>`) to
obtain a template. Once you understand this template, you can customize it by tweaking the
appropriate parts.

Using the following

.. code:: bash

   wfns_make_script.py --nelec 4 --nspin 8 --one_int_file oneint.npy \
                       --two_int_file twoint.npy --wfn_type ap1rog \
                       --optimize_orbs --objective least_squares \
                       --solver cma --filename example.py

we generate the following script

.. code:: python

    import numpy as np
    import os
    from wfns.wfn.geminal.ap1rog import AP1roG
    from wfns.ham.restricted_chemical import RestrictedChemicalHamiltonian
    from wfns.backend.sd_list import sd_list
    from wfns.objective.least_squares import LeastSquaresEquations
    from wfns.solver.equation import cma


    # Number of electrons
    nelec = 4
    print('Number of Electrons: {}'.format(nelec))

    # Number of spin orbitals
    nspin = 8
    print('Number of Spin Orbitals: {}'.format(nspin))

    # One-electron integrals
    one_int_file = 'oneint.npy'
    one_int = np.load(one_int_file)
    print('One-Electron Integrals: {}'.format(os.path.abspath(one_int_file)))

    # Two-electron integrals
    two_int_file = 'twoint.npy'
    two_int = np.load(two_int_file)
    print('Two-Electron Integrals: {}'.format(os.path.abspath(two_int_file)))

    # Nuclear-nuclear repulsion
    nuc_nuc = 0.0
    print('Nuclear-nuclear repulsion: {}'.format(nuc_nuc))

    # Initialize wavefunction
    wfn = AP1roG(nelec, nspin, params=None, memory='2gb', ref_sd=None, ngem=None)
    print('Wavefunction: AP1roG')

    # Initialize Hamiltonian
    ham = RestrictedChemicalHamiltonian(one_int, two_int, energy_nuc_nuc=nuc_nuc, params=None)
    print('Hamiltonian: RestrictedChemicalHamiltonian')

    # Projection space
    pspace = sd_list(nelec, nspin//2, num_limit=None, exc_orders=[1, 2], spin=None,
                     seniority=wfn.seniority)
    print('Projection space (orders of excitations): [1, 2]')

    # Select parameters that will be optimized
    param_selection = [(wfn, np.ones(wfn.nparams, dtype=bool)), (ham, np.ones(ham.nparams, dtype=bool))]

    # Initialize objective
    objective = LeastSquaresEquations(wfn, ham, param_selection=param_selection, tmpfile='',
                                      pspace=pspace, refwfn=None, energy_type='compute', energy=None,
                                      constraints=None, eqn_weights=None)

    # Solve
    print('Optimizing wavefunction: cma solver')
    results = cma(objective, save_file='', sigma0=0.01, options={'ftarget': None, 'timeout': np.inf,
                  'tolfun': 1e-11, 'verb_filenameprefix': 'outcmaes', 'verb_log': 0})

    # Results
    if results['success']:
        print('Optimization was successful')
    else:
        print('Optimization was not successful: {}'.format(results['message']))
    print('Final Energy: {}'.format(results['energy']))

The script can be read as a sequence of steps:

1. Appropriate modules are imported and variables are initialized.
2. Wavefunction is initialized. To customize the wavefunction, the wavefunction variable,
   :code:`wfn`, can be assigned to a different wavefunction or be initialized with different
   parameters. For more information, go to the :code:`__init__` method of the wavefunction in the
   API documentation.
3. Hamiltonian is initialized. To customize the Hamiltonian, the Hamiltonian variable, :code:`ham`,
   can be assigned to a different Hamiltonian or be initialized with different parameters. Note that
   the integrals may be required to have specialized structures (e.g.
   UnrestrictedChemicalHamiltonian). For more information, go to the :code:`__init__` method of the
   Hamiltonian in the API documentation.
4. Select the projection space. The projection space must be provided as a list (or any other
   iterable) of integers whose binary correspond to the occupation vector of the Slater determinant.
   For more information on the representation of the Slater determinant, go to the :py:mod:`slater
   <wfns.backend.slater>` module. The method :py:func:`sd_list <wfns.backend.sd_list.sd_list>` can
   be used instead to produce Slater determinants by the excitations of the ground state Slater
   determinant.
5. Parameters for optimization are selected. For more complex optimization algorithms, we need
   control the selection of parameters that will be optimized during the algorithm. In this case,
   the parameters of the wavefunction and the parameters of the Hamiltonian (responsible for orbital
   rotation) are both active in the optimization. To freeze specific parameters, change the
   corresponding element in the boolean array to :code:`False`. For example, to freeze the first and
   fifth parameters of the wavefunction, we get something like this:

.. code:: python

   wfn_selection = np.ones(wfn.nparams, dtype=bool)
   wfn_selection[[0, 4]] = False
   param_selection = [(wfn, wfn_selection), (ham, np.ones(ham.nparams, dtype=bool))]

6. Schrodinger equation (objective) is initialized. To customize the objective, the objective
   variable, :code:`objective`, can be assigned to a different Schrodinger equation instance or be
   initialized with different parameters. For more information, go to the :code:`__init__` method of
   the Schrodinger equation instance in the :py:mod:`objective
   <wfns.objective>` module. Different constraints (the default is the normalization
   constraint) can be found in the :py:mod:`objetive.constraints <wfns.objective.constraints>`
   module.
7. Solver is called to optimize the wavefunction. The solver can be changed to a different solver
   (provided that it is compatible with the given wavefunction, Hamiltonian, and objective) or be
   used with a different set of keyword arguments. Any of the parameters can be tweaked *before* the
   start of the optimization. For example, the wavefunction parameters can be imported from another
   numpy array by using the keyword :code:`--load_wfn LOAD_WFN` in :ref:`wfns_make_script.py
   <script_make_script>` or by by adding the following lines before the solver:

.. code:: python

   wfn_params = np.load('wfn_param_file.npy')
   wfn.assign_params(wfn_params)

8. Output is printed. If you would like to save the parameters after the optimization, they can be
   saved here. For example, to save the Hamiltonian parameters can be saved by using the keyword
   :code:`--save_ham SAVE_HAM` in :ref:`wfns_make_script.py <script_make_script>` or by adding the
   following line:

.. code:: python

   np.save('ham_params.npy', ham.params)
