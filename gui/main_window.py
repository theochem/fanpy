import sys
import os
import glob
import wx
import numpy as np

from orbital_selection import EditableListCtrl, OrbitalSelectionDialog
from calc_setting import CalculationSettings

import geminals
import horton


def raise_error(parent, msg):
    """ Opens a MessageDialog with a message for the purpose of "raising" an error

    Parameters
    ----------
    parent : wx.Frame
        Parent of frame
    msg : str
        Message to be written
    """
    dlg = wx.MessageDialog(parent, msg, "Error", wx.OK)
    dlg.ShowModal() # Show it
    dlg.Destroy() # finally destroy it when finished.

# FIXME: something more generalized
method_dict = {'FCI': geminals.ci.fci.FCI,
               'DOCI': geminals.ci.doci.DOCI,
               'CISD': geminals.ci.cisd.CISD,
               'CI Pairs': geminals.ci.ci_pairs.CIPairs,
               'APIG': geminals.proj.apig.APIG,
               # 'AP1roG': geminals.proj.ap1rog.AP1roG,
               # 'APr2G': geminals.proj.apr2g.APr2G,
               # 'APsetG': geminals.proj.apsetg.APsetG,
               # 'APseqG': geminals.proj.apseqg.APseqG,
}
ci_methods = ['FCI', 'CISD', 'DOCI', 'CI Pairs',]
proj_methods = ['APIG']
class ProwlFrame(wx.Frame):
    """

    Attributes
    ----------
    mol_path : str
        Absolute path of the molecule data file
    nelec : int
        Number of electrons
    energy_is_param : bool
        Flag for whether to make the energy a parameter
    wavefunction : Wavefunction Instance
        Instance of the appropriate wavefunction class
    data : dict
        Dictionary of the data from the geminals.hort
    sizer : wx.BoxSizer
        Sizer used to orient and reshape each frame used
    method_select : wx.ComboBox
        Frame (ComboBox) for selecting the wavefunction to be used for the calculation
    basis_select : wx.ComboBox
        Frame (ComboBox) for selecting the basis set used
    nset_select : wx.SpinCtrl
        Frame (SpinCtrl) for selecting the number of orbital sets
    check_mo : wx.ListCtrl
        Frame (ListCtrl) for displaying the orbital information (index, spin, occupation, energy)

    Methods
    -------
    load_mol_checkbox
        Loads check_mo from the given orbital information

    Event Handlers
    --------------
    load_mol
        Loads the molecular file path
    initial_guess
        Loads the initial guess
        Not Implemented
    set_calc_settings
        Sets the calculations settings
    initialize
        Initializes the wavefunction
    select_orbitals
        Selects the orbitals
    solve
        Solves the wavefunction

    """

    def __init__(self):
        # initializes frame
        wx.Frame.__init__(self, None, wx.ID_ANY, "PROWL GUI")

        # attributes
        self.mol_path = ''
        self.nelec = 0.0
        self.energy_is_param = False
        self.wavefunction = None
        self.data = {}

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        # select mol
        button_load_mol = wx.Button(self,
                                    id=-1,
                                    label='Load Molecule',
                                    pos=None,
                                    size=None)
        self.Bind(wx.EVT_BUTTON, self.load_mol, button_load_mol)
        self.sizer.Add(button_load_mol,
                       proportion=0,
                       flag=wx.ALIGN_CENTER)

        hsizer1 = wx.BoxSizer(wx.HORIZONTAL)
        # select method
        method_list = ci_methods + proj_methods
        self.method_select = wx.ComboBox(self,
                                         id=-1,
                                         value="Method",
                                         choices=method_list)
        hsizer1.Add(self.method_select,
                    proportion=0,
                    flag=wx.ALIGN_CENTER)
        # select basis
        basis_files = glob.glob(os.path.join(os.environ['HORTONDATA'], 'basis', '*nwchem'))
        basis_list = [os.path.splitext(os.path.basename(i))[0] for i in basis_files]
        basis_list = sorted(basis_list)
        self.basis_select = wx.ComboBox(self,
                                        id=-1,
                                        value="Basis Set",
                                        choices=basis_list)
        hsizer1.Add(self.basis_select,
                    proportion=0,
                    flag=wx.ALIGN_CENTER)

        self.sizer.Add(hsizer1, proportion=0, flag=wx.ALIGN_CENTER)

        # set calculation settings
        button_set_calc_settings = wx.Button(self,
                                    id=-1,
                                    label='Set Calculation Settings',
                                    pos=None,
                                    size=None)
        self.Bind(wx.EVT_BUTTON, self.set_calc_settings, button_set_calc_settings)
        self.sizer.Add(button_set_calc_settings,
                       proportion=0,
                       flag=wx.ALIGN_CENTER)

        # select initial guess
        # button_initial_guess = wx.Button(
        #     self, id=4, label='Initialize Guess', pos=None, size=None)
        # self.Bind(wx.EVT_BUTTON, self.initial_guess, button_initial_guess)
        # self.sizer.Add(button_initial_guess,
        #                proportion=0,
        #                flag=wx.ALIGN_CENTER)

        # initialize
        button_initialize = wx.Button(self,
                                 id=-1,
                                 label='Initialize It!',
                                 pos=None,
                                 size=None)
        self.Bind(wx.EVT_BUTTON, self.initialize, button_initialize)
        self.sizer.Add(button_initialize,
                       proportion=0,
                       flag=wx.ALIGN_CENTER)

        self.name_text = wx.StaticText(self,
                                       label="Total orbital sets",
                                       style=wx.TE_CENTRE | wx.ST_NO_AUTORESIZE)
        self.sizer.Add(self.name_text, 0, wx.ALIGN_CENTER)

        # dials and shit
        self.nset_select = wx.SpinCtrl(self, value='1', initial=1, min=1, max=100, style=wx.SP_VERTICAL)

        self.sizer.Add(self.nset_select, 0, wx.ALIGN_CENTER)

        # Checkbox MO
        self.check_mo = wx.ListCtrl(self, -1, style=wx.LC_REPORT|wx.RAISED_BORDER|wx.LC_SINGLE_SEL)
        self.check_mo.InsertColumn(
            0, 'Index', format=wx.LIST_FORMAT_CENTER, width=wx.LIST_AUTOSIZE_USEHEADER)
        self.check_mo.InsertColumn(
            1, 'Spin', format=wx.LIST_FORMAT_CENTER, width=wx.LIST_AUTOSIZE_USEHEADER)
        self.check_mo.InsertColumn(
            2, 'Occupations (in HF)', format=wx.LIST_FORMAT_CENTER, width=wx.LIST_AUTOSIZE_USEHEADER)
        self.check_mo.InsertColumn(
            3, 'Energy', format=wx.LIST_FORMAT_CENTER, width=wx.LIST_AUTOSIZE_USEHEADER)
        self.sizer.Add(self.check_mo,
                       proportion=1,
                       border=4,
                       flag=wx.ALIGN_CENTER | wx.EXPAND)

        # Select orbitals
        button_select_orbitals = wx.Button(self,
                                 id=-1,
                                 label='Select_Orbitals',
                                 pos=None,
                                 size=None)
        self.Bind(wx.EVT_BUTTON, self.select_orbitals, button_select_orbitals)
        self.sizer.Add(button_select_orbitals,
                       proportion=0,
                       flag=wx.ALIGN_CENTER)

        # solve
        button_solve = wx.Button(self,
                                 id=-1,
                                 label='Solve It!',
                                 pos=None,
                                 size=None)
        self.Bind(wx.EVT_BUTTON, self.solve, button_solve)
        self.sizer.Add(button_solve,
                       proportion=0,
                       flag=wx.ALIGN_CENTER)

        # Build
        self.SetSizer(self.sizer)
        self.SetAutoLayout(1)
        self.sizer.Fit(self)
        self.Show()

    def load_mol(self, event):
        """ Loads molecule file path upon occurrence of event

        Parameters
        ----------
        event : wx.Event
            Event that results in loading the molecule
        """
        openFileDialog = wx.FileDialog(self,
                                       message="Open mol file",
                                       defaultDir="",
                                       defaultFile="",
                                       wildcard=("xyz files (*.xyz)|*.xyz|"
                                                 "fchk files (*.fchk)|*.fchk|"
                                                 "wfn files (*.wfn)|*.wfn|"
                                                 "wfx files (*.wfx)|*.wfx"
                                                 ),
                                       style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if openFileDialog.ShowModal() == wx.ID_CANCEL:
            return
        self.mol_path = openFileDialog.GetPath()

    def initial_guess(self, event):
        """ Loads an initial guess upon occurrence of event

        Parameters
        ----------
        event : wx.Event
            Event that results in loading the molecule
        """
        raise NotImplementedError

    def set_calc_settings(self, event):
        """ Opens dialog for calculation settings upon occurrence of event

        Parameters
        ----------
        event : wx.Event
            Event that results in opening the calculation setting dialog
        """
        init_dialog = CalculationSettings(self,
                                          title='Set Calculation Settings',
                                          size=(200,225)
        )
        if (init_dialog.ShowModal() == wx.ID_OK):
            nelec = init_dialog.nelec_box.GetValue()
            try:
                # FIXME: ugly
                if float(nelec) != int(nelec):
                    raise_error(self, 'Number of Electron must be an integer')
                self.nelec = int(nelec)
            except ValueError:
                raise_error(self, 'Number of Electron must be an integer')
            energy_is_param = init_dialog.energy_is_param_box.GetSelection()
            if energy_is_param == 'Yes':
                self.energy_is_param = True
            elif energy_is_param == 'No':
                self.energy_is_param = False

    def initialize(self, event):
        """ Initialized wavefunction upon occurrence of event

        Parameters
        ----------
        event : wx.Event
            Event that results in wavefunction initialization
        """
        method = self.method_select.GetStringSelection()
        basis = self.basis_select.GetStringSelection()
        if self.mol_path == '' or method == '' or basis == '':
            raise_error(self, 'Need to provide all of the mol_path, method, and basis')
            return
        if not isinstance(self.nelec, int) or not isinstance(self.energy_is_param, bool):
            raise_error(self, 'Need configure the calculation settings')
            return
        # Run HF
        ext = os.path.splitext(self.mol_path)[1]
        if ext == '.xyz':
            # FIXME: electron number setter
            self.data = geminals.hort.hartreefock(fn=self.mol_path, basis=basis, nelec=self.nelec, horton_internal=True)
        elif ext == '.fchk':
            self.data = geminals.hort.gaussian_fchk(self.mol_path, horton_internal=True)
        self.load_mol_checkbox(*self.data['horton_internal']['orb'])
        # if CI wavefunction
        if method in ci_methods:
            self.wavefunction = method_dict[method](nelec=self.nelec,
                                                    H=self.data['H'],
                                                    G=self.data['G'],
                                                    nuc_nuc=self.data['nuc_nuc'])
        # if Projection Wavefunction
        elif method in proj_methods:
            self.wavefunction = method_dict[method](nelec=self.nelec,
                                                    H=self.data['H'],
                                                    G=self.data['G'],
                                                    nuc_nuc=self.data['nuc_nuc'],
                                                    energy_is_param=self.energy_is_param)
        else:
            raise_error(self, 'Unsuppported wavefunction, {0}'.format(method))

    def load_mol_checkbox(self, exp_alpha, exp_beta=None):
        """ Loads the self.check_mo given the orbital informations

        Parameters
        ----------
        exp_alpha : horton.matrix.dense.DenseExpansion
            Instance that contains the orbital information in HORTON
            If exp_beta is None, then exp_alpha corresponds to the spatial orbitals
            If exp_beta is not None, then exp_alpha corresponds to the alpha orbitals
        exp_beta : horton.matrix.dense.DenseExpansion
            Instance that contains the orbital information in HORTON
            Corresponds to beta orbitals
        """
        # load info
        if exp_beta is None:
            indices = range(exp_alpha.nfn)
            spins = ['spatial']*exp_alpha.nfn
            occs = exp_alpha.occupations*2
            energies = exp_alpha.energies
        elif isinstance(exp_beta, horton.matrix.dense.DenseExpansion):
            spins = ['alpha']*exp_alpha.nfn + ['beta']*exp_beta.nfn
            indices = range(exp_alpha.nfn+exp_beta.nfn)
            occs = np.hstack((exp_alpha.occupations, exp_beta.occupations))
            energies = np.hstack((exp_alpha.energies, exp_beta.energies))
            # sort
            zipped_info = zip(spins, indices, occs, energies)
            sorted_info = sorted(zipped_info, key=lambda x:x[3])
            spins, indices, occs, energies = zip(*sorted_info)
        else:
            return
        # reset check_mo
        self.check_mo.DeleteAllItems()
        occs[np.abs(occs)<1e-7] = 0
        for (index, spin, occ, energy) in zip(indices, spins, occs, energies):
            ind = self.check_mo.InsertStringItem(sys.maxint, str(index))
            self.check_mo.SetStringItem(ind, 1, spin)
            self.check_mo.SetStringItem(ind, 2, str(occ))
            self.check_mo.SetStringItem(ind, 3, str(energy))
        self.check_mo.SetColumnWidth(1, wx.LIST_AUTOSIZE)
        self.check_mo.SetColumnWidth(3, wx.LIST_AUTOSIZE)
        self.sizer.Fit(self)


    def select_orbitals(self, event):
        """ Opens dialog for selecting orbitals upon occurrence of event

        Parameters
        ----------
        event : wx.Event
            Event that results in selecting the orbitals
        """
        # check if we can select yet
        if not isinstance(self.wavefunction, geminals.wavefunction.Wavefunction):
            raise_error(self, 'Wavefunction needs to be initialized.')
            return
        # if orbitals dont need selecting:
        #     raise_error(self, 'Orbital selection not needed.')
        #     return
        # make dialog
        method = self.method_select.GetStringSelection()
        if method in ci_methods:
            init_dialog = OrbitalSelectionDialog(self, 'Select your CAS orbital type', 'cas')
        elif method in proj_methods:
            init_dialog = OrbitalSelectionDialog(self, 'Divide your orbitals into sets', 'connections')
        # if cancel
        if (init_dialog.ShowModal() == wx.ID_OK):
            return
        # FIXME: need behaviour after selection

    def solve(self, event):
        """ Solves the wavefunction upon occurence of event

        Parameters
        ----------
        event : wx.Event
            Event that results in solving the wavefunction
        """
        # check if we can select yet
        if isinstance(self.wavefunction, geminals.wavefunction.Wavefunction):
            raise_error(self, 'Wavefunction needs to be initialized.')
            return
        # if orbitals are selected
        #     raise_error(self, 'Orbitals need to be selected.')
        #     return
        # solve
        self.wavefunction()


# Run the program
if __name__ == "__main__":
    app = wx.App(False)
    frame = ProwlFrame()
    frame.Show()
    app.MainLoop()
