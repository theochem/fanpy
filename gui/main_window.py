import sys
import os
import glob
import wx
import numpy as np

from orbital_label_maker import EditableListCtrl, CalculationSettings, OrbitalSelectionDialog

import geminals
import horton


def raise_error(parent, msg):
    dlg = wx.MessageDialog(parent, msg, "Error", wx.OK)
    dlg.ShowModal() # Show it
    dlg.Destroy() # finally destroy it when finished.

# FIXME: something more generalized
method_dict = {'FCI': geminals.fci.FCI,
               'DOCI': geminals.doci.DOCI,
               'CISD': geminals.cisd.CISD,
               'CI Pairs': geminals.ci_pairs.CIPairs,
               'APIG': geminals.apig.APIG,
               # 'AP1roG': geminals.ap1rog.AP1roG,
               # 'APr2G': geminals.apr2g.APr2G,
               # 'APsetG': geminals.apsetg.APsetG,
               # 'APseqG': geminals.apseqg.APseqG,
}
ci_methods = ['FCI', 'CISD', 'DOCI', 'CI Pairs',]
proj_methods = ['APIG']
class ProwlFrame(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, "PROWL GUI")

        # attributes
        self.mol_path = ''
        self.mol_data = {}
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
        self._max_orbital_sets = 1
        self.text = wx.TextCtrl(self, size=wx.DefaultSize, value=str(
            self._max_orbital_sets), style=wx.TE_CENTRE | wx.TE_PROCESS_ENTER)
        self.spin = wx.SpinButton(self, style=wx.SP_VERTICAL)
        self.spin.SetValue(self._max_orbital_sets)
        self.spin.SetRange(1, 100)

        self.Bind(wx.EVT_SPIN_UP, self.spin_up, self.spin)
        self.Bind(wx.EVT_SPIN_DOWN, self.spin_down, self.spin)
        self.Bind(wx.EVT_TEXT_ENTER, self._text_enter, self.text)

        self.empty = wx.StaticText(self,
                                   size=wx.DefaultSize,
                                   label="",
                                   style=wx.TE_CENTRE)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(self.text, 0, wx.ALIGN_CENTER)
        hsizer.Add(self.spin, 0, wx.ALIGN_CENTER)
        hsizer.Add(self.empty, 0, wx.ALIGN_CENTER)
        self.sizer.Add(hsizer, 0, wx.ALIGN_CENTER)
        # self.Bind(wx.EVT_SPIN, self.OnSpin, self.spin)

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

    def refresh_spin_button_value(self):
        self.text.SetValue(str(self._max_orbital_sets))

    def spin_up(self, event):
        self._max_orbital_sets += 1
        self.refresh_spin_button_value()

    def spin_down(self, event):
        self._max_orbital_sets -= 1
        self.refresh_spin_button_value()

    def _text_enter(self, event):
        value = self.text.GetValue()
        self._max_orbital_sets = int(value)
        self.refresh_spin_button_value()

    def load_mol(self, event):
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
            return     # the user changed idea...
        self.mol_path = openFileDialog.GetPath()

    def initial_guess(self, event):
        pass

    def set_calc_settings(self, event):
        init_dialog = CalculationSettings(self,
                                          title='Set Calculation Settings',
                                          size=(200,225)
        )
        if (init_dialog.ShowModal() == wx.ID_OK):
            nelec = init_dialog.nelec_box.GetLineText(0)
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
        print(ext)
        if ext == '.xyz':
            # FIXME: electron number setter
            self.data = geminals.hort.hartreefock(fn=self.mol_path, basis=basis, nelec=self.nelec, horton_internal=True)
        elif ext == '.fchk':
            self.data = geminals.hort.gaussian_fchk(self.mol_path, horton_internal=True)
        print(self.data['horton_internal']['orb'], geminals.__file__)
        self.load_mol_checkbox(*self.data['horton_internal']['orb'])
        # if CI wavefunction
        if method in ci_methods:
            self.wavefunction = method_dict[method](nelec=self.nelec,
                                                    H=self.data['H'],
                                                    G=self.data['G'],
                                                    nuc_nuc=self.data['nuc_nuc'])
        elif method in proj_methods:
            self.wavefunction = method_dict[method](nelec=self.nelec,
                                                    H=self.data['H'],
                                                    G=self.data['G'],
                                                    nuc_nuc=self.data['nuc_nuc'],
                                                    energy_is_param=self.energy_is_param)

    def load_mol_checkbox(self, exp_alpha, exp_beta=None):
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
        method = self.method_select.GetStringSelection()
        if method in ci_methods:
            init_dialog = OrbitalSelectionDialog(self, 'Select your CAS orbital type', 'cas')
        elif method in proj_methods:
            init_dialog = OrbitalSelectionDialog(self, 'Divide your orbitals into sets', 'set')
        if (init_dialog.ShowModal() == wx.ID_OK):
            pass

    def solve(self, event):
        self.wavefunction()


# Run the program
if __name__ == "__main__":
    app = wx.App(False)
    frame = ProwlFrame()
    frame.Show()
    app.MainLoop()
