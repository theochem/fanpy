import os
import wx
from orbital_label_maker import EditableListCtrl
import geminals


class ProwlFrame(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, "PROWL GUI")

        # GUI Layout (Top to Bottom)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        panel = wx.Panel(self, wx.ID_ANY)

        self.sizer = wx.BoxSizer(wx.VERTICAL)

        # attributes
        self.mol_path = ''
        self.mol_data = {}

        '''model
        '''
        ###### Buttons ######
        # select mol
        button_load_mol = wx.Button(self,
                                    id=1,
                                    label='Load Molecule',
                                    pos=None,
                                    size=None)
        self.Bind(wx.EVT_BUTTON, self.load_mol, button_load_mol)
        self.sizer.Add(button_load_mol,
                       proportion=0,
                       flag=wx.ALIGN_CENTER)

        # select method
        button_method_select = wx.Button(
            self, id=2, label='Select Method', pos=None, size=None)
        self.Bind(wx.EVT_BUTTON, self.selete_method, button_method_select)
        self.sizer.Add(button_method_select,
                       proportion=0,
                       flag=wx.ALIGN_CENTER)
        # select basis
        button_basis_select = wx.Button(self, id=3, label='Select Basis', pos=None, size=None)
        self.Bind(wx.EVT_BUTTON, self.selete_basis, button_basis_select)
        self.sizer.Add(button_basis_select,
                       proportion=0,
                       flag=wx.ALIGN_CENTER)
        # select initial guess
        button_initial_guess = wx.Button(self, id=4, label='Initialize Guess', pos=None, size=None)
        self.Bind(wx.EVT_BUTTON, self.initial_guess, button_initial_guess)
        self.sizer.Add(button_initial_guess,
                       proportion=0,
                       flag=wx.ALIGN_CENTER)
        # solve
        button_solve = wx.Button(self, id=5, label='Solve It!', pos=None, size=None)
        self.Bind(wx.EVT_BUTTON, self.solve, button_solve)
        self.sizer.Add(button_solve,
                       proportion=0,
                       flag=wx.ALIGN_CENTER)
        # dials and shit
        self._max_orbital_sets = 1
        self.text = wx.TextCtrl(self, size=wx.DefaultSize, value=str(
            self._max_orbital_sets), style=wx.TE_CENTRE | wx.TE_PROCESS_ENTER)
        self.name_text = wx.StaticText(
            self, label="Total orbital sets", style=wx.TE_CENTRE | wx.ST_NO_AUTORESIZE)
        self.spin = wx.SpinButton(self, style=wx.SP_VERTICAL)
        self.spin.SetValue(self._max_orbital_sets)
        self.spin.SetRange(1, 100)

        self.Bind(wx.EVT_SPIN_UP, self.spin_up, self.spin)
        self.Bind(wx.EVT_SPIN_DOWN, self.spin_down, self.spin)
        self.Bind(wx.EVT_TEXT_ENTER, self._text_enter, self.text)

        self.empty = wx.StaticText(
            self, size=wx.DefaultSize, label="", style=wx.TE_CENTRE)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(self.name_text, 3, wx.ALIGN_CENTER)
        hsizer.Add(self.text, 1, wx.ALIGN_CENTER)
        hsizer.Add(self.spin, 0, wx.ALIGN_CENTER)
        hsizer.Add(self.empty, 4, wx.ALIGN_CENTER)
        self.sizer.Add(hsizer, 1, wx.EXPAND | wx.ALIGN_CENTER)
        # self.Bind(wx.EVT_SPIN, self.OnSpin, self.spin)

        # Checkbox MO
        self.check_mo = EditableListCtrl(self, 2)
        self.check_mo.InsertColumn(
            0, 'Index', format=wx.LIST_FORMAT_CENTER, width=wx.LIST_AUTOSIZE_USEHEADER)
        self.check_mo.InsertColumn(
            1, 'Spin', format=wx.LIST_FORMAT_CENTER, width=wx.LIST_AUTOSIZE_USEHEADER)
        self.check_mo.InsertColumn(
            2, 'Occupations (in HF)', format=wx.LIST_FORMAT_CENTER, width=wx.LIST_AUTOSIZE_USEHEADER)
        self.check_mo.InsertColumn(
            3, 'Energy', format=wx.LIST_FORMAT_CENTER, width=wx.LIST_AUTOSIZE_USEHEADER)
        self.check_mo.InsertColumn(4, 'Label', width=wx.LIST_AUTOSIZE)

        self.sizer.Add(self.check_mo,
                       proportion=1,
                       border=4,
                       flag=wx.ALIGN_CENTER | wx.EXPAND)

        # select method

        # select initial guess

        # solve

        # self.check_mo.Hide()

        self.SetSizer(self.sizer)

        # Build
        # self.SetAutoLayout(1)
        # self.sizer.Fit(self)
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
                                       wildcard=("fchk files (*.fchk)|*.fchk|"
                                                 "wfn files (*.wfn)|*.wfn|"
                                                 "wfx files (*.wfx)|*.wfx|"
                                                 "xyz files (*.xyz)|*.xyz"),
                                       style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if openFileDialog.ShowModal() == wx.ID_CANCEL:
            return     # the user changed idea...
        self.mol_path = openFileDialog.GetPath()
        ext = os.path.splitext(self.mol_path)

    def selete_method(self, event):
        pass

    def selete_basis(self, event):
        pass

    def initial_guess(self, event):
        pass

    def solve(self, event):
        pass

# Run the program
if __name__ == "__main__":
    app = wx.App(False)
    frame = ProwlFrame()
    frame.Show()
    app.MainLoop()
