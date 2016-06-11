import wx

class CalculationSettings(wx.Dialog):
    """ Dialog for setting up the calculation settings

    Attributes
    ----------
    nelec_box : wx.SpinCtrl
        Box that sets the number of electrons
    energy_is_param_box : wx.ListBox
        Box that selects whether energy is a parameter or not

    """
    def __init__(self, parent, title, pos=wx.DefaultPosition,
                 size=wx.DefaultSize, style=wx.DEFAULT_DIALOG_STYLE):
        """ Initializes

        Parameters
        ----------
        parent : wx.Frame
            Frame that contains this frame
        title : str
            Title of the dialog box
        pos : wx.Point
            Position of the dialog upon creation
        size : wx.Size
            Size of the the dialog upon creation
        style : long
            Flags for the style of the dialog box
            wx.DEFAULT_DIALOG_STYLE
                wx.CAPTION, wx.CLOSE_BOX, wx.SYSTEM_MENU
            wx.CAPTION
                Puts caption on dialog box
            wx.CLOSE_BOX
                Puts button for closing dialog box
            WX.SYSTEM_MENU
                Puts system menu
        """
        wx.Dialog.__init__(self, parent, -1, title, pos, size, style)

        # centers dialog in the middle of parent
        x, y = pos
        if x == -1 and y == -1:
            self.CenterOnScreen(wx.BOTH)

        dlgsizer = wx.BoxSizer(wx.VERTICAL)

        # number of electrons
        instruct_nelec = wx.StaticText(self, -1, 'Number of Electrons')
        dlgsizer.Add(instruct_nelec, proportion=0, flag=wx.ALIGN_CENTER, border=4)

        self.nelec_box = wx.SpinCtrl(self, value='', initial=1, min=1, max=100, style=wx.SP_VERTICAL)
        dlgsizer.Add(self.nelec_box, proportion=0, flag=wx.ALIGN_CENTER, border=4)

        # set energy is parm
        instruct_energy_param = wx.StaticText(self, -1, 'Is Energy a Parameter?')
        dlgsizer.Add(instruct_energy_param, proportion=0, flag=wx.ALIGN_CENTER, border=4)

        self.energy_is_param_box = wx.ListBox(parent=self,
                                         id=-1,
                                         pos=None,
                                         size=None,
                                         choices=['Yes', 'No'],
                                         style=wx.LB_SINGLE)
        self.energy_is_param_box.SetSelection(0)
        # FIXME: box is a little too tall
        dlgsizer.Add(self.energy_is_param_box, proportion=0, flag=wx.ALIGN_CENTER, border=4)

        # buttons
        btnsizer = wx.StdDialogButtonSizer()
        ok = wx.Button(self, wx.ID_OK, "OK")
        ok.SetDefault()
        btnsizer.AddButton(ok)
        cancel = wx.Button(self, wx.ID_CANCEL, "Cancel")
        btnsizer.AddButton(cancel)
        btnsizer.Realize()
        dlgsizer.Add(btnsizer, proportion=0, flag=wx.ALIGN_CENTER, border=4)

        # resize
        self.SetSizer(dlgsizer)
        self.Layout()

