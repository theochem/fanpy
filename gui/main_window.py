import wx


class MyForm(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, "Geminal")
        panel = wx.Panel(self, wx.ID_ANY)

        '''model
        '''
        self._max_orbital_sets = 1
        self.text = wx.TextCtrl(panel, size=wx.DefaultSize, value=str(
            self._max_orbital_sets), style=wx.TE_RIGHT|wx.TE_PROCESS_ENTER)
        self.name_text = wx.StaticText(
            panel, label="Total orbital sets", style=wx.TE_CENTRE | wx.ST_NO_AUTORESIZE)
        self.spin = wx.SpinButton(panel, style=wx.SP_VERTICAL)
        self.spin.SetValue(self._max_orbital_sets)
        self.spin.SetRange(1, 100)

        self.Bind(wx.EVT_SPIN_UP, self.spin_up, self.spin)
        self.Bind(wx.EVT_SPIN_DOWN, self.spin_down, self.spin)
        self.Bind(wx.EVT_TEXT_ENTER, self._text_enter, self.text)

        self.empty = wx.StaticText(
            panel, size=wx.DefaultSize, label="", style=wx.TE_CENTRE)
        # self.Bind(wx.EVT_SPIN, self.OnSpin, self.spin)
        vSizer = wx.BoxSizer(wx.VERTICAL)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer2 = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(self.name_text, 3, wx.CENTER)
        hsizer.Add(self.text, 1, wx.CENTER)
        hsizer.Add(self.spin, 0, wx.CENTER)
        hsizer.Add(self.empty, 4, wx.CENTER)

        vSizer.Add(hsizer, 1, wx.EXPAND)
        vSizer.Add(hsizer2, 1, wx.EXPAND)
        panel.SetSizer(vSizer)

    def refresh_spin_button_value(self):
        self.text.SetValue(str(self._max_orbital_sets))

    def spin_up(self, event):
        self._max_orbital_sets += 1
        print(self._max_orbital_sets)
        self.refresh_spin_button_value()

    def spin_down(self, event):
        print(event)
        self._max_orbital_sets -= 1
        print(self._max_orbital_sets)
        # print self._max_orbital_sets
        self.refresh_spin_button_value()

    def _text_enter(self, event):
        value = self.text.GetValue()
        self._max_orbital_sets = int(value)
        self.refresh_spin_button_value()
    # def OnSpin(self, event):
        # self.text.SetValue(str(event.GetPosition()))

# Run the program
if __name__ == "__main__":
    app = wx.App(False)
    frame = MyForm()
    frame.Show()
    app.MainLoop()
