import wx
from wx.lib.mixins.listctrl import ListCtrlAutoWidthMixin, TextEditMixin

class EditableListCtrl(wx.ListCtrl, TextEditMixin, ListCtrlAutoWidthMixin):
    def __init__(self, parent, ID, ref_box=None,
                 pos=wx.DefaultPosition, size=wx.DefaultSize,
                 style=wx.LC_REPORT|wx.RAISED_BORDER|wx.LC_SINGLE_SEL):
        wx.ListCtrl.__init__(self, parent, ID, pos, size, style)
        ListCtrlAutoWidthMixin.__init__(self)
        # editor
        self.editor = wx.TextCtrl(self, -1, style=wx.TE_LEFT)
        self.editor.SetBackgroundColour(self.editorBgColour)
        self.editor.SetForegroundColour(self.editorFgColour)
        self.editor.SetFont(self.GetFont())
        self.editor.Bind(wx.EVT_CHAR, self.text_editor_input_key)
        self.editor.Hide()

        #Buttons
        self.Bind(wx.EVT_LEFT_DCLICK, self.Edit, self)
        self.Bind(wx.EVT_CHAR, self.Browse, self)

    def Edit(self, event):
        row = self.SelectedRow
        if isinstance(row, int):
            self.Editor(row)

    def Browse(self, event):
        self.editor.Hide()
        keycode = event.GetKeyCode()
        if keycode in [wx.WXK_SPACE, wx.WXK_RETURN, wx.WXK_DELETE, wx.WXK_BACK]:
            row = self.SelectedRow
            self.Edit(wx.EVT_LEFT_DCLICK)
        if keycode == wx.WXK_ESCAPE:
            self.CloseEditor(event)
        event.Skip()

    def text_editor_input_key(self, event):
        keycode = event.GetKeyCode()
        if keycode in [wx.WXK_ESCAPE, wx.WXK_RETURN, wx.WXK_DOWN, wx.WXK_UP,
                       wx.WXK_PAGEDOWN, wx.WXK_PAGEUP]:
            self.CloseEditor(keycode)
        event.Skip()

    def Editor(self, row):

        # Find position of editor
        # from the columns
        col = 6
        x = self.col_loc
        delta_x = self.GetColumnWidth(col)
        y = self.GetItemRect(row)[1]

        # from the scrolls
        scrolloffset = self.GetScrollPos(wx.HORIZONTAL)
        if x+delta_x-scrolloffset > self.GetSize()[0]:
            if wx.Platform == "__WXMSW__":
                # don't start scrolling unless we really need to
                offset = x+delta_x-self.GetSize()[0]-scrolloffset
                # scroll a bit more than what is minimum required
                # so we don't have to scroll everytime the user presses TAB
                # which is very tireing to the eye
                addoffset = self.GetSize()[0]/4
                # but be careful at the end of the list
                if addoffset + scrolloffset < self.GetSize()[0]:
                    offset += addoffset

                self.ScrollList(offset, 0)
                scrolloffset = self.GetScrollPos(wx.HORIZONTAL)
            else:
                # Since we can not programmatically scroll the ListCtrl
                # close the editor so the user can scroll and open the editor
                # again
                self.editor.SetValue(self.GetItem(row, col).GetText())
                self.editor.Hide()
                return

        self.editor.SetDimensions(x-scrolloffset, y, delta_x, 20)
        self.editor.SetValue(self.GetItem(row, col).GetText())
        self.editor.Show()
        self.editor.Raise()
        self.editor.SetSelection(-1,-1)
        self.editor.SetFocus()

    def CloseEditor(self, event):
        if event == wx.WXK_RETURN:
            row = self.SelectedRow
            self.SetStringItem(row, 6, self.editor.GetValue())
        self.editor.Hide()
        # NOTE: There is this bug that if event==wx.WXK_DOWN and self.SetFocus(),
        # the window actually loses focus. Not sure why this happens, but this
        # condition seems to fix it
        if event != wx.WXK_DOWN:
            self.SetFocus()

    @property
    def SelectedRow(self):
        for i in range(self.ItemCount):
            if self.IsSelected(i):
                return i

    def LabelByColumn(self, indices):
        num_items = self.GetItemCount()
        for i in range(num_items):
            new_texts = [self.GetItemText(i, j) for j in indices]
            self.SetStringItem(i, 6, '_'.join(new_texts))

    def get_labels(self):
        """ Returns the new indices for the

        Returns
        -------
        label_indices_sep : list of list
            List of indices that corresponds to the new label of the orbital
            First list belongs to the spatial or alpha orbitals
            Second list (if it exists) belongs to the beta orbitals
        labels : list
            Label of the new orbital (in the same order as the label indices)

        Example
        -------
        If labels=['C0_AO(l=0)', 'C0_AO(l=1)'],
        Then label indices [0, 0, 1, 1] would mean that the first two orbitals
        are labeled with 'C0_A0(l=0)' and the last two orbitals are labeled with
        'C0_AO(l=1)'
        """
        num_items = self.GetItemCount()
        all_labels = [self.GetItemText(i, 6) for i in range(num_items)]
        label_indices_sep = [[], []]
        label_dict = {}
        for i in range(num_items):
            index = int(self.GetItemText(i, 0))
            spin = self.GetItemText(i, 1)
            label = self.GetItemText(i, 6)
            try:
                label_dict[label]
            except KeyError:
                label_dict[label] = len(label_dict)

            if spin in ['spatial', 'alpha']:
                label_indices_sep[0].append(label_dict[label])
            elif spin in ['beta']:
                label_indices_sep[1].append(label_dict[label])
        # Rename unlabeled
        if '' in label_dict:
            label_dict['Unlabeled'] = label_dict.pop('')
            # TODO: reorder Unlabeled to the end (reorder dictionary, relabel labels)
        # Invert dictionary mapping
        indices_dict = {label_index:label for label, label_index in label_dict.items()}
        labels = [indices_dict[i] for i in range(len(indices_dict))]
        return label_indices_sep, labels
