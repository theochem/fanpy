import sys
import wx
import numpy as np

from wx.lib.mixins.listctrl import ListCtrlAutoWidthMixin, TextEditMixin
import matplotlib
from matplotlib_panel import CanvasPanel
sys.path.append('/home/dkim/Work/mo_diagram')
import graph, mo_diagram

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
        col = 4
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
            self.SetStringItem(row, 4, self.editor.GetValue())
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

class OrbitalSelectionDialog(wx.Dialog):
    def __init__(self, parent, title, sel_type='',
                 pos=wx.DefaultPosition,
                 size = wx.DefaultSize, style = wx.DEFAULT_DIALOG_STYLE):
        wx.Dialog.__init__(self, parent, -1, title, pos, size, style)

        x, y = pos
        if x == -1 and y == -1:
            self.CenterOnScreen(wx.BOTH)

        self.dlgsizer = wx.BoxSizer(wx.VERTICAL)
        self.boxsizer = wx.BoxSizer(wx.HORIZONTAL)

        box_one_sizer = wx.BoxSizer(wx.VERTICAL)

        self.check_mo = EditableListCtrl(self, -1, style=wx.LC_REPORT|wx.RAISED_BORDER|wx.LC_SINGLE_SEL)
        # add the columns
        self.check_mo.InsertColumn(0, 'Index', format=wx.LIST_FORMAT_CENTER, width=wx.LIST_AUTOSIZE_USEHEADER)
        self.check_mo.InsertColumn(1, 'Spin', format=wx.LIST_FORMAT_CENTER, width=wx.LIST_AUTOSIZE_USEHEADER)
        self.check_mo.InsertColumn(2, 'Occupations (in HF)', format=wx.LIST_FORMAT_CENTER, width=wx.LIST_AUTOSIZE_USEHEADER)
        self.check_mo.InsertColumn(3, 'Energy', format=wx.LIST_FORMAT_CENTER, width=wx.LIST_AUTOSIZE_USEHEADER)
        if sel_type == 'cas':
            self.check_mo.InsertColumn(4, 'Type', format=wx.LIST_FORMAT_CENTER, width=wx.LIST_AUTOSIZE_USEHEADER)
        elif sel_type == 'set':
            self.check_mo.InsertColumn(4, 'Set', format=wx.LIST_FORMAT_CENTER, width=wx.LIST_AUTOSIZE_USEHEADER)
        elif sel_type == 'connections':
            self.check_mo.InsertColumn(4, 'Connected Orbitals', format=wx.LIST_FORMAT_CENTER, width=wx.LIST_AUTOSIZE_USEHEADER)

        for i in range(parent.check_mo.GetItemCount()):
            if parent.check_mo.GetItemText(i, col=1) == 'spatial':
                # alphas
                ind = self.check_mo.InsertStringItem(sys.maxint, str(i))
                self.check_mo.SetStringItem(ind, 1, 'alpha')
                self.check_mo.SetStringItem(ind, 2, str(float(parent.check_mo.GetItemText(i, col=2))/2))
                self.check_mo.SetStringItem(ind, 3, parent.check_mo.GetItemText(i, col=3))
                # betas
                ind = self.check_mo.InsertStringItem(sys.maxint, str(i+parent.check_mo.GetItemCount()))
                self.check_mo.SetStringItem(ind, 1, 'beta')
                self.check_mo.SetStringItem(ind, 2, str(float(parent.check_mo.GetItemText(i, col=2))/2))
                self.check_mo.SetStringItem(ind, 3, parent.check_mo.GetItemText(i, col=3))
            else:
                ind = self.check_mo.InsertStringItem(sys.maxint, str(i))
                self.check_mo.SetStringItem(ind, 1, parent.check_mo.GetItemText(i, col=1))
                self.check_mo.SetStringItem(ind, 2, parent.check_mo.GetItemText(i, col=2))
                self.check_mo.SetStringItem(ind, 3, parent.check_mo.GetItemText(i, col=3))


        if sel_type == 'cas':
            self.default_cas()
            # over write old close edit event handling
            self.check_mo.CloseEditor = lambda event: self.cas_close_editor(self.check_mo, event)
        elif sel_type == 'set':
            self.default_set()
        elif sel_type == 'connections':
            self.default_connections()
            # over write old close edit event handling
            self.check_mo.CloseEditor = lambda event: self.cas_close_editor(self.check_mo, event)
            self.check_mo.CloseEditor = lambda event: self.connections_close_editor(self.check_mo, event)

        self.check_mo.SetColumnWidth(1, wx.LIST_AUTOSIZE)
        self.check_mo.SetColumnWidth(3, wx.LIST_AUTOSIZE)

        box_one_sizer.Add(self.check_mo,
                       proportion=1,
                       border=4,
                       flag=wx.EXPAND)

        box_two_sizer = wx.BoxSizer(wx.VERTICAL)
        self.graph_panel = CanvasPanel(self)
        if sel_type == 'cas':
            self.graph_cas()
            self.graph_panel.canvas.mpl_connect('pick_event', self.select_cas)
        elif sel_type == 'set':
            self.graph_set()
            self.graph_panel.canvas.mpl_connect('pick_event', self.select_set)
        elif sel_type == 'connections':
            self.graph_connections()
            self.graph_panel.canvas.mpl_connect('pick_event', self.select_connections)

        box_two_sizer.Add(self.graph_panel)

        # Add all the box sizers
        self.boxsizer.Add(box_one_sizer,
                          proportion=1,
                          flag=wx.ALIGN_CENTER|wx.EXPAND)
        self.boxsizer.Add(box_two_sizer,
                          proportion=1,
                          flag=wx.ALIGN_CENTER|wx.EXPAND)

        self.dlgsizer.Add(self.boxsizer,
                          proportion=1,
                          flag=wx.ALIGN_CENTER|wx.EXPAND)

        btnsizer = wx.StdDialogButtonSizer()
        ok = wx.Button(self, wx.ID_OK, "OK")
        btnsizer.AddButton(ok)
        cancel = wx.Button(self, wx.ID_CANCEL, "Cancel")
        btnsizer.AddButton(cancel)
        btnsizer.Realize()
        self.dlgsizer.Add(btnsizer, proportion=0, flag=wx.ALIGN_CENTER, border=4)

        self.SetSizer(self.dlgsizer)
        self.dlgsizer.Fit(self)
        self.Layout()

        # Set editor box location (needs to be done after resizing windows)
        self.check_mo.col_loc = sum(self.check_mo.GetColumnWidth(i) for i in range(4))



    def default_cas(self):
        for i in range(self.check_mo.GetItemCount()):
            if float(self.check_mo.GetItemText(i, col=2)) > 0 :
                self.check_mo.SetStringItem(i, 4, 'frozen')
            else:
                self.check_mo.SetStringItem(i, 4, 'virtual')

    def default_set(self):
        for i in range(self.check_mo.GetItemCount()):
            if self.check_mo.GetItemText(i, col=1) == 'alpha':
                self.check_mo.SetStringItem(i, 4, '1')
            elif self.check_mo.GetItemText(i, col=1) == 'beta':
                self.check_mo.SetStringItem(i, 4, '2')
        # construct adjacency matrix
        adjacency = np.ones((num_points, num_points), dtype=bool)
        # orbitals within the same set are not connected
        set_label = np.array([self.check_mo.GetItemText(i, col=4) for i in range(num_points)])
        # select indices that are zero
        zero_indices = set_label == set_label.reshape(num_points,1)
        adjacency[zero_indices] = False
        self.adjacency = adjacency

    def default_connections(self):
        num_points = self.check_mo.GetItemCount()
        # construct adjacency matrix
        adjacency = np.ones((num_points, num_points))
        adjacency -= np.identity(num_points)
        adjacency = adjacency.astype(bool)
        self.adjacency = adjacency
        # set the column
        for i, row in enumerate(adjacency):
            connected_indices = np.where(row)[0]
            # fill column
            self.check_mo.SetStringItem(i, 4, ','.join(connected_indices.astype(str)))

    def graph_set(self):
        num_points = self.check_mo.GetItemCount()
        # FIXME: label the points
        # FIXME: reorder the points so that the graph looks nicer
        return self.graph_panel.draw(graph.generate_circle_graph, num_points=num_points, adjacency=self.adjacency, on_alpha=1.0, off_alpha=0.1)

    def graph_cas(self):
        num_points = self.check_mo.GetItemCount()
        energies = np.array([float(self.check_mo.GetItemText(i, col=3)) for i in range(num_points)])
        occupations = np.array([float(self.check_mo.GetItemText(i, col=2)) for i in range(num_points)])
        self.graph_panel.draw(mo_diagram.generate_all_mo_diagrams, list_energies=[energies], list_occupations=[occupations], pick_event_energy=False)
        # color the lines
        col_orbtype = {'frozen':'red', 'active':'blue', 'virtual':'green'}
        for i in range(num_points):
            color = col_orbtype[self.check_mo.GetItemText(i, col=4)]
            self.graph_panel.axes.lines[i].set_color(color)

        self.graph_panel.axes.legend([matplotlib.lines.Line2D([], [], color='red'),
                                      matplotlib.lines.Line2D([], [], color='blue'),
                                      matplotlib.lines.Line2D([], [], color='green')],
                                     ['Frozen',
                                      'Active',
                                      'Virtual'])

    def graph_connections(self):
        num_points = self.check_mo.GetItemCount()
        # FIXME: label the points
        # FIXME: reorder the points so that the graph looks nicer
        return self.graph_panel.draw(graph.generate_circle_graph, num_points=num_points, adjacency=self.adjacency, on_alpha=1.0, off_alpha=0.1)

    def select_set(self, event):
        # not sure if this is useful
        pass

    def select_cas(self, event):
        # cycles in the opposite direction of what is written
        color_cycle = ['green', 'blue', 'red']
        orbtype_col = {'red':'frozen', 'blue':'active', 'green':'virtual'}
        thisline = event.artist
        index = self.graph_panel.axes.lines.index(thisline)
        color = thisline.get_color()
        thisline.set_color(color_cycle[color_cycle.index(color)-1])
        self.graph_panel.figure.canvas.draw()
        # set the text corresponding to the color
        self.check_mo.SetStringItem(index, 4, orbtype_col[thisline.get_color()])

    def select_connections(self, event):
        thisline = event.artist
        points = thisline.get_xydata()
        # select the points used to construct the line
        point_indices = []
        for point in points:
            # find which points are used to make the line
            points_ind = np.all(self.graph_panel.plot._offsets == point, axis=1)
            # turn bool to ind
            points_ind = np.where(points_ind)[0]
            # there can only be one point
            assert points_ind.size == 1, 'There can only be one point'
            point_indices.append(points_ind[0])
        # remove the points from the adjacency matrix
        self.adjacency[point_indices[0], point_indices[1]] = -self.adjacency[point_indices[0], point_indices[1]]
        self.adjacency[point_indices[1], point_indices[0]] = -self.adjacency[point_indices[1], point_indices[0]]
        # set the text in column
        for i in point_indices:
            connected_indices = np.where(self.adjacency[i])[0]
            self.check_mo.SetStringItem(i, 4, ','.join(connected_indices.astype(str)))

    def cas_close_editor(self, self_elc, event):
        EditableListCtrl.CloseEditor(self_elc, event)
        col_orbtype = {'frozen':'red', 'active':'blue', 'virtual':'green'}
        row_ind = self_elc.SelectedRow
        try:
            color = col_orbtype[self_elc.GetItemText(row_ind, 4)]
            self.graph_panel.axes.lines[row_ind].set_color(color)
            self.graph_panel.figure.canvas.draw()
        except KeyError:
            msg = "The orbital type must be one of 'frozen', 'active', and 'virtual'."
            dlg = wx.MessageDialog(self, msg, "Error", wx.OK)
            dlg.ShowModal() # Show it
            dlg.Destroy() # finally destroy it when finished.

    def connections_close_editor(self, self_elc, event):
        EditableListCtrl.CloseEditor(self_elc, event)
        row_ind = self_elc.SelectedRow
        connections = self_elc.GetItemText(row_ind, 4).split(',')
        assert str(row_ind) not in connections
        if connections[0] == '':
            connections = []
        try:
            connections = [int(i) for i in connections]
            num_points = self.check_mo.GetItemCount()
            # make boolean array
            lines_are_on = np.zeros(num_points, dtype=bool)
            lines_are_on[connections] = True
            # find where the difference is
            indices_to_switch = self.adjacency[row_ind] != lines_are_on
            # update adjacency graph
            self.adjacency[row_ind, indices_to_switch] = -self.adjacency[row_ind, indices_to_switch]
            self.adjacency[indices_to_switch, row_ind] = -self.adjacency[indices_to_switch, row_ind]
            # find the index of the edges that need to be switched (in the total list of edges)
            #  indices of the edges that correspond to the row/column row_ind wrt total adjacency matrix
            row_indices = np.zeros((num_points, num_points), dtype=bool)
            row_indices[row_ind, indices_to_switch] = True
            row_indices[indices_to_switch, row_ind] = True
            row_indices = zip(*np.where(np.triu(row_indices, k=1)))
            #  indices of all of the edges wrt total adjacency matrix
            all_indices = np.ones((num_points, num_points), dtype=bool)
            all_indices = zip(*np.where(np.triu(all_indices, k=1)))
            #  indices of the line wrt the list of all lines
            line_indices = [all_indices.index(i) for i in row_indices]
            # switch the appropriate liens
            for line_index in line_indices:
                line = self.graph_panel.axes.lines[line_index]
                line.set_alpha(1.0+0.1-line.get_alpha())
            self.graph_panel.figure.canvas.draw()
            # update the text in column
            for i, row_is_changed in enumerate(indices_to_switch):
                if row_is_changed:
                    connected_indices = np.where(self.adjacency[i])[0]
                    self.check_mo.SetStringItem(i, 4, ','.join(connected_indices.astype(str)))
        except (ValueError, TypeError, IndexError):
            msg = 'Connectivity indices must be an integer between 0 and {0}'.format(self.check_mo.GetItemCount()-1)
            dlg = wx.MessageDialog(self, msg, "Error", wx.OK)
            dlg.ShowModal() # Show it
            dlg.Destroy() # finally destroy it when finished.
