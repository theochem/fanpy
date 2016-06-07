import numpy as np

import matplotlib
matplotlib.use('WXAgg')

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.figure import Figure

import wx

class CanvasPanel(wx.Window):
    def __init__(self, parent):
        wx.Window.__init__(self, parent)
        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.SetSizer(self.sizer)
        self.Fit()

    def draw(self, f, **args):
        f(self.figure, self.axes, **args)

# if __name__ == "__main__":
#     app = wx.App()
#     fr = wx.Frame(None, title='test')
#     panel = CanvasPanel(fr)

#     import sys
#     sys.path.append('/home/dkim/Work/mo_diagram')
#     import graph
#     panel.draw(graph.generate_circle_graph, num_points=4)

#     fr.Show()
#     app.MainLoop()
