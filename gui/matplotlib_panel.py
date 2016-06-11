import numpy as np

import matplotlib
matplotlib.use('WXAgg')

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.figure import Figure

import wx

class CanvasPanel(wx.Window):
    """ Embedding of matplotlib into wxPython

    Attributes
    ----------
    figure : matplotlib.figure.Figure
        Largest (*check) container of the plots
    axes : matplotlib.axes.Axes
        One plot out of potentially many
    plot
        Returned instance of the plot
        Used to get extra informaton
    canvas : matplotlib.backend_bases.FigureCanvas
        Canvas on which modifications to the plots can be made
        Used to modify the existing plots
    sizer : wx.BoxSizer
        Sizer used to orient and reshape each frame used

    Methods
    -------
    draw
        Constructs a graph to store

    """
    def __init__(self, parent):
        """
        Parameters
        ----------
        parent : wx.Window
            Parent of panel
        """
        wx.Window.__init__(self, parent)
        self.sizer = wx.BoxSizer(wx.VERTICAL)

        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.plot = None

        self.canvas = FigureCanvas(self, -1, self.figure)
        self.sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)

        # resize
        self.SetSizer(self.sizer)
        self.Fit()

    def draw(self, f, **args):
        """ Draws the graph

        Parameters
        ----------
        f : function(fig, ax, **args)
            Function that has the arguments
                `fig`, instance of matplotlib.figure.Figure
                `ax`, instance of matplotlib.axes.Axes
                other keyword arguments
            and returns
                Object returned from the plot
        """
        self.plot = f(self.figure, self.axes, **args)

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
