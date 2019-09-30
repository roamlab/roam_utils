from __future__ import absolute_import, division, print_function, unicode_literals
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
from roam_learning.robot_worlds.render_guis.render_gui import RenderGUI


class PointGUI(RenderGUI):
    def __init__(self, render_rate=None, record_fps=None, record_sim=None, time_frame=None, lim_x=None, lim_y=None,
                 subject=None):
        RenderGUI.__init__(self, render_rate, record_fps, record_sim, time_frame, lim_x, lim_y, subject)

    def render_frame(self, subject, color='b'):
        RenderGUI.render_frame(self, subject, color)
        plt.xlim(self.lim_x[0], self.lim_x[1])
        plt.ylim(self.lim_y[0], self.lim_y[1])
        state, t = subject.get_state()
        x = subject.dynamics.get_x(state)
        self._ax1.scatter(x, 0, color=color)
        self.add_time_info()
