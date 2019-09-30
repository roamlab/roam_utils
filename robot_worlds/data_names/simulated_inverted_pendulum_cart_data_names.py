from __future__ import absolute_import, division, print_function, unicode_literals

from roam_learning.robot_worlds.data_names.data_names import DataNames


class SimulatedInvertedPendulumCartDataNames(DataNames):
    def __init__(self):
        DataNames.__init__(self)
        self.x = ['x']
        self.q = ['q']
        self.xd = ['xd']
        self.qd = ['qd']

        self.force = ['f']

        self.x_delta = ['x_delta']
        self.q_delta = ['q_delta']
        self.xd_delta = ['xd_delta']
        self.qd_delta = ['qd_delta']
        self.x_norm_delta = ['x_norm_delta']
        self.q_norm_delta = ['q_norm_delta']
        self.xd_norm_delta = ['xd_norm_delta']
        self.qd_norm_delta = ['qd_norm_delta']

        self.force_delta = ['f_delta']

        self.abs_names = self.x+self.q+self.xd+self.qd+self.force
        self.delta_names = self.x_delta+self.q_delta+self.xd_delta+self.qd_delta+self.force_delta

        self.command_names = self.force
        self.delta_command_names = self.force_delta
        self.state_names = self.x+self.q+self.xd+self.qd
        self.next_state_names = [x + '_next' for x in self.state_names]
        self.delta_state_names = self.x_delta+self.q_delta+self.xd_delta+self.qd_delta
        self.norm_delta_state_names = self.x_norm_delta+self.q_norm_delta+self.xd_norm_delta+self.qd_norm_delta
