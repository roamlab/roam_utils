from __future__ import absolute_import, division, print_function, unicode_literals
from roam_learning.robot_worlds.data_names.data_names import DataNames


class SimulatedSpringDamperDataNames(DataNames):
    def __init__(self):
        DataNames.__init__(self)
        self.position = ['pos']
        self.velocity = ['vel']
        self.force = ['force']

        self.position_delta = ['pos_delta']
        self.velocity_delta = ['vel_delta']
        self.force_delta = ['force_delta']

        self.position_norm_delta = ['pos_norm_delta']
        self.velocity_norm_delta = ['vel_norm_delta']

        self.delta_names = self.position_delta+self.velocity_delta+self.force_delta
        self.abs_names = self.position+self.velocity+self.force

        self.command_names = self.force
        self.delta_command_names = self.force_delta
        self.state_names = self.position+self.velocity
        self.next_state_names = [x + '_next' for x in self.state_names]
        self.delta_state_names = self.position_delta+self.velocity_delta
        self.norm_delta_state_names = self.position_norm_delta+self.velocity_norm_delta
