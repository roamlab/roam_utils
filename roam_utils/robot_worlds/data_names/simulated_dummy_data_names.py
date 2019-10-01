from __future__ import absolute_import, division, print_function, unicode_literals
from roam_learning.robot_worlds.data_names.data_names import DataNames


class SimulatedDummyDataNames(DataNames):
    def __init__(self):
        DataNames.__init__(self)
        # These should stay constant unless the rosbag topics change format
        self.state_names = ['state_01', 'state_02']
        self.next_state_names = [x + '_next' for x in self.state_names]
        self.delta_state_names = ['state_01_delta', 'state_02_delta']
        self.norm_delta_state_names = ['state_01_norm_delta', 'state_02_norm_delta']

        self.command_names = ['action_01', 'action_02']
        self.delta_command_names = ['action_01_delta', 'action_02_delta']

        self.abs_names = self.state_names + self.command_names
        self.delta_names = self.delta_state_names + self.delta_command_names