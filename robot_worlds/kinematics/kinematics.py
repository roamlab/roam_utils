from __future__ import absolute_import, division, print_function, unicode_literals


class Kinematics(object):
    """
    provides methods that return kinematic attributes such as
    position and velocity of end-effector, center of mass etc.

    """
    def __init__(self):
        """
        :param model: instance of class ChainDynamics

        """
        self.setup_dispatch()

    def initialize_from_config(self, config_data, section_name):
        self.kinematics_type = config_data.get(section_name, 'type')

    def setup_dispatch(self):
        self.compute_from_state_dispatch = {}

    def compute_point_from_state(self, point_type, state):
        raise NotImplementedError










