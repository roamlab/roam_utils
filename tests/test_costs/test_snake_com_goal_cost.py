from __future__ import absolute_import, division, print_function, unicode_literals
import unittest
try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser
import os
import numpy as np
from roam_learning.helper_functions.factory_from_config import factory_from_config
from roam_learning.robot_worlds.robot_world_factory import robot_world_factory_base
from roam_learning.mpc.trajectory_optimizers.cost.cost_factory import cost_factory_base


class TestSnakeCOMGoalCost(unittest.TestCase):
    @classmethod
    def setupClass(cls):
        #runs once in class instantiation
        pass

    @classmethod
    def tearDownClass(cls):
        #runs once when class is torn down
        pass

    def setUp(self):
        #everything in setup gets re instantiated for each test function
        path = os.path.join(os.environ['ROBOT_LEARNING'],
                            'roam_learning/tests/test_costs/configs/test_snake_com_goal_cost.cfg')
        config_data = ConfigParser.ConfigParser()
        config_data.read(path)
        self.robot = factory_from_config(robot_world_factory_base, config_data, 'robot')
        self.cost = factory_from_config(cost_factory_base, config_data, 'cost')

    def test_snake_com_goal_cost(self):
        # get the dimension of the action
        state = np.zeros((self.robot.get_state_dim(), 1))
        self.robot.set_state(state)
        action = np.zeros((self.robot.get_action_dim(), 1))

        state_cost = self.cost.calc_state_cost(state)
        action_cost = self.cost.calc_action_cost(action)

        l = self.cost.get_l(state, action)
        self.assertEqual(l, state_cost+action_cost)


if __name__ == '__main__':
    unittest.main()