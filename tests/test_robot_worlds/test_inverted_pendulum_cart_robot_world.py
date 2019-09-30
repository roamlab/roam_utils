from __future__ import absolute_import, division, print_function, unicode_literals
import unittest
from roam_learning.helper_functions.factory_from_config import factory_from_config
from roam_learning.robot_worlds.robot_world_factory import robot_world_factory_base
try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser
import os
import math
import numpy as np
import ast


class TestInvertedPendulumCartRobot(unittest.TestCase):

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
        config_data = ConfigParser.ConfigParser()
        path = os.path.join(os.environ['ROBOT_LEARNING'], 'roam_learning/tests/test_robot_worlds/configs/inverted_pendulum_cart_robot.cfg')
        config_data.read(path)

        self.robot = factory_from_config(robot_world_factory_base, config_data, 'robot')
        initial_state = [float(x) for x in ast.literal_eval(config_data.get('run_test', 'init_robot_state'))]
        self.robot.set_state(np.asarray(initial_state).reshape((-1, 1)))

    def tearDown(self):
        pass

    def test_get_state(self):
        state, time = self.robot.get_state()
        state_dim = self.robot.get_state_dim()
        self.assertEqual(state.shape[0], state_dim)

    def test_take_action(self):

        state0, time0 = self.robot.get_state()
        for i in range(100):
            self.robot.take_action(np.array([1.0]).reshape((-1,1)))
        state1, time1 = self.robot.get_state()
        E_initial = state0[2]**2 - state0[2] * state0[3] * math.cos(state0[1]) + 0.5 * state0[3]**2 - 9.81 * math.cos(state0[1])
        E_final = state1[2]**2 - state1[2] * state1[3] * math.cos(state1[1]) + 0.5 * state1[3]**2 - 9.81 * math.cos(state1[1])
        self.assertTrue(E_final - E_initial - (state1[0]-state0[0]) < 0.004)

    def test_action_dim(self):
        action_dim = self.robot.get_action_dim()
        self.assertEqual(action_dim, 1)

    def test_state_dim(self):
        state_dim = self.robot.get_state_dim()
        self.assertEqual(state_dim, 4)


if __name__ == '__main__':
    unittest.main()