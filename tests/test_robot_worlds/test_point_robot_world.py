from __future__ import absolute_import, division, print_function, unicode_literals
import unittest
from roam_learning.helper_functions.factory_from_config import factory_from_config
from roam_learning.robot_worlds.robot_world_factory import robot_world_factory_base
try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser
import os
import numpy as np
import ast


class TestPointRobot(unittest.TestCase):

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
        path = os.path.join(os.environ['ROBOT_LEARNING'], 'roam_learning/tests/test_robot_worlds/configs/point_robot.cfg')
        config_data.read(path)

        self.robot = factory_from_config(robot_world_factory_base, config_data, 'robot')
        initial_state = [float(x) for x in ast.literal_eval(config_data.get('run_test', 'init_robot_state'))]
        self.robot.set_state(np.asarray(initial_state).reshape((-1, 1)))

    def tearDown(self):
        pass

    def test_get_state(self):
        state, time = self.robot.get_state()
        state_dim = self.robot.get_state_dim()
        self.assertEqual(len(state.shape), state_dim)

    def test_take_action(self):
        state0, time0 = self.robot.get_state()
        self.robot.take_action(np.array([1]).reshape((-1, 1)))
        state1, time1 = self.robot.get_state()
        self.robot.take_action(np.array([-1]).reshape((-1, 1)))
        state2, time2 = self.robot.get_state()
        self.assertTrue(0.0 - state0[0][0] < 1e-15)
        self.assertTrue(0.0 - state0[1][0] < 1e-15)
        self.assertEqual(0.0, time0)
        self.assertTrue(8e-6 - state1[0][0] < 1e-15)
        self.assertTrue(4e-3 - state1[1][0] < 1e-15)
        self.assertEqual(4e-3, time1)
        self.assertTrue(1.6e-5 - state2[0][0] < 1e-15)
        self.assertTrue(0.0 - state2[1][0] < 1e-15)
        self.assertEqual(8e-3, time2)

    def test_action_dim(self):
        action_dim = self.robot.get_action_dim()
        self.assertEqual(action_dim, 1)

    def test_state_dim(self):
        state_dim = self.robot.get_state_dim()
        self.assertEqual(state_dim, 2)


if __name__ == '__main__':
    unittest.main()