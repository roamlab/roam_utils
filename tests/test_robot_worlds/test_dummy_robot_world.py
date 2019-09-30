from __future__ import absolute_import, division, print_function, unicode_literals
import unittest
import ast
from roam_learning.helper_functions.factory_from_config import factory_from_config
from roam_learning.robot_worlds.robot_world_factory import robot_world_factory_base
try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser
import os
import numpy as np


class TestDummyRobot(unittest.TestCase):
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
        path = os.path.join(os.environ['ROBOT_LEARNING'], 'roam_learning/tests/test_robot_worlds/configs/dummy_numpy_simulated.cfg')
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
        action_zero = np.array([[0.0], [0.0]])
        self.robot.take_action(action_zero)
        time = self.robot.get_time()
        self.assertEqual(time, self.robot.dynamics.get_delta_t())

    def test_action_dim(self):
        action_dim = self.robot.get_action_dim()
        self.assertEqual(action_dim, 2)

    def test_state_dim(self):
        state_dim = self.robot.get_state_dim()
        self.assertEqual(state_dim, 2)

    def test_positive_known_action(self):
        for i in range(0, 5):
            action_known = np.array([[0.5], [0.5]])
            self.robot.take_action(action_known)
        state_after_five = np.array([[2.5], [2.5]])
        self.assertEqual(np.equal(self.robot.get_state()[0], state_after_five).all(), True)

    def test_negative_known_action(self):
        for i in range(0, 5):
            action_known = np.array([[-0.5], [-0.5]])
            self.robot.take_action(action_known)
        state = np.array([[-2.5], [-2.5]])
        self.assertEqual(np.equal(self.robot.get_state()[0], state).all(), True)

if __name__ == '__main__':
    unittest.main()
