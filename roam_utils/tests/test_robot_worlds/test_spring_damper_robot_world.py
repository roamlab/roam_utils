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


class TestSpringDamperRobot(unittest.TestCase):
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
        path = os.path.join(os.environ['ROBOT_LEARNING'], 'roam_learning/tests/test_robot_worlds/configs/spring_damper_robot.cfg')
        config_data.read(path)
        self.robot = factory_from_config(robot_world_factory_base, config_data, 'robot')
        self.initial_state = np.asarray([float(x) for x in ast.literal_eval(config_data.get('run_test', 'init_robot_state'))]).reshape((-1, 1))
        self.robot.set_state(self.initial_state)

    def get_state_after_five_actions_robot(self, robot):
        robot.set_state(np.asarray(self.initial_state).reshape((-1, 1)))
        action_dim = robot.get_action_dim()
        for i in range(5):
            robot_action = np.zeros((action_dim, 1))
            robot_action[:] = 1
            robot.take_action(robot_action)
            state_after_action = robot.get_state()
        return state_after_action

    def get_state_after_five_actions_dynamics(self, dynamics):
        action_dim = dynamics.get_action_dim()
        state_before_action = self.initial_state
        for i in range(5):
            action = np.zeros((action_dim, 1))
            action[:] = 1
            state_after_action = dynamics.advance(state_before_action, action)
            state_before_action = state_after_action
        return state_after_action

    def test_get_state(self):
        state, time = self.robot.get_state()
        state_dim = self.robot.get_state_dim()
        self.assertEqual(len(state.shape), state_dim)

    def test_action_dim(self):
        action_dim = self.robot.get_action_dim()
        self.assertEqual(action_dim, 1)

    def test_state_dim(self):
        state_dim = self.robot.get_state_dim()
        self.assertEqual(state_dim, 2)

    def test_final_value(self):
        # Need to be at steady state.
        u = np.array([10]).reshape((-1, 1))
        self.robot.take_action(u)
        final_value = self.robot.get_state()
        # print final_value
        self.assertTrue(abs(final_value[0][0][0]- u[0]/self.robot.dynamics.stiffness)<1e-3)


if __name__ == '__main__':
    unittest.main()


