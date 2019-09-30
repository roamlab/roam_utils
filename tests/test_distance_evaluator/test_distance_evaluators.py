#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser
from roam_learning.helper_functions.factory_from_config import factory_from_config
from roam_learning.helper_functions.distance_evaluators.distance_evaluator_factory import distance_evaluator_factory_base
import unittest
import shutil
import copy


class TestDistanceEvaluators(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # runs once in class instantiation
        cls.temporary_experiment_dir = os.path.join(os.environ['ROBOT_LEARNING'], 'roam_learning/tests/temporary_experiments')

    @classmethod
    def tearDownClass(cls):
        #runs once when class is torn down
        for root, dirs, files in os.walk(cls.temporary_experiment_dir):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))

    def test_normalized_state_distance_evaluator(self):
        path = os.path.join(os.environ['ROBOT_LEARNING'],
                            'roam_learning/tests/test_helper_functions/configs/test_normalized_state_distance_evaluator.cfg')
        config_data = ConfigParser.ConfigParser()
        config_data.read(path)

        state = np.array([1, 2, .5, .5, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]).reshape((-1, 1))
        state_min_bound = np.array([-.6, -.6, -1, -4, -1.5, -1.5, -1.5, -1.5, -1.5, -20, -20, -20, -20, -20]).reshape((-1, 1))
        state_max_bound = np.array([.6, .6, 1, 4, 1.5, 1.5, 1.5, 1.5, 1.5, 20, 20, 20, 20, 20]).reshape((-1, 1))

        expected_normalized_state = copy.deepcopy(state)
        for i in range(0, len(state)):
            scale = abs(state_max_bound[i] - state_min_bound[i])
            mid = 0.5 * (state_max_bound[i] + state_min_bound[i])
            expected_normalized_state[i] = (expected_normalized_state[i] - mid) / scale

        normalized_state_distance_evaluator = factory_from_config(distance_evaluator_factory_base, config_data, 'distance_evaluator')

        normalized_state = normalized_state_distance_evaluator.normalize_node_state(state)

        self.assertTrue(np.allclose(normalized_state, expected_normalized_state))

    def test_snake_largest_dim_normalized_state_distance_evaluator(self):
        path = os.path.join(os.environ['ROBOT_LEARNING'],
                            'roam_learning/tests/test_helper_functions/configs/test_snake_largest_dim_normalized_state_distance_evaluator.cfg')
        config_data = ConfigParser.ConfigParser()
        config_data.read(path)

        state_1 = np.array([1, 2, .5, .5, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]).reshape((-1, 1))
        state_2 = np.array([1, 2, .5, .5, 1, 1, 1, 1, 1, 2, 2, 9, 2, 2]).reshape((-1, 1))

        distance_evaluator = factory_from_config(distance_evaluator_factory_base, config_data, 'distance_evaluator')

        self.assertEqual(distance_evaluator.get_max_state_index(state_1, state_2), 11)

if __name__ == '__main__':
    unittest.main()
