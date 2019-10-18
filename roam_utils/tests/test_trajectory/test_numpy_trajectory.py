#!/usr/bin/env python
import os
import numpy as np
import configparser
import unittest
import shutil
import torch
from roam_utils.torch_device import TorchDevice
from roam_utils.trajectory import NumpyTrajectory
from roam_utils.trajectory import TorchTrajectory


class TestNumpyTrajectory(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # runs once in class instantiation
        cls.temporary_experiment_dir = os.path.join(os.environ['TEST_DIR'],
                                                    'roam_learning/tests/temporary_experiments')
        if 'EXPERIMENTS_DIR' in os.environ:
            cls.previous_experiment_dir = os.environ['EXPERIMENTS_DIR']
        else:
            cls.previous_experiment_dir = None
        os.environ['EXPERIMENTS_DIR'] = cls.temporary_experiment_dir
        cls.delta_t = .01
        cls.horizon = 15
        cls.state_dim = 5
        cls.action_dim = 2

    @classmethod
    def tearDownClass(cls):
        # runs once when class is torn down
        for root, dirs, files in os.walk(cls.temporary_experiment_dir):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))
        if cls.previous_experiment_dir is not None:
            os.environ['EXPERIMENTS_DIR'] = cls.previous_experiment_dir

    def test_preset(self):
        trajectory1 = NumpyTrajectory(self.horizon, self.state_dim, self.action_dim)

        X = np.zeros((self.horizon, self.state_dim, 1))
        U = np.ones((self.horizon-1, self.action_dim, 1))
        time_array = self.delta_t*np.arange(self.horizon).reshape(-1, 1)
        X[:] = np.nan
        U[:] = np.nan
        time_array[:] = np.nan
        trajectory2 = NumpyTrajectory()
        trajectory2.preset(X, U, time_array)

        self.assertTrue(len(trajectory1.get_X_copy().shape) == 3)
        self.assertTrue(len(trajectory2.get_X_copy().shape) == 3)
        self.assertTrue(np.allclose(trajectory1.get_U_copy(), trajectory2.get_U_copy(), equal_nan=True))


if __name__ == '__main__':
    unittest.main()
