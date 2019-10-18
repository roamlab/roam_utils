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


class TestTrajectory(unittest.TestCase):

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
        cls.X = np.arange(cls.horizon).reshape(-1, 1)
        cls.U = np.arange(cls.horizon-1).reshape(-1, 1)
        cls.time_array = cls.delta_t*np.arange(cls.horizon).reshape(-1, 1)

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

    def test_numpy_trajectory(self):
        X = np.arange(self.horizon).reshape(-1, 1)
        U = np.arange(self.horizon-1).reshape(-1, 1)
        time_array = self.delta_t*np.arange(self.horizon).reshape(-1, 1)
        trajectory = NumpyTrajectory()
        trajectory.preset(X, U, time_array)

    def test_torch_trajectory(self):
        config_data = configparser.ConfigParser()
        config_data.read(os.path.join(os.environ['TEST_DIR'], 'test_trajectory/configs/torch_device.cfg'))
        device = TorchDevice(config_data, 'torch_device')
        X = torch.tensor(self.X)
        U = torch.tensor(self.U)
        time_array = torch.tensor(self.delta_t*np.arange(self.horizon))
        trajectory = TorchTrajectory()
        trajectory.preset(X, U, time_array)



if __name__ == '__main__':
    unittest.main()
