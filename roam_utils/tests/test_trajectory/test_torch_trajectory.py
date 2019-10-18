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


class TestTorchTrajectory(unittest.TestCase):

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

        config_data = configparser.ConfigParser()
        config_data.read(os.path.join(os.environ['TEST_DIR'], 'test_trajectory/configs/torch_device.cfg'))
        cls.device = TorchDevice(config_data, 'torch_device')

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
        trajectory1 = TorchTrajectory(self.horizon, self.state_dim, self.action_dim, self.device)

        numpy_X = np.zeros((self.horizon, self.state_dim, 1))
        numpy_U = np.ones((self.horizon-1, self.action_dim, 1))
        numpy_time_array = self.delta_t*np.arange(self.horizon).reshape(-1, 1)
        numpy_X[:] = np.nan
        numpy_U[:] = np.nan
        numpy_time_array[:] = np.nan
        X = torch.as_tensor(numpy_X, dtype=torch.float64, device=self.device.device)
        U = torch.as_tensor(numpy_U, dtype=torch.float64, device=self.device.device)
        time_array = torch.as_tensor(numpy_time_array, dtype=torch.float64, device=self.device.device)
        trajectory2 = TorchTrajectory()
        trajectory2.preset(X, U, time_array)

        self.assertTrue(len(trajectory1.get_X_copy().shape) == 3)
        self.assertTrue(len(trajectory2.get_X_copy().shape) == 3)
        self.assertTrue(np.allclose(trajectory1.get_U_copy().detach().cpu().numpy(), trajectory2.get_U_copy().detach().cpu().numpy(), equal_nan=True))


if __name__ == '__main__':
    unittest.main()
