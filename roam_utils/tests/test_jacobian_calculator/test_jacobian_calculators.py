#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser
from roam_learning.helper_functions.factory_from_config import factory_from_config
from roam_learning.helper_functions.jacobian_calculators.jacobian_calculator_factory import jacobian_calculator_factory_base
from roam_learning.forward_models.forward_model_factory import forward_model_factory_base
import unittest
import shutil
import torch


class TestJacobianCalculators(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # runs once in class instantiation
        cls.temporary_experiment_dir = os.path.join(os.environ['ROBOT_LEARNING'],
                                                    'roam_learning/tests/temporary_experiments')
        if 'EXPERIMENTS_DIR' in os.environ:
            cls.previous_experiment_dir = os.environ['EXPERIMENTS_DIR']
        else:
            cls.previous_experiment_dir = None
        os.environ['EXPERIMENTS_DIR'] = cls.temporary_experiment_dir

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

    def test_numpy_jacobian_calculator_spring_damper_numerical(self):
        path = os.path.join(os.environ['ROBOT_LEARNING'],
                            'roam_learning/tests/test_helper_functions/configs/test_numpy_jacobian_calculator_spring_damper.cfg')
        config_data = ConfigParser.ConfigParser()
        config_data.read(path)

        model = factory_from_config(forward_model_factory_base, config_data, section_name='my_model')
        jacobian_calculator = factory_from_config(jacobian_calculator_factory_base, config_data,
                                                  section_name='my_jacobian_calculator')

        x_cur = np.zeros((model.get_state_dim(), 1))
        u_cur = np.full((model.get_action_dim(), 1), 0.1)

        true_fx = [[0.99995014,  0.00995], [-0.00995498,  0.99000014]]
        true_fu = [[4.98574294e-05], [9.95497783e-03]]

        fx, fu = jacobian_calculator.calc_jacobians(x_cur, u_cur)

        self.assertTrue(np.allclose(fx, true_fx))
        self.assertTrue(np.allclose(fu, true_fu))

    def test_pytorch_jacobian_calculator_spring_damper_numerical(self):
        path = os.path.join(os.environ['ROBOT_LEARNING'],
                            'roam_learning/tests/test_helper_functions/configs'
                            '/test_pytorch_jacobian_calculator_spring_damper.cfg')
        config_data = ConfigParser.ConfigParser()
        config_data.read(path)

        model = factory_from_config(forward_model_factory_base, config_data, section_name='my_model')
        jacobian_calculator = factory_from_config(jacobian_calculator_factory_base, config_data,
                                                  section_name='my_jacobian_calculator')

        x_cur = torch.zeros((model.get_state_dim(), 1), requires_grad=True, dtype=torch.float64)
        u_cur = torch.full((model.get_action_dim(), 1), 0.1, requires_grad=True, dtype=torch.float64)

        true_fx = torch.tensor([[0.99995014, 0.00995], [-0.00995498, 0.99000014]], dtype=torch.float64)
        true_fu = torch.tensor([[4.98574294e-05], [9.95497783e-03]],  dtype=torch.float64)

        fx, fu = jacobian_calculator.calc_jacobians(x_cur, u_cur)

        self.assertTrue(torch.allclose(fx, true_fx))
        self.assertTrue(torch.allclose(fu, true_fu))


    def test_numpy_jacobian_calculator_snake_numerical(self):
        path = os.path.join(os.environ['ROBOT_LEARNING'],
                            'roam_learning/tests/test_helper_functions/configs'
                            '/test_numpy_jacobian_calculator_snake_numerical.cfg')
        config_data = ConfigParser.ConfigParser()
        config_data.read(path)

        model = factory_from_config(forward_model_factory_base, config_data, section_name='my_model')
        jacobian_calculator = factory_from_config(jacobian_calculator_factory_base, config_data,
                                                  section_name='my_jacobian_calculator')

        x_cur = np.zeros((model.get_state_dim(), 1))
        u_cur = np.full((model.get_action_dim(), 1), 0.1)

        fx, fu = jacobian_calculator.calc_jacobians(x_cur, u_cur)

    def test_numpy_jacobian_calculator_snake_analytical(self):
        path = os.path.join(os.environ['ROBOT_LEARNING'],
                            'roam_learning/tests/test_helper_functions/configs'
                            '/test_numpy_jacobian_calculator_snake_analytical.cfg')
        config_data = ConfigParser.ConfigParser()
        config_data.read(path)

        model = factory_from_config(forward_model_factory_base, config_data, section_name='my_model')
        jacobian_calculator_numerical = factory_from_config(jacobian_calculator_factory_base, config_data,
                                                            section_name='my_jacobian_calculator_numerical')
        jacobian_calculator_analytical = factory_from_config(jacobian_calculator_factory_base, config_data,
                                                             section_name='my_jacobian_calculator_analytical')

        x_cur = np.zeros((model.get_state_dim(), 1))
        u_cur = np.full((model.get_action_dim(), 1), 0.1)

        fx_numerical, fu_numerical = jacobian_calculator_numerical.calc_jacobians(x_cur, u_cur)
        fx_analytical, fu_analytical = jacobian_calculator_analytical.calc_jacobians(x_cur, u_cur)

        self.assertTrue(np.allclose(fx_numerical, fx_analytical))
        self.assertTrue(np.allclose(fu_numerical, fu_analytical))


    def test_pytorch_jacobian_calculator_snake_anlytical(self):

            path = os.path.join(os.environ['ROBOT_LEARNING'],
                                'roam_learning/tests/test_helper_functions/configs'
                                '/test_pytorch_jacobian_calculator_snake.cfg')
            config_data = ConfigParser.ConfigParser()
            config_data.read(path)

            model = factory_from_config(forward_model_factory_base, config_data, section_name='my_model')
            jacobian_calculator_numerical = factory_from_config(jacobian_calculator_factory_base, config_data,
                                                      section_name='my_jacobian_calculator_numerical')
            jacobian_calculator_analytical = factory_from_config(jacobian_calculator_factory_base, config_data,
                                                                section_name='my_jacobian_calculator_analytical')

            x_cur = torch.zeros((model.get_state_dim(), 1), requires_grad=True, dtype=torch.float64)
            u_cur = torch.full((model.get_action_dim(), 1), 0.1, requires_grad=True, dtype=torch.float64)

            fx_numerical, fu_numerical = jacobian_calculator_numerical.calc_jacobians(x_cur, u_cur)
            fx_analytical, fu_analytical = jacobian_calculator_analytical.calc_jacobians(x_cur, u_cur)

            self.assertTrue(torch.allclose(fx_numerical, fx_analytical))
            self.assertTrue(torch.allclose(fu_numerical, fu_analytical))


if __name__ == '__main__':
    unittest.main()
