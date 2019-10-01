from __future__ import absolute_import, division, print_function, unicode_literals
import ast
from roam_learning.helper_functions.factory_from_config import factory_from_config
from roam_learning.forward_models.forward_model_factory import forward_model_factory_base


class JacobianCalculator(object):
    def __init__(self, model=None, jacobian_type=None, jacobian_delta_x_default=None, jacobian_delta_u_default=None,
                 auto_delta=None, jacobian_ratio=None):
        self.model = model
        self.jacobian_type = jacobian_type
        self.jacobian_delta_x_default = jacobian_delta_x_default
        self.jacobian_delta_u_default = jacobian_delta_u_default
        self.auto_delta = auto_delta
        self.jacobian_ratio = jacobian_ratio

    def set_model(self, model):
        self.model = model

    def initialize_from_config(self, config_data, section_name):
        model_section_name = config_data.get(section_name, 'model')
        self.model = factory_from_config(forward_model_factory_base, config_data, model_section_name)
        self.jacobian_type = config_data.get(section_name, 'jacobian_type')
        if self.jacobian_type == 'numerical':
            self.auto_delta = ast.literal_eval(config_data.get(section_name, 'auto_delta'))
            self.jacobian_delta_x_default = config_data.getfloat(section_name, 'jacobian_delta_x_default')
            self.jacobian_delta_u_default = config_data.getfloat(section_name, 'jacobian_delta_u_default')
            if self.auto_delta:
                self.jacobian_ratio = config_data.getfloat(section_name, 'jacobian_ratio')
        elif self.jacobian_type == 'analytical':
            pass
        else:
            raise ValueError('jacobian_type: {} not recognized'.format(self.jacobian_type))

    def calc_jacobians(self, x_cur, u_cur, history=None):
        if self.jacobian_type == 'numerical':
            return self.get_jacobians_numerical(x_cur, u_cur, history)
        elif self.jacobian_type == 'analytical':
            return self.get_jacobians_analytical(x_cur, u_cur, history)
        else:
            raise ValueError('jacobian method: {} not recognized'.format(self.jacobian_type))

    def get_empty_jacobians(self, f_dim, x_dim, u_dim):
        raise NotImplementedError

    def get_empty_jacobian_deltas(self, x_dim, u_dim):
        raise NotImplementedError

    def get_jacobians_deltas(self, x_cur, u_cur, auto_delta=False):
        raise NotImplementedError

    def get_jacobians_numerical(self, x_cur, u_cur, history=None):
        raise NotImplementedError

    def get_jacobians_analytical(self, x_cur, u_cur, history=None):
        raise NotImplementedError
