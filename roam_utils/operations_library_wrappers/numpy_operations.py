from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np


class NumpyOperations(object):
    def __init__(self):
        self.device = None

    def initialize_from_config(self, config_data, section_name):
        if config_data.has_option(section_name, 'random_seed'):
            random_seed = config_data.getint(section_name, 'random_seed')
        else:
            random_seed = None
        self.initialize(random_seed)

    def initialize_from_param_dict(self, param_dict):
        random_seed = param_dict['random_seed']
        self.initialize(random_seed)

    def initialize(self, random_seed):
        self.random_seed = random_seed

    def add_to_param_dict(self, param_dict):
        param_dict['random_seed'] = self.random_seed
        return param_dict

    def stack_label_list(self, labels):
        """

        Args:
            labels:

        Returns:

        """
        labels = np.stack(labels, axis=1)
        return labels

    def convert_from_numpy(self, list_to_convert):
        """

        Args:
            list_to_convert:

        Returns:

        """
        return tuple(list_to_convert)

    def convert_to_numpy(self, list_to_convert):
        """

        Args:
            list_to_convert:

        Returns:

        """
        return tuple(list_to_convert)

    def transpose(self, value):
        return value.T

    def subtract(self, minuend, subtrahend):
        return minuend - subtrahend

    def divide(self, dividend, divisor):
        return dividend / divisor

    def add(self, addend_1, addend_2):
        return addend_1 + addend_2

    def multiply(self, factor_1, factor_2):
        return factor_1 * factor_2

    def sqrt(self, array):
        return np.sqrt(array)

    def sum(self, array):
        return np.sum(array)

    def square(self, array):
        return array.pow(2)

    def squeeze(self, array):
        return array.squeeze()

    def concatenate(self, concat_list, dim=0):
        return np.concatenate(concat_list, dim)

    def flatten(self, array):
        return array.contiguous().view(1, -1)

    def zeros(self, shape, dtype=np.float64):
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=np.float64):
        return np.ones(shape, dtype=dtype)

    def full(self, shape, val, dtype=np.float64):
        return np.full(shape, val, dtype=dtype)

    def eye(self, n, dtype=np.float64):
        return np.eye(n, dtype=dtype)

    def zeros_like(self, input, dtype=np.float64):
        return np.zeros_like(input, dtype=dtype)





