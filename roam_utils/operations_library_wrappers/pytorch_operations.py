from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import numpy as np


class PytorchOperations(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def initialize_from_config(self, config_data, section_name):
        if config_data.has_option(section_name, 'random_seed'):
            random_seed = config_data.getint(section_name, 'random_seed')
        else:
            random_seed = None
        if config_data.has_option(section_name, 'device_name'):
            device_name = config_data.get(section_name, 'device_name')
        else:
            device_name = None
        self.initialize(random_seed, device_name)

    def initialize_from_param_dict(self, param_dict):
        random_seed = param_dict['random_seed']
        device_name = param_dict['device_name']
        self.initialize(random_seed, device_name)

    def initialize(self, random_seed, device_name):
        self.random_seed = random_seed
        self.device_name = device_name
        if random_seed is not None:
            torch.manual_seed(random_seed)
        if device_name is not None:
            if device_name == 'gpu' and torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif device_name == 'gpu' and not torch.cuda.is_available():
                raise ValueError('gpu is not available on this machine')
            elif device_name == 'cpu':
                self.device = torch.device('cpu')
            else:
                raise ValueError('device name: {} is not available'.format(device_name))

    def add_to_param_dict(self, param_dict):
        param_dict['device_name'] = self.device_name
        param_dict['random_seed'] = self.random_seed
        return param_dict

    def stack_label_list(self, labels):
        """

        Args:
            labels:

        Returns:

        """
        labels = torch.stack(labels, dim=1)
        return labels

    def convert_from_numpy(self, list_to_convert):
        """

        Args:
            list_to_convert:

        Returns:

        """
        converted_list = []
        for item in list_to_convert:
            if isinstance(item, list):
                converted_list.append(self.convert_from_numpy(item))
            if isinstance(item, torch.Tensor):
                converted_list.append(item.float().to(self.device))
            elif isinstance(item, np.ndarray):
                converted_list.append(torch.from_numpy(item).float().to(self.device))
            else:
                raise ValueError(
                    'item is a list of type {}. This is not supported by pytorch_forward_model conversion'.format(
                        type(item)))
        return tuple(converted_list)

    def convert_to_numpy(self, list_to_convert):
        """

        Args:
            list_to_convert:

        Returns:

        """
        converted_list = []
        for item in list_to_convert:
            if isinstance(item, torch.Tensor):
                converted_list.append(item.detach().cpu().numpy())
            elif isinstance(item, np.ndarray):
                converted_list.append(item)
            else:
                raise ValueError('item is a list of type {}. '
                                 'This is not supported by pytorch_forward_model conversion'.format(type(item)))
        return tuple(converted_list)

    def transpose(self, value):
        return value.t()

    def subtract(self, minuend, subtrahend):
        return minuend.to(self.device) - subtrahend.to(self.device)

    def divide(self, dividend, divisor):
        return dividend.to(self.device) / divisor.to(self.device)

    def add(self, addend_1, addend_2):
        return addend_1.to(self.device) + addend_2.to(self.device)

    def multiply(self, factor_1, factor_2):
        return factor_1.to(self.device) * factor_2.to(self.device)

    def sqrt(self, array):
        return torch.sqrt(array)

    def sum(self, array):
        return torch.sum(array)

    def square(self, array):
        return array.pow(2)

    def squeeze(self, array):
        return array.squeeze()

    def concatenate(self, concat_list, dim=0):
        return torch.cat(concat_list, dim)

    def flatten(self, array):
        return array.contiguous().view(1, -1)

    def tensor(self, data, dtype=torch.float64, requires_grad=False):
        """ create tensor on the right device """
        return torch.tensor(data, dtype=dtype, device=self.device, requires_grad=requires_grad)

    def zeros(self, shape, dtype=torch.float64, requires_grad=False):
        return torch.zeros(shape, dtype=dtype, device=self.device, requires_grad=requires_grad)

    def ones(self, shape, dtype=torch.float64, requires_grad=False):
        return torch.ones(shape, dtype=dtype, device=self.device, requires_grad=requires_grad)

    def full(self, shape, val, dtype=torch.float64, requires_grad=False):
        return torch.full(shape, val, dtype=dtype, device=self.device, requires_grad=requires_grad)

    def eye(self, n, dtype=torch.float64, requires_grad=False):
        return torch.eye(n, dtype=dtype, device=self.device, requires_grad=requires_grad)

    def zeros_like(self, input, dtype=torch.float64, requires_grad=False):
        return torch.zeros_like(input, dtype=dtype, device=self.device, requires_grad=requires_grad)





