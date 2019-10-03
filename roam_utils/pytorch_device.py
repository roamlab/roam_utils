from __future__ import absolute_import, division, print_function, unicode_literals
import torch

class PytorchDevice(object):
    def __init__(self):
        super(PytorchDevice, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def initialize(self, feature_names, label_names, config_data):
        if config_data.has_option('pytorch', 'random_seed'):
            seed = int(config_data.get('pytorch', 'random_seed'))
            torch.manual_seed(seed)
        if config_data.has_option('pytorch', 'device_name'):
            device_name  = config_data.get('pytorch', 'device_name')
            if device_name == 'gpu' and torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif device_name == 'gpu' and not torch.cuda.is_available():
                raise ValueError('gpu is not available on this machine')
            elif device_name == 'cpu':
                self.device = torch.device('cpu')
            else:
                raise ValueError('device name: {} is not available'.format(device_name))
