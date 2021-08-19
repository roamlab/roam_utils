import torch


class TorchDevice(object):

    def __init__(self):
        self.device = torch.device('cpu')
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def initialize(self, config_data, section_name='pytorch'):
        if config_data.has_option(section_name, 'random_seed'):
            seed = int(config_data.get(section_name, 'random_seed'))
            torch.manual_seed(seed)
        if config_data.has_option(section_name, 'device_name'):
            device_name = config_data.get(section_name, 'device_name')
            if device_name == 'gpu' and torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif device_name == 'gpu' and not torch.cuda.is_available():
                raise ValueError('gpu is not available on this machine')
            elif device_name == 'cpu':
                self.device = torch.device('cpu')
            else:
                raise ValueError('device name: {} is not available'.format(device_name))
