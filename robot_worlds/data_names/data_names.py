from __future__ import absolute_import, division, print_function, unicode_literals

class DataNames(object):
    def __init__(self):
        pass

    def initialize_from_config(self, config_data, section_name):
        self.base_type = config_data.get(section_name, 'type')

    def create_param_dict(self):
        return {'base_type': self.base_type}

    def load_param_dict(self, param_dict):
        self.base_type = param_dict['base_type']








