class LibraryOperations(object):
    def __init__(self, config_data, section_name):
        pass

    def initialize_from_config(self, config_data, section_name):
        raise NotImplementedError

    def stack_label_list(self, labels):
        raise NotImplementedError

    def convert_from_numpy(self, list_to_convert):
        raise NotImplementedError

    def convert_to_numpy(self, list_to_convert):
        raise NotImplementedError

    def transpose(self, value):
        raise NotImplementedError

    def subtract(self, minuend, subtrahend):
        raise NotImplementedError

    def divide(self, dividend, divisor):
        raise NotImplementedError

    def add(self, addend_1, addend_2):
        raise NotImplementedError

    def multiply(self, factor_1, factor_2):
        raise NotImplementedError

    def squeeze(self, array):
        raise NotImplementedError

    def concat(self, concat_list, dim):
        raise NotImplementedError

    def flatten(self, array):
        raise NotImplementedError




