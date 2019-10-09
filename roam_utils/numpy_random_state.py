import numpy as np
from roam_utils.provenance import PathGenerator
import joblib
import os


class NumpyRandomState(object):
    def __init__(self, config_data, section_name):
        self.seed_value = config_data.getint(section_name, 'random_seed')
        self.rs = np.random.RandomState(self.seed_value)

    def save_seed_state(self, save_dir, name=None):
        save_path = PathGenerator.get_random_state_seed_path(save_dir, name)
        seed_state = self.rs.get_state()
        joblib.dump(seed_state, save_path)

    def load_seed_state(self, load_dir, name=None):
        load_path = PathGenerator.get_random_state_seed_path(load_dir, name)
        if os.path.exists(load_path):
            seed_state = joblib.load(load_path)
            self.rs.set_state(seed_state)
        else:
            print('could not find seed state to load in NumpyRandomState; not loading seed state')
