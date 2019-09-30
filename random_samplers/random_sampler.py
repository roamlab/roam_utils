#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import ast
from roam_learning.numpy_random_state import NumpyRandomState


class RandomSampler(object):
    def __init__(self, random_state):
        self.random_state = random_state
        self.dim = None

    def initialize_from_config(self, config_data, section_name):
        random_state_section_name = config_data.get(section_name, 'random_state_section_name')
        self.random_state = NumpyRandomState()
        self.random_state.initialize_from_config(config_data, random_state_section_name)

    def get_value_dim(self):
        raise NotImplementedError

    def get_random_value(self, i):
        raise NotImplementedError

    def get_new_values(self):
        new_values = np.zeros((self.get_value_dim(), 1))
        for i in range(self.get_value_dim()):
            new_values[i] = self.get_random_value(i)
        return new_values


class UniformSampler(RandomSampler):
    def __init__(self, random_state=None, min_rand=None, max_rand=None, partitions=1):
        RandomSampler.__init__(self, random_state)
        self.min_rand = min_rand
        self.max_rand = max_rand
        self.partitions = partitions

    def initialize_from_config(self, config_data, section_name):
        RandomSampler.initialize_from_config(self, config_data, section_name)
        self.min_rand = [float(x) for x in ast.literal_eval(config_data.get(section_name, 'min_rand'))]
        self.max_rand = [float(x) for x in ast.literal_eval(config_data.get(section_name, 'max_rand'))]
        self.dim = len(self.min_rand)
        #self.partitions = config_data.getint(section_name, 'partitions')

    def sample(self):
        sample = np.zeros((self.dim, 1))
        for i in range(self.dim):
            sample[i] = self.random_state.rs.uniform(self.min_rand[i], self.max_rand[i])
        return sample

    #implemented for partitions
    def get_random_sampling_area_coordinates(self):
        return self.min_rand[0], self.max_rand[0], self.min_rand[1], self.max_rand[1]

    #random sampling with partitions
    def sample_with_partitions(self, partition_min_rand, partition_max_rand):
        partition_min_rand_with_dims = self.min_rand
        partition_min_rand_with_dims[0] = partition_min_rand[0]
        partition_min_rand_with_dims[1] = partition_min_rand[1]

        partition_max_rand_with_dims = self.max_rand
        partition_max_rand_with_dims[0] = partition_max_rand[0]
        partition_max_rand_with_dims[1] = partition_max_rand[1]

        sample = np.zeros((self.dim, 1))
        for i in range(self.dim):
            sample[i] = self.random_state.rs.uniform(partition_min_rand_with_dims[i], partition_max_rand_with_dims[i])
        return sample

    def save_seed_state(self, save_dir, name=None):
        self.random_state.save_seed_state(save_dir, name)

    def load_seed_state(self, load_dir, name=None):
        self.random_state.load_seed_state(load_dir, name)


class GaussianSampler(RandomSampler):
    def __init__(self, random_state=None, mu=None, sigma=None):
        RandomSampler.__init__(self, random_state)
        self.mu = mu
        self.sigma = sigma

    def initialize_from_config(self, config_data, section_name):
        RandomSampler.initialize_from_config(self, config_data, section_name)
        self.mu = [float(x) for x in ast.literal_eval(config_data.get(section_name, 'mu'))]
        self.sigma = [float(x) for x in ast.literal_eval(config_data.get(section_name, 'sigma'))]
        self.dim = len(self.sigma)

    def sample(self):
        sample = np.zeros((self.dim, 1))
        for i in range(self.dim):
            sample[i] = self.random_state.rs.normal(self.mu[i], self.sigma[i])
        return sample

    def save_seed_state(self, save_dir, name=None):
        self.random_state.save_seed_state(save_dir, name)

    def load_seed_state(self, load_dir, name=None):
        self.random_state.load_seed_state(load_dir, name)