from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import copy
from roam_learning.helper_functions.factory_from_config import factory_from_config
from roam_learning.robot_worlds.data_names.data_names_factory import data_names_factory_base
from roam_learning.robot_worlds.robot_world import RobotWorld
from roam_learning.simulated_dynamics.simulated_dynamics_factory import simulated_dynamics_factory_base
from roam_learning.robot_worlds.sensor_models.sensor_model_factory import sensor_model_factory_base
import mujoco_py


class SimulatedRobotWorldBase(RobotWorld):
    def __init__(self, steps_per_action=None):
        super().__init__()
        self.steps_per_action = steps_per_action
        self.steps = 0
        self.sensor_model = None

    def initialize_from_config(self, config_data, section_name):
        super().initialize_from_config(config_data, section_name)
        self.steps_per_action = config_data.getint(section_name, 'steps_per_action')
        if config_data.has_option(section_name, 'sensor_model'):
            sensor_model_section_name = config_data.get(section_name, 'sensor_model')
            self.sensor_model = factory_from_config(sensor_model_factory_base, config_data, sensor_model_section_name)

    def get_state_dim(self):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def set_state(self, state):
        raise NotImplementedError

    def set_state_to_zero(self):
        state = np.zeros((self.get_state_dim(), 1))
        self.set_state(state)

    def get_delta_t(self):
        raise NotImplementedError

    @property
    def delta_t(self):
        return self.get_delta_t()

    def reset_time(self):
        self.steps = 0

    def get_time(self):
        return self.steps * self.get_delta_t()

    def set_time(self, time):
        self.steps = time/self.get_delta_t()

    def get_model_base_type(self):
        raise NotImplementedError

    @property
    def base_type(self):
        return self.get_model_base_type()


class SimulatedRobotWorld(SimulatedRobotWorldBase):
    def __init__(self, steps_per_action=None, data_names=None, dynamics=None):
        super().__init__(steps_per_action=steps_per_action)
        self.data_names = data_names
        self.dynamics = dynamics
        self.state = None

    def initialize_from_config(self, config_data, section_name):
        super().initialize_from_config(config_data, section_name)
        dynamics_section_name = config_data.get(section_name, 'dynamics')
        self.data_names = factory_from_config(data_names_factory_base, config_data, dynamics_section_name)
        self.dynamics = factory_from_config(simulated_dynamics_factory_base, config_data, dynamics_section_name)

    def get_action_dim(self):
        return self.dynamics.get_action_dim()

    def get_state_dim(self):
        return self.dynamics.get_state_dim()

    def take_action(self, action):
        assert isinstance(action, np.ndarray), 'simulated robot action must be of type np.ndarray'
        for _ in range(self.steps_per_action):
            self.set_state(self.dynamics.advance(self.state, action))
            self.steps += 1

    def get_state(self):
        if self.state is None:
            raise ValueError('ROBOT STATE HAS NOT BEEN SET. Make sure you are setting the state. '
                             'It is not initialized to anything (to avoid ambiguities), '
                             'you must have a line in code that calls set_state().')
        timestamp = np.asarray([self.get_time()])
        return copy.copy(self.state), timestamp

    def get_obs(self):
        assert self.sensor_model != None, 'sensor model has not been in initialized'
        state, _ = self.get_state()
        obs = self.sensor_model.sense(state)
        return obs

    def set_state(self, x):
        self.get_state_dim()
        assert x.shape[0] == self.get_state_dim()
        self.state = copy.copy(x)

    def get_delta_t(self):
        return self.dynamics.get_delta_t()

    def get_model_base_type(self):
        return self.dynamics.base_type


class MujocoRobotWorld(SimulatedRobotWorldBase):
    def __init__(self, steps_per_action=None, xml_path=None):
        super().__init__(steps_per_action=steps_per_action)
        if xml_path is not None:
            mujoco_model = mujoco_py.load_model_from_path(xml_path)
            self.sim = mujoco_py.MjSim(mujoco_model)

    def initialize_from_config(self, config_data, section_name):
        super().initialize_from_config(config_data, section_name)
        self.steps_per_action = config_data.getint(section_name, 'steps_per_action')
        xml_path = config_data.get(section_name, 'xml_path')
        mujoco_model = mujoco_py.load_model_from_path(xml_path)
        self.sim = mujoco_py.MjSim(mujoco_model)

    def get_action_dim(self):
        return self.sim.model.nu

    def get_state_dim(self):
        state_dim = self.sim.model.nq + self.sim.model.nv
        return state_dim

    def take_action(self, action):
        assert action.shape == (self.get_action_dim(), 1)
        self.sim.data.ctrl[:] = action.flatten()
        for i in range(self.steps_per_action):
            self.sim.step()
            self.steps += 1

    def get_delta_t(self):
        return self.sim.model.opt.timestep

    def get_state(self):
        sim_state = self.sim.get_state()
        state = np.concatenate([sim_state.qpos, sim_state.qvel]).reshape(-1, 1)
        return state, self.get_time()

    def get_obs(self):
        assert self.sensor_model is not None, 'sensor model not initialized'
        obs = self.sensor_model.sense(self.sim)
        return obs

    def set_state(self, state):
        nq = self.sim.model.nq
        nv = self.sim.model.nv
        assert state.shape == (nq + nv, 1)
        qpos = state[0:nq].flatten()
        qvel = state[-nv - 1:-1].flatten()
        sim_state_prev = self.sim.get_state()
        sim_state = mujoco_py.MjSimState(sim_state_prev.time, qpos, qvel, sim_state_prev.act, sim_state_prev.udd_state)
        self.sim.set_state(sim_state)
        self.sim.forward()

    def get_model_base_type(self):
        raise NotImplementedError
