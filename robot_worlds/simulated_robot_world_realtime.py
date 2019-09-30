from __future__ import absolute_import, division, print_function, unicode_literals
import ast
import torch
import numpy as np
import copy
from roam_learning.helper_functions.factory_from_config import factory_from_config
from roam_learning.robot_worlds.data_names.data_names_factory import data_names_factory_base
from roam_learning.robot_worlds.robot_world import RobotWorld
from roam_learning.simulated_dynamics.simulated_dynamics_factory import simulated_dynamics_factory_base
import time
import threading
import queue
from roam_learning.helper_functions.trajectory.numpy_trajectory import NumpyTrajectory
from roam_learning.path_generator import PathGenerator


class SimulatedRobotWorldRealtime(RobotWorld):
    def __init__(self, dynamics=None, data_names=None, action_timeout = 0.01, slowdown_factor=1.0, buffersize = 5000):
        RobotWorld.__init__(self)
        self.dynamics = dynamics
        self.data_names = data_names
        self.state = None
        self.convert_flag = False
        self.data_type = None
        self.lock = threading.Lock()
        self.last_action_timestamp = 0
        self.action_timeout = action_timeout
        self.slowdown_factor = slowdown_factor
        if dynamics is not None:
            self.last_action = np.zeros((dynamics.get_action_dim(), 1))
        self.action_storage = np.asarray([])
        self.updated_storage = False
        self.start_time_stamp = 0
        self.steps_per_action = 1
        self.step_count =0
        self.buffer0 = queue.Queue(buffersize)
        self.buffer1 = queue.Queue(buffersize)
        self.buffer_num_write = 0
        self.buffer_num_read = None
        self.ready_to_read_buffer = False

    def initialize_from_config(self, config_data, section_name):
        RobotWorld.initialize_from_config(self, config_data, section_name)
        self.data_type = config_data.get(section_name, 'data_type')
        dynamics_section_name = config_data.get(section_name, 'dynamics')
        self.dynamics = factory_from_config(simulated_dynamics_factory_base, config_data, dynamics_section_name)
        self.action_dim = self.dynamics.get_action_dim()
        self.steps_per_action = config_data.getint(section_name, 'steps_per_action')
        if self.dynamics is not None:
            self.last_action = np.zeros((self.action_dim, 1))

        self.data_names = factory_from_config(data_names_factory_base, config_data, dynamics_section_name)
        if self.data_type == self.dynamics.get_data_type():
            self.convert_flag = True
        if config_data.has_option(section_name, 'action_timeout'):
            self.action_timeout = config_data.getfloat(section_name, 'action_timeout')
        if config_data.has_option(section_name, 'slowdown_factor'):
            self.slowdown_factor = config_data.getfloat(section_name, 'slowdown_factor')
        if config_data.has_option(section_name, 'buffer_size'):
            buffersize = config_data.getint(section_name, 'buffer_size')
            self.buffer0 = queue.Queue(buffersize)
            self.buffer1 = queue.Queue(buffersize)

    # def loop(self):
    #     delta_t = self.delta_t
    #     desired_duration = delta_t * self.slowdown_factor
    #
    #     while self.running:
    #         start_time = time.time()
    #         action = self.read_action_from_storage()
    #         new_state = self.state
    #         self.save_trajectory_in_buffer(new_state, action, self.robot_time)
    #         new_state = self.dynamics.advance(new_state, action)
    #         self.set_state(new_state)
    #
    #         end_time = time.time()
    #         real_duration = end_time - start_time
    #         self.robot_time += delta_t
    #         self.total_time_steps += 1
    #         if desired_duration < real_duration:
    #             self.missed_time_steps += 1
    #             print('Missed steps: ' + str(self.missed_time_steps))
    #             print('Total steps: ' + str(self.total_time_steps))
    #         else:
    #             time.sleep(desired_duration - real_duration)
    #
    #     self.save_trajectory_in_buffer(new_state, None)


    def loop(self):
        delta_t = self.delta_t
        desired_duration = delta_t * self.slowdown_factor*self.steps_per_action

        while self.running:
            start_time = time.time()
            action = self.read_action_from_storage()
            new_state = self.state
            self.save_trajectory_in_buffer(new_state, action, self.robot_time)
            for i in range(self.steps_per_action):
                new_state = self.dynamics.advance(new_state, action)
            self.set_state(new_state)

            end_time = time.time()
            real_duration = end_time - start_time
            self.robot_time += delta_t*self.steps_per_action
            self.total_time_steps += self.steps_per_action
            if desired_duration < real_duration:
                self.missed_time_steps += 1
                print('Missed steps: ' + str(self.missed_time_steps))
                print('Total steps: ' + str(self.total_time_steps))
            else:
                time.sleep(desired_duration - real_duration)

        self.save_trajectory_in_buffer(new_state, None)



    def start(self):
        self.running = True
        self.trajectory = NumpyTrajectory()
        self.running_thread = threading.Thread(target=self.loop, name='SimulatedRobotWorldRealtime')
        self.robot_time = 0.0
        self.total_time_steps = 0
        self.missed_time_steps = 0
        self.updated_storage = True
        self.action_storage = []
        self.action_timeout = self.delta_t*self.steps_per_action
        self.running_thread.start()

    def stop(self):
        self.running = False
        self.running_thread.join()

    def get_time(self):
        return self.robot_time

    def get_last_action_time(self):
            return self.last_action_timestamp

    def set_state_to_zero(self):
        state = np.zeros((self.get_state_dim(), 1))
        self.set_state(state)

    def get_state(self):
        if self.state is None:
            raise ValueError(
                'ROBOT STATE HAS NOT BEEN SET. Make sure you are setting the state. It is not initialized to anything (to avoid ambiguities), you must have a line in code that calls set_state().')

        with self.lock:
            return copy.copy(self.state), self.get_time()

    # def get_state_and_action_time(self):
    #     if self.state is None:
    #         raise ValueError(
    #             'ROBOT STATE HAS NOT BEEN SET. Make sure you are setting the state. It is not initialized to anything (to avoid ambiguities), you must have a line in code that calls set_state().')
    #     with self.lock:
    #         return copy.copy(self.state), self.get_time(), self.get_last_action_time()

    def set_state(self, x):
        assert x.shape == (self.get_state_dim(), 1)
        with self.lock:
            self.state = copy.copy(x)

    def get_state_dim(self):
        return self.dynamics.get_state_dim()

    def get_action_dim(self):
        return self.dynamics.get_action_dim()

    def get_model_type(self):
        return self.dynamics.base_type

    @property
    def robot_type(self):
        return self.dynamics.base_type

    @property
    def delta_t(self):
        return self.dynamics.get_delta_t()

    def save_action_to_storage(self, actions):
        assert isinstance(actions, np.ndarray), 'simulated robot action must be of type np.ndarray'
        with self.lock:
            self.action_storage = actions.copy()
            self.updated_storage = True
        print("Updated actions in the robot.")





    # def read_action_from_storage(self):
    #     with self.lock:
    #         current_time = self.get_time()
    #         if self.updated_storage:
    #             self.updated_storage = False
    #             self.current_action_index = 0
    #             self.last_action_timestamp = current_time
    #         if current_time - self.last_action_timestamp > self.action_timeout:
    #             self.current_action_index += 1
    #             self.last_action_timestamp = current_time
    #         if len(self.action_storage)<=self.current_action_index:
    #             action = np.zeros((self.get_action_dim(), 1))
    #             return action
    #         action = self.action_storage[self.current_action_index]
    #         return action


    def read_action_from_storage(self):
        with self.lock:
            current_time = self.get_time()
            if self.updated_storage:
                self.updated_storage = False
                self.current_action_index = 0
                self.last_action_timestamp = current_time
            if current_time - self.last_action_timestamp > self.action_timeout:
                self.current_action_index += 1
                self.last_action_timestamp = current_time
            if len(self.action_storage)<=self.current_action_index:
                action = np.zeros((self.get_action_dim(), 1))
                return action
            action = self.action_storage[self.current_action_index]
            return action


    def save_trajectory_in_buffer(self, state, action, robot_time):

        if self.buffer_num_write == 0:
            self.buffer0.put([state, action, robot_time])
            if self.buffer0.full():
                self.buffer_num_write = 1
                self.buffer_num_read = 0
                self.ready_to_read_buffer = True
                if not self.buffer1.empty():
                    raise TimeoutError('The buffer to be written is not empty.')
        elif self.buffer_num_write == 1:
            self.buffer1.put([state, action, robot_time])
            if self.buffer1.full():
                self.buffer_num_write = 0
                self.buffer_num_read = 1
                self.ready_to_read_buffer = True
                if not self.buffer0.empty():
                    raise TimeoutError('The buffer to be written is not empty.')
        else:
            raise ValueError('The buffer num exceeds the number of buffers.')


    def read_trajectory_from_buffer(self):
        state = None
        action = None
        robot_time = None

        if self.buffer_num_read == 0:
            if self.buffer0.empty():
                print('Current buffer is finished')
                self.ready_to_read_buffer = False
            else:
                state, action, robot_time = self.buffer0.get()

        elif self.buffer_num_read == 1:
            if self.buffer1.empty():
                print('Current buffer is finished')
                self.ready_to_read_buffer = False
            else:
                state, action, robot_time = self.buffer1.get()

        return state, action, robot_time



    # def get_state_and_action(self):
    #     with self.lock:
    #         if self.state is None:
    #             raise ValueError('ROBOT STATE HAS NOT BEEN SET. Make sure you are setting the state. It is not initialized to anything (to avoid ambiguities), you must have a line in code that calls set_state().')
    #         if self.step_count < len(self.action_storage):
    #             action_to_take = copy.copy(self.action_storage[self.step_count])
    #         else:
    #             print('MPC takes too long to feed the new actions')
    #             action_to_take = np.zeros((self.action_dim, 1))
    #         return copy.copy(self.state), time.time(), action_to_take, copy.copy(self.step_count)
    #
    # def get_state(self):
    #     if self.state is None:
    #         raise ValueError(
    #             'ROBOT STATE HAS NOT BEEN SET. Make sure you are setting the state. It is not initialized to anything (to avoid ambiguities), you must have a line in code that calls set_state().')
    #     timestamp = np.asarray(time.time())
    #     return copy.copy(self.state), timestamp
    #
    # def set_state(self, x):
    #     assert x.shape == (self.get_state_dim(), 1)
    #     self.state = copy.copy(x)
    #
    # def take_action(self, action):
    #     assert isinstance(action, np.ndarray), 'simulated robot action must be of type np.ndarray'
    #     for _ in range(self.steps_per_action):
    #         self.set_state(self.dynamics.advance(self.state, action))
    #         self.steps += 1
