from __future__ import absolute_import, division, print_function, unicode_literals
import ast
import numpy
from roam_learning.mpc.trajectory_optimizers.cost.cost import Cost
from roam_learning.robot_worlds.robot_world_factory import robot_world_factory_base
from roam_learning.helper_functions.factory_from_config import factory_from_config
from roam_learning.helper_functions.goal.goal_factory import goal_factory_base

'''
_Robot_List = ['newtonian_type_1_snake',
               'mujoco_swimmer', 
               'mujoco_half_cheetah', 
               'mujoco_cart_pole']

_Snake_Variable_List = ['position_x', 'position_y', 'joint_angle', 'velocity_x', 'velocity_y', 'angular_velocity']
_Swimmer_Variable_List = ['position_x', 'position_y', 'joint_angle', 'velocity_x', 'velocity_y', 'angular_velocity']
_Half_Cheetah_Variable_List = ['body_position_x', 'body_position_y', 'body_rot', 'back_thigh_rot', 'back_shin_rot',
                      'back_foot_rot',' front_thigh_rot', 'front_shin_rot', 'front_foot_rot',
                      'body_velocity_x', 'body_velocity_y', 'body_rot_velocity',
                      'back_thigh_rot_velocity', 'back_shin_rot_velocity', 'back_foot_rot_velocity',
                      'front_thigh_rot_velocity', 'front_shin_rot_velocity', 'front_foot_rot_velocity']
_Cart_Pole_Variable_list = ['cart_pos', 'pole_angle', 'cart_vel', 'pole_angular_vel']
'''


class GeneralizedQuadCost(Cost):
    def __init__(self, robot=None, alpha=None, beta=None, interested_variable_name=None, target_value=None, goal_name=None, goal=None):
        Cost.__init__(self, robot)
        self.alpha = alpha
        self.beta = beta
        self.target_value = target_value
        self.interested_variable_index = []
        self.interested_variable_name = interested_variable_name
        self.goal_name = goal_name
        self.goal = goal
        self.if_generalized_cost = True

        if robot is not None:
            self.set_dims_from_robot(robot)
            self.get_interested_variable_index(robot)
            self.set_goal_value_in_cost()


    def initialize_from_config(self, config_data, section_name):
        Cost.initialize_from_config(self, config_data, section_name)
        self.alpha = numpy.asarray([float(x) for x in ast.literal_eval(config_data.get(section_name, 'weights_of_interested_values'))])

        self.beta = numpy.asarray([float(x) for x in ast.literal_eval(config_data.get(section_name, 'weights_of_actions'))])

        self.target_value = numpy.asarray([float(x) for x in ast.literal_eval(config_data.get(section_name, 'target_value'))])

        self.interested_variable_index = []
        self.interested_variable_name = ast.literal_eval(config_data.get(section_name, 'interested_variable_name'))
        if config_data.has_option(section_name,'goal'):
            goal_section_name = config_data.get(section_name, 'goal')
            self.goal = factory_from_config(goal_factory_base, config_data, goal_section_name)
        else:
            self.goal = None

        if config_data.has_option(section_name,'robot'):
            robot_section_name = config_data.get(section_name, 'robot')
            robot = factory_from_config(robot_world_factory_base, config_data, robot_section_name)
        elif config_data.has_option('robot', 'type'):
            robot_type = config_data.get('robot', 'type')
            robot = robot_world_factory_base(robot_type)
            robot.initialize_from_config(config_data, 'robot')

        self.set_dims_from_robot(robot)
        self.get_interested_variable_index(robot)
        if self.goal is not None:
            self.set_goal_value_in_cost()

    def set_goal_value_in_cost(self):
        for i in range(len(self.interested_variable_name)):
            if self.goal_name == self.interested_variable_name[i]:
                if self.target_value[i] != self.goal.goal:
                    self.target_value[i] = self.goal.goal
                    print('Reset the target value of the goal in the cost')

    def set_dims_from_robot(self, robot):
        raise NotImplementedError

    def get_interested_variable_index(self, robot):
        raise NotImplementedError

    def get_l(self, state, action, x_pre=None):
        assert len(self.alpha) == len(self.interested_variable_index) and len(self.beta) == self.action_dim
        state_of_interest = list([state[i] for i in self.interested_variable_index])
        state_of_interest = numpy.asarray(state_of_interest)
        state_of_interest = numpy.reshape(state_of_interest, -1)
        action = numpy.reshape(action, -1)
        lA = 0.5*numpy.dot(self.alpha, numpy.square(state_of_interest - self.target_value))
        lB = 0.5*numpy.dot(self.beta, numpy.square(action))
        if (lA is numpy.nan) or (lB is numpy.nan):
            print('The cost contains NAN.')
        return lA + lB

    def get_l_x(self, state, action, x_pre=None):
        assert len(self.alpha) == len(self.interested_variable_index)
        l_x = numpy.zeros((self.state_dim, 1))
        for i in range(len(self.interested_variable_index)):
            l_x[self.interested_variable_index[i]] = self.alpha[i]*(state[self.interested_variable_index[i]]-self.target_value[i])
        return l_x

    def get_l_xx(self, state, action, x_pre=None):
        l_xx = numpy.zeros((self.state_dim, self.state_dim))
        for i in range(len(self.interested_variable_index)):
            l_xx[self.interested_variable_index[i], self.interested_variable_index[i]] = self.alpha[i]
        return l_xx

    def get_l_u(self, state, action, x_pre=None):
        action = numpy.reshape(action, -1)
        assert self.beta.shape == action.shape
        l_u = numpy.multiply(self.beta, action)
        l_u = numpy.reshape(l_u, (len(l_u),1))
        return l_u

    def get_l_uu(self, state, action, x_pre=None):
        return numpy.diag(self.beta)

    def get_l_ux(self, state, action, x_pre=None):
        return numpy.zeros((self.action_dim, self.state_dim))

    def get_lf(self, state, x_pre=None):
        assert len(self.alpha) == len(self.interested_variable_index)
        state_of_interest = list(state[i] for i in self.interested_variable_index)
        state_of_interest = numpy.asarray(state_of_interest)
        state_of_interest = numpy.reshape(state_of_interest, -1)
        lA = 0.5*numpy.dot(self.alpha, numpy.square(state_of_interest - self.target_value))
        if (lA is numpy.nan):
            print('The cost contains NAN.')
        return lA

    def get_lf_x(self, state, x_pre=None):
        return self.get_l_x(state=state, action=None)

    def get_lf_xx(self, state, x_pre=None):
        return self.get_l_xx(state=state, action=None)


class RoamSnakeQuadCost(GeneralizedQuadCost):

    def set_dims_from_robot(self, robot):
        Cost.set_dims_from_robot(self, robot)
        self.num_links = robot.dynamics.get_num_links()

    def get_interested_variable_index(self, robot):
        self.set_dims_from_robot(robot)
        self.interested_variable_index = []
        for x in self.interested_variable_name:
            if x == "position_x":
                self.interested_variable_index.append(0)
            elif x == "position_y":
                self.interested_variable_index.append(1)
            elif x == "joint_angle":
                for j in self.num_links:
                    self.interested_variable_index.append(2+j)
            elif x == "velocity_x":
                self.interested_variable_index.append(self.num_links+2)
            elif x == "velocity_y":
                self.interested_variable_index.append(self.num_links+3)
            elif x == "angular_velocity":
                for j in self.num_links:
                    self.interested_variable_index.append(self.num_links+4+j)
        self.interested_variable_index = numpy.asarray(self.interested_variable_index)
        self.interested_variable_index.reshape((len(self.interested_variable_index), 1))

class SwimmerQuadCost(GeneralizedQuadCost):

    def set_dims_from_robot(self, robot):
        Cost.set_dims_from_robot(self, robot)
        self.num_links = robot.dynamics.mujoco_model.nq-2

    def get_interested_variable_index(self, robot):
        self.set_dims_from_robot(robot)
        for x in self.interested_variable_name:
            if x == "position_x":
                self.interested_variable_index.append([0])
            elif x == "position_y":
                self.interested_variable_index.append([1])
            elif x == "joint_angle":
                for j in self.num_links:
                    self.interested_variable_index.append([2+j])
            elif x == "velocity_x":
                self.interested_variable_index.append([self.num_links+2])
            elif x == "velocity_y":
                self.interested_variable_index.append([self.num_links+3])
            elif x == "angular_velocity":
                for j in self.num_links:
                    self.interested_variable_index.append([self.num_links+4+j])
        self.interested_variable_index = numpy.asarray(self.interested_variable_index)


class HalfCheetahQuadCost(GeneralizedQuadCost):

    def set_dims_from_robot(self, robot):
        Cost.set_dims_from_robot(self, robot)

    def get_interested_variable_index(self, robot):
        self.set_dims_from_robot(robot)
        for x in self.interested_variable_name:
            if x == "body_position_x":
                self.interested_variable_index.append([0])
            elif x == "body_position_y":
                self.interested_variable_index.append([1])
            elif x == "body_rot":
                self.interested_variable_index.append([2])
            elif x == "back_thigh_rot":
                self.interested_variable_index.append([3])
            elif x == "back_shin_rot":
                self.interested_variable_index.append([4])
            elif x == "back_foot_rot":
                self.interested_variable_index.append([5])
            elif x == "front_thigh_rot":
                self.interested_variable_index.append([6])
            elif x == "front_shin_rot":
                self.interested_variable_index.append([7])
            elif x == "front_foot_rot":
                self.interested_variable_index.append([8])
            elif x == "body_velocity_x":
                self.interested_variable_index.append([9])
            elif x == "body_velocity_y":
                self.interested_variable_index.append([10])
            elif x == "body_rot_velocity":
                self.interested_variable_index.append([11])
            elif x == "back_thigh_rot_velocity":
                self.interested_variable_index.append([12])
            elif x == "back_shin_rot_velocity":
                self.interested_variable_index.append([13])
            elif x == "back_foot_rot_velocity":
                self.interested_variable_index.append([14])
            elif x == "front_thigh_rot_velocity":
                self.interested_variable_index.append([15])
            elif x == "front_shin_rot_velocity":
                self.interested_variable_index.append([16])
            elif x == "front_foot_rot_velocity":
                self.interested_variable_index.append([17])
        self.interested_variable_index = numpy.asarray(self.interested_variable_index)

class CartPoleQuadCost(GeneralizedQuadCost):

    def set_dims_from_robot(self, robot):
        Cost.set_dims_from_robot(self, robot)

    def get_interested_variable_index(self, robot):
        self.set_dims_from_robot(robot)
        for x in self.interested_variable_name:
            if x == "cart_pos":
                self.interested_variable_index.append([0])
            elif x == "pole_angle":
                self.interested_variable_index.append([1])
            elif x == "cart_vel":
                self.interested_variable_index.append([2])
            elif x == "pole_angular_vel":
                self.interested_variable_index.append([3])
        self.interested_variable_index = numpy.asarray(self.interested_variable_index)

