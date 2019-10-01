from __future__ import absolute_import, division, print_function, unicode_literals
import ast
import copy
import numpy as np
from roam_learning.helper_functions.factory_from_config import factory_from_config
from roam_learning.robot_worlds.kinematics.kinematics_factory import kinematics_factory_base
import math


class DistanceEvaluator(object):
    def __init__(self):
        pass

    def initialize_from_config(self, config_data, section_name):
        self.base_type = config_data.get(section_name, 'type')

    def evaluate(self, state_1, state_2):
        raise NotImplementedError


class StateDistanceEvaluator(DistanceEvaluator):
    def __init__(self):
        DistanceEvaluator.__init__(self)

    def evaluate(self, state_1, state_2):
        # both state_1 and state_2 must be of shape ((-1, 1))
        return np.linalg.norm(state_1 - state_2)


class NormalizedStateDistanceEvaluator(DistanceEvaluator):
    def __init__(self, min_bound=None, max_bound=None):
        DistanceEvaluator.__init__(self)
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.name = 'normalized_state_distance_evaluator'

    def initialize_from_config(self, config_data, section_name):
        DistanceEvaluator.initialize_from_config(self, config_data, section_name)
        self.min_bound = np.array(ast.literal_eval(config_data.get(section_name, 'state_min_bound'))).reshape((-1, 1))
        self.max_bound = np.array(ast.literal_eval(config_data.get(section_name, 'state_max_bound'))).reshape((-1, 1))
        self.scale = abs(self.max_bound-self.min_bound)
        self.mid = .5*(self.max_bound+self.min_bound)

    def normalize_node_state(self, state):
        state_normalized = copy.copy(state)
        state_normalized = (state_normalized-self.mid)/self.scale
        return state_normalized

    def evaluate(self, state_1, state_2):
        assert (len(state_1) == len(state_2))
        state_1_normalized = self.normalize_node_state(state_1)
        state_2_normalized = self.normalize_node_state(state_2)
        return np.linalg.norm(state_1_normalized - state_2_normalized)


class Type1SnakeNormalizedCOMandJointStateDistanceEvaluator(NormalizedStateDistanceEvaluator):
    def __init__(self, min_bound=None, max_bound=None, kinematics=None):
        NormalizedStateDistanceEvaluator.__init__(self, min_bound, max_bound)
        self.kinematics = kinematics

    def initialize_from_config(self, config_data, section_name):
        NormalizedStateDistanceEvaluator.initialize_from_config(self, config_data, section_name)
        kinematics_section_name = config_data.get(section_name, 'kinematics')
        self.kinematics = factory_from_config(kinematics_factory_base, config_data, kinematics_section_name)
        # COM position bounds are set to be the same as head position bounds
        self.scale = np.concatenate((self.scale[:2], self.scale[4:]))
        self.mid = np.concatenate((self.mid[:2], self.mid[4:]))

    def evaluate(self, state_1, state_2):
        com_q_state_1_normalized = self.normalize_node_state(np.concatenate((self.kinematics.compute_com_from_state(state_1), state_1[4:])))
        com_q_state_2_normalized = self.normalize_node_state(np.concatenate((self.kinematics.compute_com_from_state(state_2), state_2[4:])))
        return np.linalg.norm(com_q_state_1_normalized - com_q_state_2_normalized)


class Type2SnakeNormalizedStateDistanceEvaluator(NormalizedStateDistanceEvaluator):
    def __init__(self, min_bound=None, max_bound=None, kinematics=None):
        NormalizedStateDistanceEvaluator.__init__(self, min_bound, max_bound)
        self.kinematics = kinematics

    def initialize_from_config(self, config_data, section_name):
        NormalizedStateDistanceEvaluator.initialize_from_config(self, config_data, section_name)
        kinematics_section_name = config_data.get(section_name, 'kinematics')
        self.kinematics = factory_from_config(kinematics_factory_base, config_data, kinematics_section_name)

    def normalize_node_state(self, state):
        wrapped_state = self.wrap_state_angles(state)
        return NormalizedStateDistanceEvaluator.normalize_node_state(self, wrapped_state)

    def wrap_state_angles(self, state):
        state_return = copy.copy(state)
        qs = self.kinematics.get_q_from_state(state)
        qs_wrapped = copy.deepcopy(qs)
        qs_wrapped = np.asarray([(q + math.pi) % (2 * math.pi) - math.pi for q in qs_wrapped]).reshape((-1, 1))
        state_return[self.kinematics.get_q_idxs()] = qs_wrapped
        return state_return


class ConstantDistanceEvaluator(DistanceEvaluator):
    def __init__(self):
        DistanceEvaluator.__init__(self)
        self.name = 'constant_distance_evaluator'

    def evaluate(self, state_1, state_2):
        return 1


class SnakeClosestNodeDistanceEvaluator(DistanceEvaluator):
    def __init__(self):
        DistanceEvaluator.__init__(self)

    def initialize_from_config(self, config_data, section_name):
        DistanceEvaluator.initialize_from_config(self, config_data, section_name)

    def evaluate(self, state_1, state_2):
        #TODO: update this to fit new snake state order
        return np.linalg.norm(state_1[4:] - state_2[4:])


class SnakeLargestDimNormalizedStateDistanceEvaluator(NormalizedStateDistanceEvaluator):
    def __init__(self):
        NormalizedStateDistanceEvaluator.__init__(self)

    def initialize_from_config(self, config_data, section_name):
        NormalizedStateDistanceEvaluator.initialize_from_config(self, config_data, section_name)
        self.relative_idxs = np.asarray([int(x) for x in
                                         ast.literal_eval(config_data.get(section_name, 'relative_idxs'))])

    def get_max_state_index(self, state_1, state_2):
        max_index = None
        max_magnitude = None
        state_diff = abs(state_1-state_2)
        for i in self.relative_idxs:
            if (max_index is None) or state_diff[i] > max_magnitude:
                max_index = i
                max_magnitude = state_diff[i]
        return max_index

    def evaluate(self, state_1, state_2):
        state_1_normalized = self.normalize_node_state(state_1)
        state_2_normalized = self.normalize_node_state(state_2)
        max_index = self.get_max_state_index(state_1_normalized, state_2_normalized)
        return abs(state_1_normalized[max_index] - state_2_normalized[max_index])


class MapDistanceEvaluator(DistanceEvaluator):
    def __init__(self, kinematics=None):
        DistanceEvaluator.__init__(self)
        self.kinematics = kinematics

    def initialize_from_config(self, config_data, section_name):
        kinematics_section_name = config_data.get(section_name, 'kinematics')
        self.kinematics = factory_from_config(kinematics_factory_base, config_data, kinematics_section_name)

    def convert_to_distance_space(self, state):
        raise NotImplementedError

    def evaluate(self, state_1, state_2):
        converted_state_1 = self.convert_to_distance_space(state_1)
        converted_state_2 = self.convert_to_distance_space(state_2)
        return np.linalg.norm(converted_state_1 - converted_state_2)

    def evaluate_state_and_distance_space_point(self, state, point):
        state_point = self.convert_to_distance_space(state).reshape((-1, 1))
        assert(state_point.shape == point.shape)
        return np.linalg.norm(state_point-point)


class TwoDimPointMapDistanceEvaluator(MapDistanceEvaluator):
    def __init__(self):
        MapDistanceEvaluator.__init__(self)

    def convert_to_distance_space(self, state):
        return self.kinematics.get_p_from_state(state)


class ArmMapDistanceEvaluator(MapDistanceEvaluator):
    def __init__(self, kinematics=None):
        MapDistanceEvaluator.__init__(self, kinematics)

    def convert_to_distance_space(self, state):
        return self.kinematics.compute_end_effector_from_state(state)


class PointMapDistanceEvaluator(MapDistanceEvaluator):
    def __init__(self, kinematics=None):
        MapDistanceEvaluator.__init__(self, kinematics)

    def convert_to_distance_space(self, state):
        return self.kinematics.get_p_from_state(state)


class SnakeMapDistanceEvaluator(MapDistanceEvaluator):
    def __init__(self, kinematics=None):
        MapDistanceEvaluator.__init__(self, kinematics)

    def convert_to_distance_space(self, state):
        return self.kinematics.get_p0_from_state(state)


class SnakeCOMDistanceEvaluator(MapDistanceEvaluator):
    def __init__(self, kinematics=None):
        MapDistanceEvaluator.__init__(self, kinematics)

    def convert_to_distance_space(self, state):
        return self.kinematics.compute_com_from_state(state)


class SnakeCOMandJointDistanceEvaluator(MapDistanceEvaluator):
    def __init__(self, kinematics=None):
        MapDistanceEvaluator.__init__(self, kinematics)

    def convert_to_distance_space(self, state):
        return np.concatenate((self.kinematics.compute_com_from_state(state), state[4:]))

# ToDO: This is just for MPC using the distance evaluator. Kinematics model may need to be added for rrt.
class MujocoMapDistanceEvaluator(MapDistanceEvaluator):
    '''
    Use MujocoDynamics as the kinematics here.
    '''
    def __init__(self, kinematics=None):
        MapDistanceEvaluator.__init__(self, kinematics)

    def convert_to_distance_space(self, state):
        return self.kinematics.get_p0_from_state(state)