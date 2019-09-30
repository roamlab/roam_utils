import  ast
from roam_learning.helper_functions.factory_from_config import factory_from_config
from roam_learning.robot_worlds.kinematics.kinematics_factory import kinematics_factory_base
from collections import OrderedDict

class SensorModelBase(object):
    """ abstract class

    mapping from robot state to observation implemented by the sense method()

    """

    def __init__(self, attributes):
        self.attributes = attributes

    def initialize_from_config(self, config_data, section_name):
        self.attributes = ast.literal_eval(config_data.get(section_name, 'attributes'))


class SensorModel(SensorModelBase):

    """ sensor model for simulated robot world, uses the kinematics to compute attributes """

    def __init__(self, attributes=None, kinematics=None):
        super().__init__(attributes=attributes)
        self.kinematics = kinematics

    def initialize_from_config(self, config_data, section_name):
        super().initialize_from_config(config_data, section_name)
        kinematics_section_name = config_data.get(section_name, 'kinematics')
        self.kinematics = factory_from_config(kinematics_factory_base, config_data, kinematics_section_name)

    def sense(self, state):
        obs = self._sense(state)
        assert type(obs) == OrderedDict, 'obs is type {} but OrderedDict required'.format(type(obs))
        return obs

    def _sense(self, state):
        raise NotImplementedError

class MujocoSensorModel(SensorModelBase):

    """ sensor model for mujoco robot world, uses MjSim object to compute attributes """

    def __init__(self, attributes=None):
        super().__init__(attributes=attributes)

    def initialize_from_config(self, config_data, section_name):
        super().initialize_from_config(config_data, section_name)

    def sense(self, sim):
        obs = self._sense(sim)
        assert type(obs) == OrderedDict, 'obs is type {} but OrderedDict required'.format(type(obs))
        return obs

    def _sense(self, sim):
        raise NotImplementedError