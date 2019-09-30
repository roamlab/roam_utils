class RobotWorld(object):
    """ abstract class

    take_action() and get_obs() are mandatory

    """
    def __init__(self):
        pass

    def initialize_from_config(self, config_data, section_name):
        pass

    def get_action_dim(self):
        raise NotImplementedError

    def take_action(self, action):
        raise NotImplementedError

    def get_obs(self):
        raise NotImplementedError



