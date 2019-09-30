import numpy as np
from roam_learning.helper_functions.factory_from_config import factory_from_config
from roam_learning.helper_functions.goal.goal_factory import goal_factory_base
from roam_learning.mpc.trajectory_optimizers.cost.cost import Cost

class SpringDamperGoalCost(Cost):
    """ 
    l = alpha*lA + beta*lB. lA is squared distance between position and 
    specified goal position, lB is squared force
    """
    def __init__(self, robot=None, alpha=None, beta=None, goal=None):
        Cost.__init__(self, robot)
        self.goal = goal
        self.alpha = alpha
        self.beta = beta

    def initialize_from_config(self, config_data, section_name):
        Cost.initialize_from_config(self, config_data, section_name)
        goal_section_name = config_data.get(section_name, 'goal')
        self.goal = factory_from_config(goal_factory_base, config_data, goal_section_name)
        self.alpha = float(config_data.get(section_name, 'alpha'))
        self.beta = float(config_data.get(section_name, 'beta'))

    def get_goal(self):
        return self.goal

    def get_l(self, x, u0):
        lA = 0.5*np.square(x[0] - self.goal.goal[0])
        lB = 0.5*np.sum(np.square(u0))
        return self.alpha*lA + self.beta*lB

    def get_l_x(self, x, u0):
        dldx = np.zeros((self.state_dim, 1))
        dldx[0] = self.alpha*(x[0]-self.goal.goal[0])
        return dldx

    def get_l_xx(self, x, u0):
        d2ld2x = np.zeros((self.state_dim, self.state_dim))
        d2ld2x[0,0] = self.alpha
        return d2ld2x

    def get_l_u(self, x, u0):
        return self.beta*u0

    def get_l_uu(self, x, u0):
        return self.beta*np.eye(self.action_dim)

    def get_l_ux(self, x, u0):
        return np.zeros((self.action_dim, self.state_dim))

    def get_lf(self, x):
        lA = .5*np.square(x[0]-self.goal.goal[0])
        return self.alpha*lA

    def get_lf_x(self, x):
        dldx = np.zeros((self.state_dim, 1))
        dldx[0] = self.alpha*(x[0] - self.goal.goal[0])
        return dldx

    def get_lf_xx(self, x):
        d2ld2x = np.zeros((self.state_dim, self.state_dim))
        d2ld2x[0,0] = self.alpha
        return d2ld2x

