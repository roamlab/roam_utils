from __future__ import absolute_import, division, print_function, unicode_literals


class Trajectory(object):
    def __init__(self, X=None, U=None, cost_calculator=None, time_array=None):
        super(Trajectory, self).__init__()
        self.cost = 0.0
        self.converged = False
        self.improved = False
        self.X = X
        self.U = U
        self.time_array = time_array
        self.traj_length = None
        if self.X is not None:
            self.set_traj_length(len(X))
        self.cost_calculator = cost_calculator

    def initialize(self, traj_length, state_dim, action_dim, cost_calculator=None):
        self.traj_length = traj_length
        self.set_cost_calculator(cost_calculator)

    def set_cost_calculator(self, cost_calculator=None):
        self.cost_calculator = cost_calculator

    def update_cost(self, x, u, x_pre=None):
        raise NotImplementedError

    def calculate_cost(self):
        raise NotImplementedError

    def set_T_idx(self, t, idx):
        raise NotImplementedError

    def set_X_idx(self, x, idx):
        raise NotImplementedError

    def set_U_idx(self, u, idx):
        raise NotImplementedError

    def get_cost_numpy(self):
        raise NotImplementedError

    def set_converged_flag(self, converged):
        raise NotImplementedError

    def set_improved_flag(self, improved):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def get_x_copy(self, t):
        raise NotImplementedError

    def get_u_copy(self, t):
        raise NotImplementedError

    def set_traj_length(self, traj_length):
        self.traj_length = traj_length

    def set_T(self, time_array):
        self.time_array = time_array

    def set_X(self, X):
        self.X = X

    def set_U(self, U):
        self.U = U

    def update_U(self, updated_trajectory):
        raise NotImplementedError

    def get_traj_length(self):
        return self.traj_length