class Trajectory(object):
    def __init__(self, traj_length=None, state_dim=None, action_dim=None):
        super(Trajectory, self).__init__()
        self.X = None
        self.U = None
        self.time_array = None

    def preset(self, X, U, time_array):
        self.X = X
        self.U = U
        self.time_array = time_array

    def set_T_idx(self, t, idx):
        raise NotImplementedError

    def set_X_idx(self, x, idx):
        raise NotImplementedError

    def set_U_idx(self, u, idx):
        raise NotImplementedError

    def get_t_copy(self, idx):
        raise NotImplementedError

    def get_x_copy(self, idx):
        raise NotImplementedError

    def get_u_copy(self, idx):
        raise NotImplementedError

    def set_T(self, time_array):
        self.time_array = time_array

    def set_X(self, X):
        self.X = X

    def set_U(self, U):
        self.U = U

    def get_traj_length(self):
        return len(self.X)
