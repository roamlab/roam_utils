import numpy as np
from roam_utils.provenance import PathGenerator
from roam_utils.trajectory import Trajectory
import joblib


class NumpyTrajectory(Trajectory):
    def __init__(self, traj_length=None, state_dim=None, action_dim=None):
        super(NumpyTrajectory, self).__init__(traj_length, state_dim, action_dim)
        if traj_length and state_dim and action_dim:
            self.X = np.zeros([traj_length, state_dim, 1])
            self.U = np.zeros([traj_length - 1, action_dim, 1])
            self.time_array = np.zeros([traj_length, 1])
            self.X[:] = np.nan
            self.U[:] = np.nan
            self.time_array[:] = np.nan

    def set_time_with_delta_t(self, delta_t):
        self.time_array = np.arange(0, len(self.X)*delta_t, delta_t)

    def set_t_idx(self, t, idx):
        self.time_array[idx] = t

    def set_T_idx(self, t, idx):
        self.time_array[idx] = t

    def set_X_idx(self, x, idx):
        self.X[idx] = x

    def set_U_idx(self, u, idx):
        self.U[idx] = u

    def get_T_copy(self):
        return self.time_array.copy()

    def get_X_copy(self):
        return self.X.copy()

    def get_U_copy(self):
        return self.U.copy()

    def get_t_copy(self, idx):
        return self.time_array[idx].copy()

    def get_x_copy(self, idx):
        return self.X[idx].copy()

    def get_u_copy(self, idx):
        return self.U[idx].copy()

    def save(self, save_dir, name=None, number=None):
        param_dict = {'X': self.X, 'U': self.U}
        if not np.all(np.isnan(self.time_array)):
            param_dict['time_array'] = self.time_array
        save_path = PathGenerator.get_trajectory_savepath(save_dir, ext='.sav', name=name, number=number)
        joblib.dump(param_dict, save_path)

    def save_text(self, save_dir, name=None, number=None):
        save_path = PathGenerator.get_trajectory_savepath(save_dir, ext='.txt', name=name, number=number)
        traj_length = len(self.X)
        with open(save_path, "w") as text_file:
            for i in np.arange(0, traj_length-1):
                text_file.write('i:{}, x:{}, u:{}\n'.format(i, self.X[i].squeeze().tolist(), self.U[i].squeeze().tolist()))
            i = traj_length-1
            text_file.write('i: {}, x: {}\n'.format(i, self.X[i].squeeze().tolist()))

    def load(self, load_dir, name=None, number=None):
        load_path = PathGenerator.get_trajectory_savepath(load_dir, '.sav', name=name, number=number)
        load_dict = joblib.load(load_path)
        self.set_X(load_dict['X'])
        self.set_U(load_dict['U'])
        self.set_traj_length(len(self.X))
        if 'time_array' in load_dict:
            self.set_T(load_dict['time_array'])
