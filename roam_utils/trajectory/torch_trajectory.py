import torch
import numpy as np
from roam_utils.trajectory import Trajectory


class TorchTrajectory(Trajectory):
    def __init__(self, traj_length=None, state_dim=None, action_dim=None, torch_device=None):
        Trajectory.__init__(traj_length, state_dim, action_dim)
        if traj_length and state_dim and action_dim and torch_device:
            self.torch_device = torch_device
            tensor_float_64 = torch.empty(2, dtype=torch.float64, device=torch_device.device)
            self.X = tensor_float_64.new_full([traj_length, state_dim, 1], fill_value=np.nan)
            self.U = tensor_float_64.new_full([traj_length - 1, action_dim, 1], fill_value=np.nan)
            self.time_array = tensor_float_64.new_full([traj_length, 1], fill_value=np.nan)

    # def update_cost(self, x, u, x_pre=None):
    #     """
    #
    #     Args:
    #         idx:
    #
    #     Returns:
    #
    #     """
    #     step_cost = self.cost_calculator.get_l(x, u)
    #     self.cost += step_cost

    def set_X_idx(self, x, idx):
        self.X[idx] = x

    def set_U_idx(self, u, idx):
        self.U[idx] = u

    def get_X_copy(self):
        return self.X.clone()

    def get_U_copy(self):
        return self.U.clone()

    def get_t_copy(self, idx):
        return self.time_array[idx].clone()

    def get_x_copy(self, idx):
        return self.X[idx].clone()

    def get_u_copy(self, idx):
        return self.U[idx].clone()

    def get_u_numpy(self, idx):
        return self.U[idx].clone().cpu().numpy()

    # def set_U(self, U):
    #     self.U = U.requires_grad_(True)
    #
    # def update_U(self, updated_trajectory):
    #     """
    #
    #     Args:
    #         updated_trajectory:
    #
    #     Returns:
    #
    #     """
    #     num_actions = self.horizon - 1
    #     self.U = self.U.zero_()
    #     self.U[0:num_actions - 1] = updated_trajectory.U[1:num_actions].clone()
    #     self.U[num_actions - 1] = self.U[num_actions - 2]





