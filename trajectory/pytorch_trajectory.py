from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import numpy as np
from roam_learning.helper_functions.trajectory.trajectory import Trajectory
from roam_learning.pytorch_device import PytorchDevice


class PytorchTrajectory(Trajectory, PytorchDevice):
    def __init__(self, horizon, state_dim, action_dim, cost_object=None):
        Trajectory.__init__(self)
        PytorchDevice.__init__(self)
        self.X = torch.zeros([horizon, state_dim, 1], requires_grad=True).to(self.device)
        self.U = torch.zeros([horizon-1, action_dim, 1], requires_grad=True).to(self.device)
        self.cost_array = torch.zeros([horizon, 1], requires_grad=False)

    def initialize(self, horizon, state_dim, action_dim, cost_calculator=None):
        Trajectory.initialize(self, horizon, state_dim, action_dim, cost_calculator)
        PytorchDevice.initialize(self)

    def update_cost(self, x, u):
        """

        Args:
            idx:

        Returns:

        """
        step_cost = self.cost_object.get_l(x, u)
        self.cost += step_cost

    def set_X_idx(self, x, idx):
        """

        Args:
            x:
            idx:

        Returns:

        """
        self.X[idx] = x.data

    def set_U_idx(self, u, idx):
        self.U[idx] = u.data

    def get_cost_numpy(self):
        """

        Returns:

        """
        return self.cost

    def set_converged_flag(self, converged):
        self.converged = converged

    def set_improved_flag(self, improved):
        self.improved = improved

    def reset_cost(self):
        self.cost = self.cost_object.get_l(self.X[0], self.U[0])

    def reset_X(self):
        x0 = self.X[0]
        self.X = torch.zeros(self.X.shape)
        self.set_X_idx(x0, 0)

    def get_X_copy(self):
        return self.X.clone()

    def get_U_copy(self):
        return self.U.clone()

    def get_x_copy(self, t):
        return self.X[t].clone()

    def get_u_copy(self, t):
        return self.U[t].clone()

    def get_u_numpy(self, t):
        return self.U[t].clone().cpu().numpy()

    def set_U(self, U):
        self.U = U.requires_grad_(True)

    def update_U(self, updated_trajectory):
        """

        Args:
            updated_trajectory:

        Returns:

        """
        num_actions = self.horizon - 1
        self.U = self.U.zero_()
        self.U[0:num_actions - 1] = updated_trajectory.U[1:num_actions].clone()
        self.U[num_actions - 1] = self.U[num_actions - 2]

    def perturb_U(self):
        for i in range(self.horizon-1):
            sign = ((-1)**np.random.choice([1, 2]))
            percent = np.random.ranf()
            self.set_U_idx(self.U[i]*(1 + percent*sign), i)
        self.reset_cost()
        self.reset_X()





