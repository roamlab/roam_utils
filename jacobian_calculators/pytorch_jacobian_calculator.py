from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import copy
import time
from roam_learning.helper_functions.jacobian_calculators.jacobian_calculator import JacobianCalculator
from roam_learning.helper_functions.operations_library_wrappers.pytorch_operations import PytorchOperations


class PytorchJacobianCalculator(JacobianCalculator):
    def __init__(self, model=None, jacobian_type=None, jacobian_delta_x_default=None, jacobian_delta_u_default=None,
                 auto_delta=None, jacobian_ratio=None):
        JacobianCalculator.__init__(self, model, jacobian_type, jacobian_delta_x_default, jacobian_delta_u_default,
                 auto_delta, jacobian_ratio)
        self.data_type = torch.Tensor
        self.torch = PytorchOperations()

    def initialize_from_config(self, config_data, section_name):
        JacobianCalculator.initialize_from_config(self, config_data, section_name)
        self.torch = PytorchOperations()
        self.torch.initialize_from_config(config_data, section_name='pytorch')

    def get_empty_jacobian(self, f_dim, x_dim, u_dim):
        """Creates empty torch tensors

        Args:
            f_dim: An integer specifying the output dimension (label dim) of this forward model
            x_dim: An integer specifying the state dim of the robot this forward model is trained for
            u_dim: An integer specifying the action dimension of the robot this forward model is trained for

        Returns:
            A tuple containing two empty torch tensors representing dfdx and dfdu

        """
        dfdx = self.torch.zeros((f_dim, x_dim))  #dfdx must be square so labels must be same shape # as x
        dfdu = self.torch.zeros((f_dim, u_dim))
        return dfdx, dfdu

    def get_empty_jacobian_deltas(self, x_dim, u_dim):
        jacobian_delta_x = self.torch.zeros((x_dim, 1))
        jacobian_delta_u = self.torch.zeros((u_dim, 1))
        return jacobian_delta_x, jacobian_delta_u

    def get_jacobians_deltas(self, x_cur, u_cur, auto_delta=False):
        jacobian_delta_x, jacobian_delta_u = self.get_empty_jacobian_deltas(len(x_cur), len(u_cur))
        if auto_delta:
            x_next = self.model.predict(x_cur, u_cur)
            state_check = (x_next - x_cur) == 0
            jacobian_delta_x = (x_next - x_cur) / self.jacobian_ratio
            state_difference = abs(x_next - x_cur)

            for i in range(0, len(state_check)):
                if state_check[i]:
                    jacobian_delta_x[i] = self.jacobian_delta_x_default

            jacobian_delta_u = self.torch.full(u_cur.shape, self.jacobian_delta_u_default)
            for i in range(0, len(u_cur)):
                while True:
                    u_cur_copy = copy.copy(u_cur)
                    u_cur_copy[i] = u_cur_copy[i] + jacobian_delta_u[i]
                    perturb_u_influence = abs(self.model.predict(x_cur, u_cur_copy) - self.model.predict(x_cur, u_cur))
                    ratio_check = torch.divide(state_difference, perturb_u_influence) >= self.jacobian_ratio

                    if torch.all(torch.logical_or(state_check, ratio_check)):
                        break
                    jacobian_delta_u = jacobian_delta_u / self.jacobian_ratio
        else:
            jacobian_delta_x = self.torch.full(x_cur.shape, self.jacobian_delta_x_default)
            jacobian_delta_u = self.torch.full(u_cur.shape, self.jacobian_delta_u_default)

        return jacobian_delta_x, jacobian_delta_u

    def get_jacobians_numerical(self, x_cur, u_cur, history=None, auto_delta=False):
        jacobian_delta_x, jacobian_delta_u = self.get_jacobians_deltas(x_cur, u_cur, auto_delta=auto_delta)
        # must make copy of x_cur and u_cur before doing any operations on them
        # x_cur and u_cur are used later in __rollout and cannot be changed by this function
        dfdx, dfdu = self.get_empty_jacobian(len(x_cur), len(x_cur), len(u_cur))
        #dfdx = np.zeros((self.state_dim, self.state_dim))
        x_cur_copy = x_cur.clone()
        for i in range(0, len(x_cur)):
            x_cur_copy[i] = x_cur_copy[i] + jacobian_delta_x[i]
            fp = self.model.predict(x_cur_copy, u_cur, history)
            x_cur_copy[i] = x_cur_copy[i] - 2 * jacobian_delta_x[i]
            fm = self.model.predict(x_cur_copy, u_cur, history)
            x_cur_copy[i] = x_cur_copy[i] + jacobian_delta_x[i]
            dfdx[:, i] = ((fp - fm) / (2 * jacobian_delta_x[i])).reshape(-1)
        #dfdu = np.zeros((self.state_dim, self.action_dim))
        u_cur_copy = u_cur.clone()
        for i in range(0, len(u_cur)):
            u_cur_copy[i] = u_cur_copy[i] + jacobian_delta_u[i]
            fp = self.model.predict(x_cur, u_cur_copy, history)
            u_cur_copy[i] = u_cur_copy[i] - 2 * jacobian_delta_u[i]
            fm = self.model.predict(x_cur, u_cur_copy, history)
            u_cur_copy[i] = u_cur_copy[i] + jacobian_delta_u[i]
            dfdu[:, i] = ((fp - fm) / (2 * jacobian_delta_u[i])).reshape(-1)

        return dfdx, dfdu

    def get_jacobians_analytical(self, x_cur, u_cur, history=None):
        """

        Args:
            x_cur: A 1d pytorch Tensor (requires_grad=True)
            u_cur: A 1d pytorch Tensor (requires_grad=True)
            history: A 2d pytorch Tensor (requires_grad=False)

        Returns:
            dfdx of shape (len(x_cur), len(x_cur)) and dfdu of shape (len(x_cur), len(u_cur))

        """
        assert x_cur.requires_grad, 'x_cur needs required_grad=True to calculate jacobian'
        assert u_cur.requires_grad, 'u_cur needs requires_grad=True to calculate jacobian'
        if history:
            assert not history.requires_grad, 'history does not need requires_grad. '

        x_new = self.model.predict(x_cur, u_cur, history)
        dfdx, dfdu = self.get_empty_jacobian(f_dim=len(x_new), x_dim=len(x_cur), u_dim=len(u_cur))
        for i in range(len(x_new)):
            mask = torch.zeros_like(x_new)  # mask shape = [1, len(x_new)]  ||  mask_shape = [1,1]
            mask[i, 0] = 1  # mask = [[0], ..., [1], ...,, [0]] || mask = [1]
            dfdx_i, dfdu_i = torch.autograd.grad([x_new], [x_cur, u_cur], grad_outputs=mask, retain_graph=True)
            # dfdx[i,:] = dfdx_1, dfdx_2, ... dfdx_label  ||  dfdx[0,:] = dfdx_label
            dfdx[i, :] = dfdx_i.detach().view((1, -1))
            # dfdu[i,:] = dfdu_1, dfdu_2, dfdu_3, dfdu_4  ||  dfdu[0,:] = dfdu_label,
            dfdu[i, :] = dfdu_i.detach().view((1, -1))
            # don't need to zero gradients because autograd.grad doesn't accumulate gradients in inputs .grad
        return dfdx, dfdu  #returns completed dfdx, dfdu || returns dfdx, dfdu empty except for [0,:]
