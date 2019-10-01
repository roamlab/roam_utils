from __future__ import absolute_import, division, print_function, unicode_literals
import time
import copy
import numpy as np
import torch
from roam_learning.helper_functions.jacobian_calculators.jacobian_calculator import JacobianCalculator


class NumpyJacobianCalculator(JacobianCalculator):
    def __init__(self, model=None, jacobian_type=None, jacobian_delta_x_default=None, jacobian_delta_u_default=None,
                 auto_delta=None, jacobian_ratio=None):
        JacobianCalculator.__init__(self, model, jacobian_type, jacobian_delta_x_default, jacobian_delta_u_default,
                 auto_delta, jacobian_ratio)
        self.data_type = np.ndarray

    def get_empty_jacobian(self, f_dim, x_dim, u_dim):
        dfdx = np.zeros((f_dim, x_dim))  # dfdx must be square so labels must be same shape as x
        dfdu = np.zeros((f_dim, u_dim))
        return dfdx, dfdu

    def get_empty_jacobian_deltas(self, x_dim, u_dim):
        jacobian_delta_x = np.zeros((x_dim, 1))
        jacobian_delta_u = np.zeros((u_dim, 1))
        return jacobian_delta_x, jacobian_delta_u

    def get_jacobians_deltas(self, x_cur, u_cur, auto_delta=False):
        jacobian_delta_x, jacobian_delta_u = self.get_empty_jacobian_deltas(len(x_cur), len(u_cur))
        if auto_delta:
            x_next = self.model.predict(x_cur, u_cur)
            state_check = (x_next - x_cur) < 1e-12
            jacobian_delta_x = (x_next - x_cur) / self.jacobian_ratio
            state_difference = abs(x_next - x_cur)

            for i in range(0, len(state_check)):
                if state_check[i]:
                    jacobian_delta_x[i] = self.jacobian_delta_x_default

            jacobian_delta_u = np.full(u_cur.shape, self.jacobian_delta_u_default)
            for i in range(0, len(u_cur)):
                while True:
                    u_cur_copy = copy.copy(u_cur)
                    u_cur_copy[i] = u_cur_copy[i] + jacobian_delta_u[i]
                    perturb_u_influence = abs(self.model.predict(x_cur, u_cur_copy) - self.model.predict(x_cur, u_cur))

                    ratio_check = np.multiply(self.jacobian_ratio, perturb_u_influence) <= state_difference
                    #ratio_check = np.divide(state_difference, perturb_u_influence) >= self.jacobian_ratio

                    if np.all(np.logical_or(state_check, ratio_check)):
                        break
                    jacobian_delta_u = jacobian_delta_u / self.jacobian_ratio
        else:
            jacobian_delta_x[:] = self.jacobian_delta_x_default
            jacobian_delta_u[:] = self.jacobian_delta_u_default

        return jacobian_delta_x, jacobian_delta_u

    def get_jacobians_numerical(self, x_cur, u_cur, history=None):
        jacobian_start_time = time.time()
        # delta_state = self.model.dynamics.delta_state
        jacobian_delta_x, jacobian_delta_u = self.get_jacobians_deltas(x_cur, u_cur, auto_delta=self.auto_delta)
        # must make copy of x_cur and u_cur before doing any operations on them
        # x_cur and u_cur are used later in __rollout and cannot be changed by this function
        dfdx, dfdu = self.get_empty_jacobian(len(x_cur), len(x_cur), len(u_cur))
        #dfdx = np.zeros((self.state_dim, self.state_dim))
        x_cur_copy = copy.copy(x_cur)
        for i in range(0, len(x_cur)):
            x_cur_copy[i] = x_cur_copy[i] + jacobian_delta_x[i]
            fp = self.model.predict(x_cur_copy, u_cur, history)
            x_cur_copy[i] = x_cur_copy[i] - 2 * jacobian_delta_x[i]
            fm = self.model.predict(x_cur_copy, u_cur, history)
            x_cur_copy[i] = x_cur_copy[i] + jacobian_delta_x[i]
            dfdx[:, i] = (self.delta_state(fp, fm) / (2 * jacobian_delta_x[i])).reshape(-1)
        #dfdu = np.zeros((self.state_dim, self.action_dim))
        u_cur_copy = copy.copy(u_cur)
        for i in range(0, len(u_cur)):
            u_cur_copy[i] = u_cur_copy[i] + jacobian_delta_u[i]
            fp = self.model.predict(x_cur, u_cur_copy, history)
            u_cur_copy[i] = u_cur_copy[i] - 2 * jacobian_delta_u[i]
            fm = self.model.predict(x_cur, u_cur_copy, history)
            u_cur_copy[i] = u_cur_copy[i] + jacobian_delta_u[i]
            dfdu[:, i] = (self.delta_state(fp, fm) / (2 * jacobian_delta_u[i])).reshape(-1)

        jacobian_end_time = time.time()
        jacobian_time = jacobian_end_time - jacobian_start_time

        return dfdx, dfdu

    def delta_state(self, state1, state2):
        state1 = self.model.convert(state1, self.model.dynamics.get_data_type())
        state2 = self.model.convert(state2, self.model.dynamics.get_data_type())
        delta_state = self.model.dynamics.delta_state(state1, state2)
        return self.model.convert(delta_state, self.model.get_data_type())

    def get_jacobians_analytical(self, x_cur, u_cur, history=None):
        x_cur = self.model.convert(x_cur, self.model.dynamics.get_data_type())
        u_cur = self.model.convert(u_cur, self.model.dynamics.get_data_type())
        x_cur.requires_grad_()
        u_cur.requires_grad_()
        x_new = x_cur
        for i in range(self.model.steps_per_prediction):
            x_new = self.model.dynamics.advance(x_new, u_cur)
        dfdx, dfdu = self.get_empty_jacobian(f_dim=len(x_new), x_dim=len(x_cur), u_dim=len(u_cur))
        for i in range(len(x_new)):
            mask = torch.zeros_like(x_new)  # mask shape = [1, len(x_new)]  ||  mask_shape = [1,1]
            mask[i, 0] = 1  # mask = [[0], ..., [1], ...,, [0]] || mask = [1]
            dfdx_i, dfdu_i = torch.autograd.grad([x_new], [x_cur, u_cur], grad_outputs=mask, retain_graph=True)
            # dfdx[i,:] = dfdx_1, dfdx_2, ... dfdx_label  ||  dfdx[0,:] = dfdx_label
            dfdx[i, :] = dfdx_i.detach().view((1, -1)).cpu().numpy()
            # dfdu[i,:] = dfdu_1, dfdu_2, dfdu_3, dfdu_4  ||  dfdu[0,:] = dfdu_label,
            dfdu[i, :] = dfdu_i.detach().view((1, -1)).cpu().numpy()
            # don't need to zero gradients because autograd.grad doesn't accumulate gradients in inputs .grad
        return dfdx, dfdu  # returns completed dfdx, dfdu || returns dfdx, dfdu empty except for [0,:]
