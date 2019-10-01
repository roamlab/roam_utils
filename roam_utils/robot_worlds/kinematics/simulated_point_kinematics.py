from __future__ import absolute_import, division, print_function, unicode_literals
from roam_learning.robot_worlds.kinematics.kinematics import Kinematics
import copy
import numpy as np


class SimulatedPointKinematics(Kinematics):
    def __init__(self):
        Kinematics.__init__(self)

    def setup_dispatch(self):
        Kinematics.setup_dispatch(self)
        self.compute_from_state_dispatch['position'] = self.get_p_from_state
        self.compute_from_state_dispatch['velocity'] = self.get_v_from_state

    def get_p_from_state(self, x):
        raise NotImplementedError

    def get_v_from_state(self, x):
        raise NotImplementedError

class Simulated1DPointKinematics(SimulatedPointKinematics):
    def __init__(self):
        SimulatedPointKinematics.__init__(self)

    def get_p_from_state(self, x):
        if len(x.shape) < 2:
            print('failed')
        px = copy.copy(x[0,0])
        py = 0.0
        return np.asarray([px, py]).reshape((-1, 1))

    def get_v_from_state(self, x):
        vx = copy.copy(x[1,0])
        vy = 0.0
        v = np.asarray([vx, vy]).reshape((-1, 1))
        return np.asarray([vx, vy]).reshape((-1, 1))

class Simulated2DPointKinematics(SimulatedPointKinematics):
    def __init__(self):
        SimulatedPointKinematics.__init__(self)

    def get_p_from_state(self, x):
        return copy.copy(x[:2]).reshape((-1, 1))

    def get_v_from_state(self, x):
        return copy.copy(x[2:4]).reshape((-1, 1))

    def get_p0_from_state(self, x):
        return copy.copy(x[self.get_p0_idxs()]).reshape((-1, 1))

    def get_p0_idxs(self):
        return np.arange(0, 2)
