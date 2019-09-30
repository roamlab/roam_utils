from __future__ import absolute_import, division, print_function, unicode_literals
import ast
import numpy as np
from roam_learning.helper_functions.trajectory.numpy_trajectory import NumpyTrajectory
from roam_learning.path_generator import PathGenerator


def get_trajectory_from_path(path):
    # path is from start to goal
    path_len = len(path)
    state_dim = path[0].state.shape[0]
    action_dim = path[0].action.shape[0]
    X = np.zeros((path_len, state_dim, 1))
    U = np.zeros((path_len-1, action_dim, 1))
    # start node has no action because it has no parent
    X[0] = path[0].state
    for i in np.arange(1, path_len):
        node = path[i]
        X[i] = node.state
        U[i-1] = node.action
    X = np.asarray(X)
    U = np.asarray(U)
    trajectory = NumpyTrajectory()
    trajectory.init_from_X_U(X, U)
    return trajectory


def load_trajectory_from_log(filepath):
    file = open(filepath, 'r')
    U = []
    X = []
    count = 0
    for line in file:
        if 'x' in line:
            x = np.asarray([float(x) for x in ast.literal_eval(line.split(':')[1])])
            X.append(x)
        if 'u' in line:
            u = np.asarray([float(ui) for ui in ast.literal_eval(line.split(':')[1])]).reshape(1, -1)
            U.append(u)
        count += 1
    U = np.asarray(U)
    X = np.asarray(X)
    trajectory = NumpyTrajectory()
    trajectory.init_from_X_U(X, U)
    return trajectory


def get_trajectory_from_action_sequence(x0, U, model):
    horizon = len(U)+1
    state_dim = x0.shape[0]
    X = np.zeros((horizon, state_dim, 1))
    cur_state = x0
    for i in np.arange(0, horizon-1):
        action = U[i]
        X[i] = cur_state.reshape((-1,1))
        if not isinstance(cur_state, model.data_type):
            cur_state, = model.convert_from_numpy([cur_state.reshape(-1, 1)])
        if not isinstance(action, model.data_type):
            action, = model.convert_from_numpy([action.reshape(-1,1)])

        next_state = model.predict(cur_state, action)

        if not isinstance(next_state, np.ndarray):
            next_state, = model.convert_to_numpy([next_state])
        next_state = next_state
        cur_state = next_state
    X[horizon-1] = cur_state
    trajectory = NumpyTrajectory()
    trajectory.init_from_X_U(X, U)
    return trajectory


def render_trajectory_dict(trajectory_dict, save_dir, gui):

    horizon = [0]*len(gui.subject_dict)
    for s in range(len(gui.subject_dict)):
        horizon[s] = len(trajectory_dict[s].U + 1)
    for h in range(np.max(horizon)):
        for s in range(len(gui.subject_dict)):
            if h < horizon[s]:
                state = trajectory_dict[s].get_x_copy(h)
                if state is not None:
                    gui.subject_dict[s].set_state(state)
                else:
                    state = trajectory_dict[s].get_x_copy(h-1)
                    gui.subject_dict[s].set_state(state)
            else:
                state = trajectory_dict[s].get_x_copy(horizon[s])
                gui.subject_dict[s].set_state(state)
        gui.render()
        save_path = PathGenerator.get_gui_render_path(save_dir, h)
        gui.save_frame(save_path)


def render_trajectory_dict_live(trajectory_dict, save_dir, gui):

    horizon = [0]*len(gui.subject_dict)
    for s in range(len(gui.subject_dict)):
        horizon[s] = len(trajectory_dict[s].U + 1)
        x0 = trajectory_dict[s].get_x_copy(0)
        gui.subject_dict[s].set_state(x0)
    gui.render()

    for h in range(np.max(horizon)):
        for s in range(len(gui.subject_dict)):
            if h < horizon[s]:
                U = trajectory_dict[s].get_U_copy()
                if U is not None:
                    gui.subject_dict[s].take_action(U[h])
                else:
                    state = trajectory_dict[s].get_x_copy(h-1)
                    gui.subject_dict[s].set_state(state)
            else:
                state = trajectory_dict[s].get_x_copy(horizon[s])
                gui.subject_dict[s].set_state(state)
        gui.render()
        save_path = PathGenerator.get_gui_render_path(save_dir, h)
        gui.save_frame(save_path)



def render_trajectory(trajectory, save_dir, gui):
    state_sequence = trajectory.get_X_copy()
    for i in range(len(state_sequence)):
        gui.subject.set_state(state_sequence[i])
        gui.render()
        save_path = PathGenerator.get_gui_render_path(save_dir, i)
        gui.save_frame(save_path)


def render_trajectory_live(trajectory, save_dir, gui):
    x0 = trajectory.get_x_copy(0)
    U = trajectory.get_U_copy()
    gui.subject.set_state(x0)
    #time.sleep(.001)
    #print('len_u'), len(U)
    for i in range(len(U)):
        gui.subject.take_action(U[i])
        gui.render()
        save_path = PathGenerator.get_gui_render_path(save_dir, i)
        gui.save_frame(save_path)





