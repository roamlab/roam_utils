from __future__ import absolute_import, division, print_function, unicode_literals
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
from roam_learning.path_generator import PathGenerator
import matplotlib.cm as cm
# not compatible with matplotlib-2.2.3, matplotlib-2.0.2 required
import math
import numpy as np
import time
import threading
import ast
import roam_learning.script_helpers.matplotlib_helpers as plt_helpers


def rot(theta):
    R = np.zeros((2, 2))
    R[0, 0] = math.cos(theta)
    R[0, 1] = -math.sin(theta)
    R[1, 0] = math.sin(theta)
    R[1, 1] = math.cos(theta)
    return R


class RenderGUI(object):
    def __init__(self, render_rate=None, record_fps=None, record_sim=None, time_frame=None, lim_x=None, lim_y=None,
                 subject=None, obstacle_list=None, color_list=None, transparency=1.0, joint_dot=False,
                 fading_transparency=False, add_subject_legend=None, separate_axis=None, scale=None, tick_font_size=20,
                 legend_flag=True, title="title", title_fontsize=40, subplot_title_fontsize=30,
                 x_axis_label="x_axis_label", y_axis_label='y_axis_label', axis_label_fontsize=20, figsize_x=10, figsize_y=10):
        self.subject_dict = {}
        if subject:
            self.add_subject(subject)
        self.start_time = time.time()
        self.figure_created = False

        # recordingZ
        self.render_dir = ''
        self.prev_frame_time = 0
        self.frame_count = 0

        self.point_dict = {}
        self.axis_dict = {}
        self.vel_dir = None

        self.rate = render_rate
        self.record_fps = record_fps
        self.record_sim = record_sim
        self.time_frame = time_frame
        self.lim_x = lim_x
        self.lim_y = lim_y
        self.color_list = color_list
        self.obstacle_list = obstacle_list
        self.transparency = transparency
        self.joint_dot = joint_dot
        self.fading_transparency = fading_transparency
        self.add_subject_legend = add_subject_legend
        self.separate_axis = separate_axis
        self.scale = scale
        self.tick_font_size = tick_font_size
        self.legend_flag = legend_flag
        self.title = title
        self.title_fontsize = title_fontsize
        self.subplot_title_fontsize = subplot_title_fontsize
        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        self.axis_label_fontsize = axis_label_fontsize
        self.figsize_x = figsize_x
        self.figsize_y = figsize_y

    def initialize_from_config(self, config_data, section_name):
        self.rate = config_data.getfloat(section_name, 'render_rate')
        self.record_fps = config_data.getfloat(section_name, 'record_fps')
        self.record_sim = config_data.getboolean(section_name, 'record_sim')
        self.time_frame = config_data.get(section_name, 'time_frame')
        self.lim_x = [float(x) for x in ast.literal_eval(config_data.get(section_name, 'lim_x'))]
        self.lim_y = [float(x) for x in ast.literal_eval(config_data.get(section_name, 'lim_y'))]
        if config_data.has_option(section_name, 'transparency'):
            self.transparency = config_data.getfloat(section_name, 'transparency')
        else:
            self.transparency = 1
        if config_data.has_option(section_name, 'joint_dot'):
            self.joint_dot = config_data.getboolean(section_name, 'joint_dot')
        else:
            self.joint_dot = False
        self.fading_transparency = False
        if config_data.has_option(section_name, 'subject_legend'):
            self.add_subject_legend = config_data.getboolean(section_name, 'subject_legend')
        else:
            self.add_subject_legend = True

        if config_data.has_option(section_name, 'separate_axis'):
            self.separate_axis = config_data.getboolean(section_name, 'separate_axis')
        else:
            self.separate_axis = False
        self.scale = 2
        if config_data.has_option(section_name, 'scale'):
            self.scale = config_data.getfloat(section_name, 'scale')
        if config_data.has_option(section_name, 'tick_font_size'):
            self.tick_font_size = config_data.getfloat(section_name, 'tick_font_size')
        self.legend_flag = False
        if config_data.has_option(section_name, 'legend_flag'):
            self.legend_flag = config_data.getboolean(section_name, 'legend_flag')
        if config_data.has_option(section_name, 'title'):
            self.title = config_data.get(section_name, 'title')
        if config_data.has_option(section_name, 'title_fontsize'):
            self.title_fontsize = config_data.getfloat(section_name, 'title_fontsize')
        if config_data.has_option(section_name, 'subplot_title_fontsize'):
            self.subplot_title_fontsize = config_data.getfloat(section_name, 'subplot_title_fontsize')
        if config_data.has_option(section_name, 'x_axis_label'):
            self.x_axis_label = config_data.get(section_name, 'x_axis_label')
        if config_data.has_option(section_name, 'y_axis_label'):
            self.y_axis_label = config_data.get(section_name, 'y_axis_label')
        if config_data.has_option(section_name, 'axis_label_fontsize'):
            self.axis_label_fontsize = config_data.getfloat(section_name, 'axis_label_fontsize')
        if config_data.has_option(section_name, 'figsize_x'):
            self.figsize_x = config_data.getint(section_name, 'figsize_x')
        if config_data.has_option(section_name, 'figsize_y'):
            self.figsize_y = config_data.getint(section_name, 'figsize_y')

    def set_fading_transparency(self, fading_rate, fading_transparency, number_of_frames=None):
        if fading_transparency is True:
            self.transparency = self.transparency * (fading_rate ** (number_of_frames + 1))
            self.fading_transparency = True
            self.fading_rate = 1 / fading_rate
        else:
            self.fading_transparency = False

    def add_subject(self, subject, name=None):
        if name is None:
            name = len(self.subject_dict)
        self.subject_dict[name] = subject

    def setup_figure(self):
        if self.figure_created is False:
            #w, h = self.lim_x[1] - self.lim_x[0] + 0.25, self.lim_y[1] - self.lim_y[0] + 0.25
            self._fig, axis_array = plt.subplots(nrows=len(self.subject_dict), ncols=1, sharex=False, figsize=(self.figsize_x, self.figsize_y))
            i = 0

            for name, subject in self.subject_dict.items():
                if len(self.subject_dict) ==1:
                    axis = axis_array
                else:
                    axis = axis_array[i]
                axis.set_ylabel(self.y_axis_label, fontsize=self.axis_label_fontsize)
                axis.set_xlabel(self.x_axis_label, fontsize=self.axis_label_fontsize)
                axis.xaxis.set_tick_params(which='both', labelbottom=True, labeltop=False)
                # if i == len(self.subject_dict)-1:
                #     axis.xaxis.set_tick_params(which='both', labelbottom=True, labeltop=False)
                # else:
                #     axis.xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
                axis.xaxis.offsetText.set_visible(False)

                axis.set_autoscale_on(False)
                plt_helpers.set_axis_ticksize(axis, self.tick_font_size)
                #Zplt_helpers.set_axis_xy_lim(axis, self.lim_x, self.lim_y)
                axis.set_title(name, fontsize=self.subplot_title_fontsize)
                self.axis_dict[name] = axis

                i = i+1
            self.figure_created = True
            self._fig.canvas.draw()
            plt.show(block=False)
        plt.figure(self._fig.number)

    def set_color_list(self, color_list):
        assert len(color_list) == len(self.subject_dict), 'color list is not same length as subject dict'
        self.color_list = color_list

    def render(self):
        self.setup_figure()
        for name, ax in self.axis_dict.items():
            ax.clear()
        lines = []
        labels = []
        color_idx = 0
        if self.color_list is None:
            self.color_list = cm.viridis(np.linspace(0, 1, len(self.subject_dict)))
        for name, subject in self.subject_dict.items():
            axis = self.axis_dict[name]
            self.render_frame(name, color=self.color_list[color_idx])
            labels.append(name)
            line, = axis.plot([0], [0], color=self.color_list[color_idx], lw=4)
            lines.append(line)
            color_idx += 1
            if self.add_subject_legend:
                self.add_legend(lines, labels, axis, loc='upper left')
                pass

            if self.point_dict is not None:
                lines, labels = self.plot_point_dict(self.point_dict, axis)
                self.add_legend(lines, labels, axis, loc='upper right')

            self.add_time_info(axis)

        self._fig.canvas.draw()

    def plot_point_dict(self, point_dict, axis):
        lines = []
        labels = []
        for label, point in point_dict.items():
            labels.append(label)
            line, = axis.plot(point[0], point[1], marker='+', color='g')
            lines.append(line)
        return lines, labels

    def add_legend(self, lines, labels, axis, loc):
        legend = plt.legend(lines, labels, loc=loc)
        axis.add_artist(legend)

    def plot_arrow(self, pos):
        plt.arrow(x=2 * pos[0], y=2 * pos[1], dx=0.1 * pos[0], dy=0.1 * pos[1], color='r', width=0.01)

    def render_frame(self, dict_name, color='b'):
        axis = self.axis_dict[dict_name]
        if self.obstacle_list is not None:
            for (ox, oy, size) in self.obstacle_list:
                circle = plt.Circle((ox, oy), size, color='blue')
                axis.add_artist(circle)

    def save_frame_based_on_fps(self, save_path):
        current_frame_time = self.get_time()

        if (current_frame_time - self.prev_frame_time) > (1.0 / self.record_fps):
            self.save_frame(save_path)
            self.prev_frame_time = current_frame_time
            self.frame_count += 1
            # print('saving frame..')

    def save_frame(self, save_path):
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path)

    def loop(self):
        prev_time = time.time()
        while True:
            self.render()
            curr_time = time.time()
            if (curr_time - prev_time) < self.rate:
                time.sleep(self.rate - (curr_time - prev_time))
            prev_time = curr_time
            if self.record_sim is True:
                save_path = PathGenerator.get_gui_render_path(self.render_dir, self.frame_count)
                self.save_frame_based_on_fps(save_path)

    def start(self):
        # Cannot actually run gui in separate thread
        self.my_thread = threading.Thread(target=self.loop)
        self.my_thread.daemon = True
        self.my_thread.start()

    def set_render_dir(self, render_dir):
        self.render_dir = render_dir

    def set_point_dict(self, point_dict):
        self.point_dict = point_dict

    def add_to_point_dict(self, point_dict):
        self.point_dict.update(point_dict)

    def set_goal(self, goal):
        if self.point_dict is None:
            self.point_dict = {}
        self.point_dict['goal'] = goal

    def show_vel_dir(self, heading_angle):
        heading_angle = math.radians(heading_angle)
        self.vel_dir = [math.cos(heading_angle), math.sin(heading_angle)]

    def reset(self):
        self.prev_frame_time = 0.0

    def add_time_info(self, axis):
        time_str = ('model : ' + str(round(self.subject.get_time(), 3)) + 's\n'
                    + '    cpu : ' + str(round(time.time() - self.start_time, 1)) + 's')
        plt.text(0.825, 0.05, time_str, ha='left', va='top', transform=axis.transAxes)

    def get_time(self):
        if self.time_frame == 'robot':
            return self.subject.get_time()
        elif self.time_frame == 'cpu':
            return time.time()

    def set_obstacle_list(self, obstacle_list):
        self.obstacle_list = obstacle_list

    # subject property method is just for allowing convenient access to one of the subjects
    # primarily used for getting time
    @property
    def subject(self):
        try:
            return self.subject_dict.values()[0]
        except TypeError:
            return list(self.subject_dict.values())[0]
