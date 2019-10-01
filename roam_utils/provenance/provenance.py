from __future__ import absolute_import, division, print_function, unicode_literals
import os
import shutil
from roam_utils.provenance.path_generator import PathGenerator


def save_svn(experiment_dir, experiment_no):
    # saves svn info into experiment folder
    os.system('svn info ' + os.path.abspath(os.path.join(os.getcwd(), os.pardir))
              + ' > ' + str(PathGenerator.get_svn_info_pathname(experiment_dir, experiment_no)))


def copy_config(config_path, new_dir):
    shutil.copy2(config_path, new_dir)


def save_config(config_data, experiment_dir, experiment_no):
    config_save_path = PathGenerator.get_config_pathname(experiment_dir, experiment_no)
    with open(config_save_path, 'w') as configfile:
        config_data.write(configfile)


def save_config_with_custom_name(config_data, experiment_dir, custom_name):
    config_save_path = PathGenerator.get_config_with_custom_name(experiment_dir, custom_name)
    with open(config_save_path, 'w') as configfile:
        config_data.write(configfile)