import shutil
from roam_utils.provenance import PathGenerator


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