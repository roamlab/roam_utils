import ast
import shutil
from roam_utils.provenance import PathGenerator
from configparser import ConfigParser


def copy_section_from_old_config_to_new_config(old_config, new_config, section, rename_section=None, overwrite=False):
    if rename_section is not None:
        new_section_name = rename_section
    else:
        new_section_name = section
    if new_config.has_section(new_section_name):
        if overwrite:
            new_config.add_section(new_section_name)
    else:
        new_config.add_section(new_section_name)

def recursive_copy_section_from_old_config_to_new_config(config_from, config_to, section, rename_section=None):
    if rename_section is not None:
        new_section_name = rename_section
    else:
        new_section_name = section
    if not config_to.has_section(new_section_name):
        config_to.add_section(new_section_name)
    for option, value in config_from.items(section):
        config_to.set(new_section_name, option, value)
        if config_from.has_section(value):
            recursive_copy_section_from_old_config_to_new_config(config_from, config_to, value)


def recursive_find_option_value(config_data, start_section, option_name):
    options = list(config_data.items(start_section))
    if option_name not in options:
        for option, value in config_data.items(start_section):
            if config_data.has_section(value):
                recursive_find_option_value(config_data, value, option_name)
    else:
        return option_name, config_data.get(start_section, option_name)


def pull_from_config(params_to_pull, config_data, section_name):
    """ returns a dict of parameter names and values using the options dict argument which specifies
    the parameters names and type """
    assert isinstance(params_to_pull, dict), 'params_to_pull must be dict specifying the parameter names and data types'
    params = {}
    for param, param_type in params_to_pull.items():
        if config_data.has_option(section_name, param):
            if param_type == 'bool':
                params[param] = config_data.getboolean(section_name, param)
            elif param_type == 'int':
                params[param] = config_data.getint(section_name, param)
            elif param_type == 'float2int':
                params[param] = int(config_data.getfloat(section_name, param))
            elif param_type == 'float':
                params[param] = config_data.getfloat(section_name, param)
            elif param_type == 'list':
                params[param] = ast.literal_eval(config_data.getint(section_name, param))
            elif param_type == 'eval':
                params[param] = eval(config_data.get(section_name, param))
            elif param_type == 'str':
                params[param] = config_data.get(section_name, param)
    return params


def get_section_config_data(config_data, section_name):
    section_config_data = ConfigParser()
    recursive_copy_section_from_old_config_to_new_config(config_data, section_config_data, section_name)
    return section_config_data


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
