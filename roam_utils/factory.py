import importlib
import ray

def ray_make(config_data, section_name):
    entrypoint = config_data.get(section_name, 'entrypoint')
    attr = get(entrypoint)
    return attr.remote(config_data, section_name)

def make(config_data, section_name):
    entrypoint = config_data.get(section_name, 'entrypoint')
    attr = get(entrypoint)
    return attr(config_data, section_name)


def get(entrypoint):
    module, name = entrypoint.split(':')
    # name = config_data.get(section_name, 'name')
    # module = config_data.get(section_name, 'module')
    module = importlib.import_module(module)
    attr = getattr(module, name)
    return attr