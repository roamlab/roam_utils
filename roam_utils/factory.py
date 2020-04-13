import importlib


def make(config_data, section_name):
    entrypoint = config_data.get(section_name, 'entrypoint')
    module, name = entrypoint.split(':')
    attr = get(module, name)
    return attr(config_data, section_name)


def get(module, name):

    # name = config_data.get(section_name, 'name')
    # module = config_data.get(section_name, 'module')
    module = importlib.import_module(module)
    attr = getattr(module, name)
    return attr