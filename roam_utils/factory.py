import importlib
from class_registry import ClassRegistry

roam_utils_registry = ClassRegistry(attr_name='__name__', unique=True)


def roam_util_make(config_data, section_name):
    registry_dict = {'roam_utils': roam_utils_registry}
    return make(config_data, section_name, registry_dict)


def make(config_data, section_name, registry_dict=None):
    class_name = config_data.get(section_name, 'name')
    # first check to see if class is defined by module and class name in config file
    try:
        attr = get_attr(class_name, config_data, section_name)
        instance = attr(config_data, section_name)
    except ValueError:
        print('section: {} does not have a module item'.format(section_name))
        instance = None
    if instance is not None:
        return instance

    # if it is not found that way, check the registries
    if registry_dict is not None:
        for registry_name, registry in registry_dict:
            try:
                instance = registry.get(class_name, config_data, section_name)
            except:
                print('{} registry does not have class: {}'.format(registry_name, class_name))
                instance = None
            if instance is not None:
                return instance
    if instance is None:
        raise ValueError('class: {} not found using getattr or using registries'.format(class_name))


def get_attr(class_name, config_data, section_name):
    if config_data.has_option(section_name, 'module'):
        module_name = config_data.get(section_name, 'module')
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
        return class_(config_data, section_name)
    else:
        raise ValueError('no module given in section: {}'.format(section_name))