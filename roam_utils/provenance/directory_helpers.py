import os


def make_dir(path):
    # recursive function to create directory
    if not os.path.exists(path):
        # try:
        #     head, tail = os.path.split(path)
        # except:
        #     raise ValueError('the directory you are trying to create is impossible')
        #
        # while not os.path.exists(head):
        #     make_dir(head)
        os.makedirs(os.path.join(path)) #, exist_ok=True) ## for python 3
    return path


def get_max_dirno(path, keyword):
    if not os.path.exists(path):
        return None
    subdirs = [subdirs for subdirs in os.listdir(path) if os.path.isdir(os.path.join(path, subdirs))]
    epoch_subdirs = [subdir for subdir in subdirs if keyword in subdir]
    epoch_nos = [int(epoch_subdir.split('_')[-1]) for epoch_subdir in epoch_subdirs]
    if epoch_nos:
        return max(epoch_nos)
    return None


def get_max_fileno(path, keyword):
    if not os.path.exists(path):
        return None
    filenames = [filename for filename in os.listdir(path) if os.path.isfile(os.path.join(path, filename))]
    node_list_files = [filename for filename in filenames if keyword in filename]
    filenos = [int((node_list_file.split('_')[-1]).split('.')[0]) for node_list_file in node_list_files]
    if filenos:
        return max(filenos)
    return None


def get_file_of_specific_extension_from_dir(any_dir, ext):
    num_files_with_ext = 0
    file_path = None
    for file in os.listdir(any_dir):
        if ext in file:
            file_path = os.path.join(any_dir, file)
            num_files_with_ext += 1
    if num_files_with_ext > 1:
        raise ValueError('dir: {} contains more than one file with ext: {}, '
                         'no convention for choosing which one'.format(any_dir, ext))
    if file_path is None:
        raise ValueError('no file with ext: {} found in dir: {}. '
                         'The path being returned is none, must be looking in the wrong directory'.format(ext, any_dir))
    return file_path
