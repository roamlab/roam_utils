from __future__ import absolute_import, division, print_function, unicode_literals
import inspect
from roam_utils.provenance.path_generator import PathGenerator
import logging
import multiprocessing


class IndentFormatter(logging.Formatter):
    def __init__( self, fmt=None, datefmt=None ):
        logging.Formatter.__init__(self, fmt, datefmt)
        self.baseline = len(inspect.stack())
    def format( self, rec ):
        stack = inspect.stack()
        rec.indent = '    '*(len(stack)-12)
        rec.function = stack[8][3]
        out = logging.Formatter.format(self, rec)
        del rec.indent; del rec.function
        return out


def setup_logger(name, path, level, formatter, mode=None):
    logger = logging.getLogger(name)
    if mode is None:
        filehandler = logging.FileHandler(path)
    else:
        filehandler = logging.FileHandler(path, mode=mode)
    filehandler.setLevel(level)
    filehandler.setFormatter(formatter)

    logger.addHandler(filehandler)
    print('logger.handlers', logger.handlers)


def setup_roam_logger(log_dir, experiment_no, base_level, detail_level, debug_level):
    name='ROAM'
    debug_filepath = PathGenerator.get_logger_debug_pathname(log_dir, experiment_no)
    base_filepath = PathGenerator.get_logger_base_pathname(log_dir, experiment_no)
    detailed_filepath = PathGenerator.get_logger_detail_pathname(log_dir, experiment_no)
    formatter = IndentFormatter("%(levelname)s:%(indent)s%(message)s")
    #if log_flag:
    setup_logger(name, base_filepath, base_level, formatter)
    setup_logger(name, detailed_filepath, detail_level, formatter)
    setup_logger(name, debug_filepath, debug_level, formatter)
    logging.getLogger(name).setLevel(debug_level)


def setup_cost_logger(log_dir, experiment_no, level):
    name='COST'
    filepath = PathGenerator.get_costlog_pathname(log_dir, experiment_no)
    formatter = IndentFormatter("%(message)s")
    #if log_flag:
    setup_logger(name, filepath, level, formatter, mode='w')
    logging.getLogger(name).setLevel(level)


def setup_success_logger(log_dir, experiment_no, level):
    name="SUCCESS"
    filepath = PathGenerator.get_successlog_pathname(log_dir, experiment_no)
    formatter = IndentFormatter("%(message)s")
    setup_logger(name, filepath, level, formatter, mode='w')
    logging.getLogger(name).setLevel(level)


def setup_state_action_logger(log_dir, experiment_no, level):
    name='STATEACTION'
    filepath = PathGenerator.get_state_action_logger_pathname(log_dir, experiment_no)
    formatter = IndentFormatter("%(message)s")
    # if log_flag:
    setup_logger(name, filepath, level, formatter)
    logging.getLogger(name).setLevel(level)


def setup_queue_logger(log_dir, level):
    name='QUEUE'
    filepath = PathGenerator.get_queue_logger_pathname(log_dir)
    formatter = IndentFormatter("%(levelname)s:%(indent)s%(message)s")
    #if log_flag:
    setup_logger(name, filepath, level, formatter)
    logging.getLogger(name).setLevel(level)


def setup_timer_logger(log_dir, experiment_no, level):
    name='TIMER'
    filepath = PathGenerator.get_logger_timer_pathname(log_dir, experiment_no)
    formatter = IndentFormatter("%(levelname)s:%(indent)s%(message)s")
    #if log_flag:
    setup_logger(name, filepath, level, formatter)
    logging.getLogger(name).setLevel(level)


def get_roam_logger():
    return logging.getLogger('ROAM')


def get_queue_logger():
    return logging.getLogger('QUEUE')


def get_cost_logger():
    return logging.getLogger('COST')


def get_state_action_logger():
    return logging.getLogger('STATEACTION')


def get_timer_logger():
    return logging.getLogger('TIMER')


def get_success_logger():
    return logging.getLogger("SUCCESS")


def get_roam_logger_parallel():
    return multiprocessing.get_logger('ROAM_PARALLEL')


def get_timer_logger_parallel():
    return multiprocessing.get_logger('TIMER_PARALLEL')


def setup_logger_parallel(name, path, level, formatter, mode = None):
    logger = multiprocessing.get_logger(name)
    if mode is None:
        filehandler = logging.FileHandler(path)
    else:
        filehandler = logging.FileHandler(path, mode = mode)
    filehandler.setLevel(level)
    filehandler.setFormatter(formatter)

    logger.addHandler(filehandler)
    print('logger.handlers', logger.handlers)


def setup_roam_logger_parallel(log_dir, experiment_no, base_level, detail_level):
    name = 'ROAM_PARALLEL'
    debug_filepath = PathGenerator.get_logger_debug_pathname(log_dir, experiment_no)
    base_filepath = PathGenerator.get_logger_base_pathname(log_dir, experiment_no)
    detailed_filepath = PathGenerator.get_logger_detail_pathname(log_dir, experiment_no)
    formatter = IndentFormatter("%(levelname)s:%(indent)s%(message)s")
    setup_logger_parallel(name, base_filepath, base_level, formatter)
    setup_logger_parallel(name, detailed_filepath, detail_level, formatter)
    setup_logger_parallel(name, debug_filepath, 10, formatter)
    multiprocessing.getLogger(name).setLevel(10)


def setup_timer_parallel(log_dir, experiment_no, level):
    name = 'TIMER_PARALLEL'
    filepath = PathGenerator.get_logger_timer_pathname(log_dir, experiment_no)
    formatter = IndentFormatter("%(levelname)s:%(indent)s%(message)s")
    setup_logger_parallel(name, filepath, level, formatter)
