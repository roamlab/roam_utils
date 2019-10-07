import os
import time
from datetime import datetime
import roam_utils.provenance.directory_helpers as dh

class PathGenerator(object):
    def __init__(self):
        pass

    ##################################################
    ################### GENERAL ######################
    ##################################################
    @staticmethod
    def check_path_existance(path):
        if os.path.exists(path):
            raise ValueError('save path {} already exists, cannot overwrite!'.format(path))

    @staticmethod
    def get_max_dirno(path, keyword):
        return dh.get_max_dirno(path, keyword)

    @staticmethod
    def get_max_fileno(path, keyword):
        return dh.get_max_fileno(path, keyword)

    @staticmethod
    def get_save_dir(experiment_dir):
        # returns save directory for an experiment
        save_dir = os.path.join(experiment_dir, 'save')
        dh.make_dir(save_dir)
        return save_dir

    @staticmethod
    def get_loss_plot_pathname(plot_dir):
        return os.path.join(plot_dir, 'loss_plot.png')

    @staticmethod
    def get_experiment_dir(path_to_experiment_dir, robot_name, experiment_type, experiment_no):
        experiment_dir = os.path.join(path_to_experiment_dir, 'experiments', robot_name, experiment_type,
                                      'experiment_{}'.format(str(experiment_no).zfill(3)))
        dh.make_dir(experiment_dir)
        return experiment_dir

    @staticmethod
    def make_dir(path):
        return dh.make_dir(path)

    @staticmethod
    def get_custom_path(directory, name, ext):
        return os.path.join(directory, name+ext)

    @staticmethod
    def get_plot_dir(experiment_dir, plot_no=None):
        plot_dir = os.path.join(experiment_dir, 'plots')
        if plot_no is None:
            plot_no = dh.get_max_dirno(plot_dir, 'plot')
            if plot_no is None: plot_no = 0
            plot_no += 1
        plot_dir = os.path.join(plot_dir, 'plot_'+str(plot_no).zfill(5))
        dh.make_dir(plot_dir)
        return plot_dir

    @staticmethod
    def get_dft_plot_dir(experiment_dir):
        directory = os.path.join(experiment_dir, 'dft_analysis_plots')
        dh.make_dir(directory)
        return directory
    ##################################################
    ################### CONFIGS ######################
    ##################################################

    @staticmethod
    def get_config_from_dir(any_dir):
        return dh.get_file_of_specific_extension_from_dir(any_dir, '.cfg')

    @staticmethod
    def get_svg_from_dir(any_dir):
        return dh.get_file_of_specific_extension_from_dir(any_dir, '.svg')

    @staticmethod
    def get_png_from_dir(any_dir):
        return dh.get_file_of_specific_extension_from_dir(any_dir, '.png')

    @staticmethod
    def get_config_pathname(load_dir, experiment_no):
        return os.path.join(load_dir, 'config_'+str(experiment_no).zfill(2)+'.cfg')

    @staticmethod
    def get_config_with_custom_name(load_dir, name):
        return os.path.join(load_dir, name+'.cfg')


    ##################################################
    ################## FORWARD MODEL #################
    ##################################################
    @staticmethod
    def get_predictor_experiment_dir(path_to_experiment_dir, robot_name, direction, algorithm_name, predictor_no=None):
        # returns experiment directory path
        algorithm_dir = os.path.join(path_to_experiment_dir, 'experiments', robot_name,
                                     '{}_predictors'.format(direction), algorithm_name)
        if predictor_no is None:
            max_dir_no = dh.get_max_dirno(algorithm_dir, 'predictor')
            if max_dir_no is None: max_dir_no = 0
            predictor_no = max_dir_no+1
        experiment_dir = os.path.join(algorithm_dir,'predictor_' + str(predictor_no).zfill(2))
        dh.make_dir(experiment_dir)
        return experiment_dir

    ##################################################
    ################### TRAJECTORY ###################
    ##################################################
    @staticmethod
    def get_trajectory_optimizer_dir(experiment_dir):
        traj_dir = os.path.join(experiment_dir, 'trajectory_optimizer_save')
        dh.make_dir(traj_dir)
        return traj_dir

    @staticmethod
    def get_trajectory_savepath(save_dir, ext='.sav', number=None, name=None):
        filename = 'trajectory_'+str(name) if (name is not None) else 'trajectory'
        filename = filename+'_'+str(number).zfill(5) if (number is not None) else filename
        filename = filename+ext
        return os.path.join(save_dir, filename)

    @staticmethod
    def get_record_sim_dir(experiment_dir):
        record_sim_dir = os.path.join(experiment_dir, 'pngs')
        dh.make_dir(record_sim_dir)
        return record_sim_dir

    @staticmethod
    def get_serpenoid_curve_experiment_dir(experiments_dir, robot_name, experiment_no=None, experiment_name='experiment'):
        experiments_dir = os.path.join(experiments_dir, 'experiments', robot_name, 'serpenoid_curve')
        dh.make_dir(experiments_dir)
        if experiment_no is None:
            experiment_no = dh.get_max_dirno(experiments_dir, experiment_name+'_')
            if experiment_no is None: experiment_no = 0
            experiment_no += 1
        experiments_dir = os.path.join(experiments_dir, experiment_name + '_' + str(experiment_no).zfill(5))
        dh.make_dir(experiments_dir)
        return experiments_dir

    @staticmethod
    def get_trajectory_dir(experiment_dir, trajectory_no=None, trajectory_name=None):
        trajectory_dir = os.path.join(experiment_dir, 'trajectories')
        trajectory_dir_name = 'trajectory_'+str(trajectory_name) if (trajectory_name is not None) else 'trajectory'
        if trajectory_no is not None:
            trajectory_dir_name = trajectory_dir_name + '_'+str(trajectory_no).zfill(5)
        trajectory_dir = os.path.join(trajectory_dir, trajectory_dir_name)
        dh.make_dir(trajectory_dir)
        return trajectory_dir

    @staticmethod
    def get_trajectory_filepaths_from_dir(trajectory_dir):
        if not os.path.exists(trajectory_dir):
            return None
        filenames = [filename for filename in os.listdir(trajectory_dir) if os.path.isfile(os.path.join(trajectory_dir, filename))]
        trajectory_filenames = [filename for filename in filenames if (('trajectory' in filename) and ('.sav' in filename))]
        trajectory_filepaths = [os.path.join(trajectory_dir, filename) for filename in trajectory_filenames]
        return trajectory_filepaths

    @staticmethod
    def get_trajectory_name_and_no_from_filepath(trajectory_filepath):
        trajectory_filename = os.path.basename(trajectory_filepath)
        trajectory_filename_without_ext = trajectory_filename.split('.')[0]
        trajectory_filename_split = trajectory_filename_without_ext.split('_')
        if len(trajectory_filename_split) == 3:
            trajectory_name = trajectory_filename_split[1]
            trajectory_no = int(trajectory_filename_split[2])
        elif len(trajectory_filename_split) == 2:
            try:
                trajectory_no = int(trajectory_filename_split[1])
                trajectory_name = None
            except:
                trajectory_no = None
                trajectory_name = trajectory_filename_split[1]
        else:
            trajectory_no = None
            trajectory_name = None
        return trajectory_name, trajectory_no

    @staticmethod
    def get_new_trajectory_dir(experiment_dir, trajectory_name=None):
        trajectory_base_dir = os.path.join(experiment_dir, 'trajectories')
        trajectory_dir_name = 'trajectory_' + str(trajectory_name) if (trajectory_name is not None) else 'trajectory'
        dh.make_dir(trajectory_base_dir)
        numstr = dh.get_max_dirno(trajectory_base_dir, trajectory_dir_name)
        if numstr is None: numstr = 0
        numstr += 1
        trajectory_dir_name = trajectory_dir_name+'_'+str(numstr).zfill(5)
        new_dir = os.path.join(trajectory_base_dir, trajectory_dir_name)
        dh.make_dir(new_dir)
        return new_dir

    @staticmethod
    def get_icra_trajectory_dir(experiment_dir, trajectory_name=None, icra_section_no=None):
        trajectory_base_dir = os.path.join(experiment_dir, 'trajectories')
        trajectory_dir_name = 'icra_section'
        trajectory_dir_name = trajectory_dir_name+'_'+str(icra_section_no).zfill(5)
        trajectory_dir_name = os.path.join(trajectory_dir_name, 'trajectory_'+str(trajectory_name).zfill(5))
        # trajectory_dir_name = os.path.join(trajectory_dir_name, 'trajectory_'+str(trajectory_name).zfill(5)+'.sav')
        new_dir = os.path.join(trajectory_base_dir, trajectory_dir_name)

        return new_dir

    ##################################################
    #################  LOGGING  ######################
    ##################################################
    @staticmethod
    def get_logger_debug_pathname(log_dir, experiment_no):
        return os.path.join(log_dir, 'log_'+str(experiment_no).zfill(2)+'_debug.txt')

    @staticmethod
    def get_logger_detail_pathname(experiment_dir, experiment_no):
        return os.path.join(experiment_dir, 'log_'+ str(experiment_no).zfill(2)+'_detail.txt')

    @staticmethod
    def get_logger_base_pathname(log_dir, experiment_no):
        return os.path.join(log_dir, 'log_'+ str(experiment_no).zfill(2)+'_base.txt')

    @staticmethod
    def get_logger_timer_pathname(log_dir, experiment_no):
        return os.path.join(log_dir, 'log_'+str(experiment_no).zfill(2)+'_timer.txt')

    @staticmethod
    def get_queue_logger_pathname(queue_dir):
        return os.path.join(queue_dir, 'configs_queue_log.txt')

    @staticmethod
    def get_costlog_pathname(experiment_dir, experiment_no):
        return os.path.join(experiment_dir, 'costlog_'+ str(experiment_no).zfill(2) + '.txt')

    @staticmethod
    def get_successlog_pathname(experiment_dir, experiment_no):
        return os.path.join(experiment_dir, 'successlog_' + str(experiment_no).zfill(2) + '.txt')

    @staticmethod
    def get_state_action_logger_pathname(experiment_dir, experiment_no):
        return os.path.join(experiment_dir, 'state_action_log_'+str(experiment_no).zfill(2)+'.txt')

    @staticmethod
    def get_svn_info_pathname(experiment_dir, experiment_no):
        return os.path.join(experiment_dir, 'svn_info_'+ str(experiment_no).zfill(2) + '.txt')

    @staticmethod
    def get_train_loss_csv_pathname(experiment_dir, experiment_no):
        return os.path.join(experiment_dir, 'train_loss_' + str(experiment_no).zfill(2) + '.csv')

    @staticmethod
    def get_val_loss_csv_pathname(experiment_dir, experiment_no):
        return os.path.join(experiment_dir, 'val_loss_' + str(experiment_no).zfill(2) + '.csv')

    @staticmethod
    def get_test_loss_csv_pathname(experiment_dir, experiment_no):
        return os.path.join(experiment_dir, 'test_loss_' + str(experiment_no).zfill(2) + '.csv')

    ##################################################
    ################## PLOTTING ######################
    ##################################################
    @staticmethod
    def get_new_plot_dir(experiment_dir, use_timestamp_in_path = True):
        plot_dir = os.path.join(experiment_dir, 'plots')
        if (use_timestamp_in_path):
            timestr = time.strftime("%Y%m%d-%H%M%S")
            plot_dir = os.path.join(plot_dir, timestr)
        dh.make_dir(plot_dir)
        return plot_dir

    @staticmethod
    def get_most_recent_plot_dir(experiment_dir):
        plot_dir = os.path.join(experiment_dir, 'plots')
        if not os.path.exists(plot_dir):
            return None
        subdirs = [subdir for subdir in os.listdir(plot_dir) if os.path.isdir(os.path.join(plot_dir, subdir))]
        best_time = None
        recent_dir = None
        for subdir in subdirs:
            time = datetime.strptime(subdir, "%Y%m%d-%H%M%S")
            if best_time == None or time > best_time:
                best_time = time
                recent_dir = subdir
        return os.path.join(plot_dir, recent_dir)

    @staticmethod
    def get_plot_pathname(plot_dir, plot_name):
        return os.path.join(plot_dir, plot_name)

    ##################################################
    ################### RENDER #######################
    ##################################################
    @staticmethod
    def get_rendering_dir(trajectory_dir, rendering_no):
        rendering_base_dir = os.path.join(trajectory_dir, 'renderings')
        render_dir = os.path.join(rendering_base_dir, 'render_{}'.format(str(rendering_no).zfill(5)))
        dh.make_dir(render_dir)
        return render_dir

    @staticmethod
    def get_new_rendering_dir(trajectory_dir):
        rendering_base_dir = os.path.join(trajectory_dir, 'renderings')
        dh.make_dir(rendering_base_dir)
        numstr = dh.get_max_dirno(rendering_base_dir, 'render')
        if numstr is None: numstr = 0
        numstr += 1
        new_dir = os.path.join(rendering_base_dir, 'render_{}'.format(str(numstr).zfill(5)))
        dh.make_dir(new_dir)
        return new_dir

    @staticmethod
    def get_rendering_plot_dir(rendering_dir):
        new_dir = os.path.join(rendering_dir, 'pngs')
        dh.make_dir(new_dir)
        return new_dir

    @staticmethod
    def get_gui_render_path(save_dir, count):
        return os.path.join(save_dir, 'chain_gui_{}.png'.format(str(count).zfill(5)))



    ##################################################
    ################ RANDOM SAMPLERS #################
    ##################################################

    @staticmethod
    def get_random_state_seed_path(save_dir, name=None):
        path = os.path.join(save_dir, 'random_state_seed_state')
        if name is not None:
            path = path+'_'+name
        return path+'.sav'


    @staticmethod
    def get_icra_point_samples_section_dir(save_dir, section_no):
        return 'scripts/icra_experiments/data/section_no_{}'
