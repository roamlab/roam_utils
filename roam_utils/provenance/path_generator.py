import os
import time
from datetime import datetime


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
        os.makedirs(os.path.join(path), exist_ok=True)
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
        return get_max_dirno(path, keyword)

    @staticmethod
    def get_max_fileno(path, keyword):
        return get_max_fileno(path, keyword)

    @staticmethod
    def get_save_dir(experiment_dir):
        # returns save directory for an experiment
        save_dir = os.path.join(experiment_dir, 'save')
        make_dir(save_dir)
        return save_dir

    @staticmethod
    def get_loss_plot_pathname(plot_dir):
        return os.path.join(plot_dir, 'loss_plot.png')

    @staticmethod
    def get_experiment_dir(path_to_experiment_dir, robot_name, experiment_type, experiment_no):
        experiment_dir = os.path.join(path_to_experiment_dir, 'experiments', robot_name, experiment_type,
                                      'experiment_{}'.format(str(experiment_no).zfill(3)))
        make_dir(experiment_dir)
        return experiment_dir

    @staticmethod
    def make_dir(path):
        return make_dir(path)

    @staticmethod
    def get_custom_path(directory, name, ext):
        return os.path.join(directory, name+ext)

    @staticmethod
    def get_plot_dir(experiment_dir, plot_no=None):
        plot_dir = os.path.join(experiment_dir, 'plots')
        if plot_no is None:
            plot_no = get_max_dirno(plot_dir, 'plot')
            if plot_no is None: plot_no = 0
            plot_no += 1
        plot_dir = os.path.join(plot_dir, 'plot_'+str(plot_no).zfill(5))
        make_dir(plot_dir)
        return plot_dir

    @staticmethod
    def get_dft_plot_dir(experiment_dir):
        directory = os.path.join(experiment_dir, 'dft_analysis_plots')
        make_dir(directory)
        return directory
    ##################################################
    ################### CONFIGS ######################
    ##################################################

    @staticmethod
    def get_config_from_dir(any_dir):
        return get_file_of_specific_extension_from_dir(any_dir, '.cfg')

    @staticmethod
    def get_svg_from_dir(any_dir):
        return get_file_of_specific_extension_from_dir(any_dir, '.svg')

    @staticmethod
    def get_png_from_dir(any_dir):
        return get_file_of_specific_extension_from_dir(any_dir, '.png')

    @staticmethod
    def get_config_pathname(load_dir, experiment_no):
        return os.path.join(load_dir, 'config_'+str(experiment_no).zfill(2)+'.cfg')

    @staticmethod
    def get_config_with_custom_name(load_dir, name):
        return os.path.join(load_dir, name+'.cfg')

    ##################################################
    #################### RRT #########################
    ##################################################
    @staticmethod
    def get_rrt_experiment_dir(path_to_experiments, robot_name, experiment_no):
        experiment_dir = os.path.join(path_to_experiments, 'experiments', robot_name, 'rrt',
                                      'experiment_' + str(experiment_no).zfill(2))
        make_dir(experiment_dir)
        return experiment_dir

    @staticmethod
    def get_rrt_tree_savepath(save_dir, iter):
        return os.path.join(save_dir, 'node_list_{}.sav'.format(str(iter).zfill(6)))

    @staticmethod
    def get_max_rrt_node_list_len(save_dir):
        return get_max_fileno(save_dir, 'node_list_')


    @staticmethod
    def get_rrt_gui_save_name(save_dir, node_list_len, ext='.png'):
        return os.path.join(save_dir, 'node_list_{}'.format(str(node_list_len).zfill(6))+ext)

    @staticmethod
    def get_rrt_path_savepath(rendering_dir):
        return os.path.join(rendering_dir, 'path.sav')

    @staticmethod
    def get_node_list_analysis_dir(rrt_experiment_dir, node_list_len):
        node_list_analysis_dir = os.path.join(rrt_experiment_dir, 'node_list_analysis', 'node_list_{}'.format(str(node_list_len).zfill(5)))
        make_dir(node_list_analysis_dir)
        return node_list_analysis_dir

    @staticmethod
    def get_node_list_analysis_state_bar_graph_savepath(node_list_analysis_dir):
        return os.path.join(node_list_analysis_dir, 'percent_outside_state_bounds_bar_graph.png')

    @staticmethod
    def get_node_list_analysis_action_bar_graph_savepath(node_list_analysis_dir):
        return os.path.join(node_list_analysis_dir, 'percent_outside_action_bounds_bar_graph.png')

    @staticmethod
    def get_node_list_analysis_state_distribution_graph_savepath(node_list_analysis_dir, state_idxs):
        return os.path.join(node_list_analysis_dir, 'state_distribution_graph_idxs_{}.png'.format(str(state_idxs)))

    @staticmethod
    def get_node_list_analysis_action_distribution_graph_savepath(node_list_analysis_dir, action_idxs):
        return os.path.join(node_list_analysis_dir, 'action_distribution_graph_idxs_{}.png'.format(str(action_idxs)))

    @staticmethod
    def get_node_list_histogram_save_dir(rrt_experiment_dir, save_no=None):
        histogram_dir = os.path.join(rrt_experiment_dir, 'histograms')
        if save_no is None:
            save_no = get_max_dirno(histogram_dir, 'histogram_')
            if save_no is None: save_no = 0
            save_no += 1
        dataset_dir = os.path.join(histogram_dir, 'histogram_{}'.format(str(save_no).zfill(5)))
        make_dir(dataset_dir)
        return dataset_dir


    ##################################################
    #################### SST #########################
    ##################################################

    @staticmethod
    def get_sst_experiment_dir(path_to_experiments, robot_name, experiment_no):
        experiment_dir = os.path.join(path_to_experiments, 'experiments', robot_name, 'sst',
                                      'experiment_' + str(experiment_no).zfill(2))
        make_dir(experiment_dir)
        return experiment_dir

    @staticmethod
    def get_sst_tree_savepath(save_dir, iter):
        return os.path.join(save_dir, 'node_list_{}.sav'.format(str(iter).zfill(6)))

    @staticmethod
    def get_sst_gui_save_name(save_dir, iter):
        return os.path.join(save_dir, 'node_list_{}.png'.format(str(iter).zfill(6)))

    @staticmethod
    def get_sst_path_savepath(rendering_dir):
        return os.path.join(rendering_dir, 'path.sav')

    ##################################################
    ###################### MPC #######################
    ##################################################
    @staticmethod
    def get_mpc_experiment_dir(path_to_experiments, robot_name, optimizer_name, experiment_name=None,
                               experiment_no=None):
        if experiment_name is None:
            experiment_name = 'experiment'
        experiment_dir = os.path.join(path_to_experiments, 'experiments', robot_name, 'mpc', optimizer_name)
        if experiment_no is None:
            max_dir_no = get_max_dirno(experiment_dir, experiment_name)
            if max_dir_no is None: max_dir_no = 0
            experiment_no = max_dir_no + 1
        experiment_dir = os.path.join(experiment_dir, experiment_name+'_'+str(experiment_no).zfill(5))
        make_dir(experiment_dir)
        return experiment_dir

    @staticmethod
    def get_mpc_iter_dir(save_dir, iter):
        mpc_iter_dir = os.path.join(save_dir, 'mpc_iter_'+str(iter).zfill(5))
        make_dir(mpc_iter_dir)
        return mpc_iter_dir

    @staticmethod
    def get_mpc_sequential_raw_dataset_dir(mpc_dir):
        mpc_sequential_raw_dataset_dir = os.path.join(mpc_dir, 'sequential')
        make_dir(mpc_sequential_raw_dataset_dir)
        return mpc_sequential_raw_dataset_dir

    @staticmethod
    def get_mpc_sequential_data_pathname(mpc_sequential_raw_dataset_dir, extention='.sav'):
        save_sequential_path = os.path.join(mpc_sequential_raw_dataset_dir, 'sequential'+extention)
        return save_sequential_path

    ##################################################
    ################## FORWARD MODEL #################
    ##################################################
    @staticmethod
    def get_predictor_experiment_dir(path_to_experiment_dir, robot_name, direction, algorithm_name, predictor_no=None):
        # returns experiment directory path
        algorithm_dir = os.path.join(path_to_experiment_dir, 'experiments', robot_name,
                                     '{}_predictors'.format(direction), algorithm_name)
        if predictor_no is None:
            max_dir_no = get_max_dirno(algorithm_dir, 'predictor')
            if max_dir_no is None: max_dir_no = 0
            predictor_no = max_dir_no+1
        experiment_dir = os.path.join(algorithm_dir,'predictor_' + str(predictor_no).zfill(2))
        make_dir(experiment_dir)
        return experiment_dir


    ##################################################
    #################### PREDICTOR ###################
    ##################################################
    @staticmethod
    def get_forward_model_experiment_dir(path_to_experiments, robot_name, algorithm_name, experiment_no):
        # returns experiment directory path
        experiment_dir = os.path.join(path_to_experiments, 'experiments', robot_name, 'forward_models', algorithm_name, 'model_' + str(experiment_no).zfill(2))
        make_dir(experiment_dir)
        return experiment_dir

    @staticmethod
    def get_path_to_dir_of_forward_model_experiment_dirs(path_to_experiments, robot_name, algorithm_name):
        path_to_dir_of_forward_model_experiment_dirs = os.path.join(path_to_experiments, 'experiments', robot_name, 'forward_models', algorithm_name)
        make_dir(path_to_dir_of_forward_model_experiment_dirs)
        return path_to_dir_of_forward_model_experiment_dirs

    @staticmethod
    def get_standardized_save_dir(save_dir):
        standardized_save_dir = os.path.join(save_dir, 'standardized')
        make_dir(standardized_save_dir)
        return standardized_save_dir

    @staticmethod
    def get_epoch_save_dir(save_dir, epoch_no):
        # returns save directory for specific epoch of forward model training
        epoch_save_dir = os.path.join(save_dir, 'epoch_{}'.format(str(epoch_no)))
        make_dir(epoch_save_dir)
        return epoch_save_dir

    @staticmethod
    def get_max_epoch_no(save_dir):
        return get_max_dirno(save_dir, 'epoch')

    @staticmethod
    def get_epoch_pytorch_save_pathname(epoch_save_dir):
        return os.path.join(epoch_save_dir, 'pytorch.pth')

    @staticmethod
    def get_predictor_standardizer_pathname(save_dir):
        return os.path.join(save_dir, 'standardizer.sav')

    @staticmethod
    def get_predictor_pathname(save_dir):
        return os.path.join(save_dir, 'predictor.sav')

    @staticmethod
    def get_subpredictor_save_dir(save_dir, name):
        submodel_save_dir = os.path.join(save_dir, name)
        make_dir(submodel_save_dir)
        return submodel_save_dir

    @staticmethod
    def get_best_epoch_save_pathname(save_dir, name):
        return os.path.join(save_dir, name + '_best_checkpoint.pth')

    @staticmethod
    def get_experiment_evaluate_dir(experiment_dir, evaluate_no=None):
        # type: (object, object) -> object
        evaluate_dir = os.path.join(experiment_dir, 'evaluations')
        if evaluate_no is None:
            max_dir_no = get_max_dirno(evaluate_dir, 'predictor')
            if max_dir_no is None: max_dir_no = 0
            evaluate_no = max_dir_no+1
        else:
            evaluate_no=0
        evaluate_dir = os.path.join(evaluate_dir, 'evaluation_{}'.format(str(evaluate_no).zfill(5)))
        make_dir(evaluate_dir)
        return evaluate_dir

    @staticmethod
    def get_test_specific_evaluate_dir(evaluate_dir, test_filename, loss_name, horizon_steps, start_time, end_time):
        specific_name = os.path.join(test_filename + '_' + loss_name + '_' + str(horizon_steps).zfill(2) + '_'
                                     + str(start_time).replace('.','') + '_' + str(end_time).replace('.',''))
        test_specific_evaluate_dir = os.path.join(evaluate_dir, specific_name)
        make_dir(test_specific_evaluate_dir)
        return test_specific_evaluate_dir

    @staticmethod
    def get_horizon_loss_save_pathname(save_dir):
        return os.path.join(save_dir, 'horizon_loss_params.json')

    @staticmethod
    def get_epoch_loss_plot_pathname(experiment_dir, experiment_no):
        save_path = os.path.join(experiment_dir, 'epoch_loss_{}.png'.format(str(experiment_no).zfill(2)))
        return save_path

    @staticmethod
    def get_loss_storage_pathname(loss_storage_dir, ext):
        return os.path.join(loss_storage_dir, 'loss_storage'+ext)

    @staticmethod
    def get_plot_name(start, horizon, custom_commands=[]):
        plot_name_format = 'horizon_{}_start_{}.png'
        plot_name = ""
        for command_to_plot in custom_commands:
            plot_name = "{}{}_".format(plot_name,command_to_plot)
        plot_name = plot_name + plot_name_format.format(str(int(horizon)),
                                                    str(round(start*1000, 4)).replace('.','').zfill(4))
        return plot_name

    @staticmethod
    def get_test_plot_dir(plot_dir, test_filename):
        test_plot_dir = os.path.join(plot_dir, test_filename)
        make_dir(test_plot_dir)
        return test_plot_dir

    @staticmethod
    def get_plot_config_pathname(plot_dir, config_path):
        config_filename = os.path.basename(config_path)
        return os.path.join(plot_dir, config_filename)

    ##################################################
    ################## DATASETS ######################
    ##################################################
    @staticmethod
    def get_dataset_pathname(dataset_dir, filename, extension, tsi):
        filename_no_extension = filename.split('.')[0]
        filename_with_tsi = filename_no_extension+'_'+str(tsi)
        return os.path.join(dataset_dir, filename_with_tsi+extension)

    @staticmethod
    def get_feature_filepath(load_dir, ext='.sav'):
        set_no = os.path.basename(load_dir)
        return os.path.join(load_dir, set_no+'_features'+ext)

    @staticmethod
    def get_label_filepath(load_dir, ext='.sav'):
        set_no = os.path.basename(load_dir)
        return os.path.join(load_dir, set_no+'_labels'+ext)

    @staticmethod
    def get_label_feature_param_filepath(load_dir, ext='.sav'):
        set_no = os.path.basename(load_dir)
        return os.path.join(load_dir, set_no+'_params'+ext)

    @staticmethod
    def get_label_feature_meat_save_pathname(save_dir):
        set_no = os.path.basename(save_dir)
        return os.path.join(save_dir, set_no+'_load_dict.sav')

    @staticmethod
    def get_sequential_data_pathname(save_dir, set_no=None):
        return os.path.join(save_dir, 'sequential_dataset.sav')

    @staticmethod
    def get_sequential_dataset_param_dict_pathname(save_dir, filename):
        return os.path.join(save_dir, filename.split('.')[0]+'param_dict.sav')

    @staticmethod
    def get_meta_save_pathname(save_dir):
        return os.path.join(save_dir, 'meta.sav')

    @staticmethod
    def get_data_driven_predictor_dataset_dict_save_path(save_dir, name=None):
        pathname = os.path.join(save_dir, 'data_driven_predictor_dataset')
        if name is not None:
            pathname = pathname+name
        pathname = pathname+'.sav'
        return pathname

    @staticmethod
    def get_raw_data_pathname(robot_raw_data_dir, ext='.sav'):
        return os.path.join(robot_raw_data_dir, 'dataset'+ext)

    @staticmethod
    def get_data_filter_pathname(save_dir):
        return os.path.join(save_dir, 'data_filter.sav')

    @staticmethod
    def get_label_feature_dataset_dict_pathname(label_feature_data_dir):
        return os.path.join(label_feature_data_dir, 'label_feature_dataset.sav')

    @staticmethod
    def get_csv_dir(save_dir):
        csv_dir = os.path.join(save_dir, 'csv')
        make_dir(csv_dir)
        return csv_dir

    ##################################################
    ############## COLLECT DATASET ###################
    ##################################################

    @staticmethod
    def get_sequential_raw_data_dir(path_to_data_dir, robot_name, set_no=None, custom_dir_list=None):
        dataset_dir = os.path.join(path_to_data_dir, robot_name, 'sequential_raw_datasets')
        if custom_dir_list is not None:
            for custom_dir in custom_dir_list:
                dataset_dir = os.path.join(dataset_dir, custom_dir)
        if set_no is None:
            set_no = get_max_dirno(dataset_dir, 'dataset_')
            if set_no is None:
                set_no = 0
            set_no += 1
        dataset_dir = os.path.join(dataset_dir, 'dataset_{}'.format(str(set_no).zfill(5)))
        make_dir(dataset_dir)
        return dataset_dir

    @staticmethod
    def get_random_raw_data_dir(path_to_data_dir, robot_name, set_no=None, custom_dir_list=None):
        dataset_dir = os.path.join(path_to_data_dir, robot_name, 'random_raw_datasets')
        if custom_dir_list is not None:
            for custom_dir in custom_dir_list:
                dataset_dir = os.path.join(dataset_dir, custom_dir)
        if set_no is None:
            set_no = get_max_dirno(dataset_dir, 'dataset_')
            if set_no is None:
                set_no = 0
            set_no += 1
        dataset_dir = os.path.join(dataset_dir, 'dataset_{}'.format(str(set_no).zfill(5)))
        make_dir(dataset_dir)
        return dataset_dir

    @staticmethod
    def get_label_feature_data_dir(path_to_data_dir, robot_name, data_direction, features_type, label_type, set_no=None, custom_dir_list=None):
        dataset_dir = os.path.join(path_to_data_dir, robot_name,
                                   '{}_model_label_feature_datasets'.format(data_direction))
        dataset_dir = os.path.join(dataset_dir, '{}'.format(features_type))
        dataset_dir = os.path.join(dataset_dir, '{}_labels'.format(label_type))
        if custom_dir_list is not None:
            for custom_dir in custom_dir_list:
                dataset_dir = os.path.join(dataset_dir, custom_dir)
        make_dir(dataset_dir)
        if set_no is None:
            set_no = get_max_dirno(dataset_dir, 'dataset_')
            if set_no is None: set_no = 0
            set_no += 1
        dataset_dir = os.path.join(dataset_dir, 'dataset_{}'.format(str(set_no).zfill(5)))
        make_dir(dataset_dir)
        return dataset_dir

    @staticmethod
    def get_label_feature_dataset_save_dir(label_feature_data_dir, iteration):
        label_feature_dataset_save_dir = os.path.join(label_feature_data_dir, 'save_{}'.format(str(iteration).zfill(7)))
        make_dir(label_feature_dataset_save_dir)
        return label_feature_dataset_save_dir

    ##################################################
    #################### PPO #########################
    ##################################################
    @staticmethod
    def get_ppo_experiment_dir(path_to_experiments, robot_name, experiment_no):
        experiment_dir = os.path.join(path_to_experiments, 'experiments', robot_name, 'ppo',
                                      'experiment_' + str(experiment_no).zfill(2))
        make_dir(experiment_dir)
        return experiment_dir

    @staticmethod
    def get_ppo_her_experiment_dir(path_to_experiments, robot_name, experiment_no):
        experiment_dir = os.path.join(path_to_experiments, 'experiments', robot_name, 'ppo_her',
                                      'experiment_' + str(experiment_no).zfill(2))
        make_dir(experiment_dir)
        return experiment_dir

    @staticmethod
    def get_ppo_model_path(experiments_dir, seed):
        model_dir = os.path.join(experiments_dir, 'seed-{}'.format(seed))
        make_dir(model_dir)
        model_path = os.path.join(model_dir, 'model.pkl'.format(seed))
        return model_path

    @staticmethod
    def get_env_path(experiments_dir, seed):
        model_dir = os.path.join(experiments_dir, 'env')
        make_dir(model_dir)
        model_path = os.path.join(model_dir, 'env_seed_{}.pkl'.format(seed))
        return model_path

    @staticmethod
    def get_ppo_log_dir(experiments_dir, seed):
        logdir = os.path.join(experiments_dir, 'seed-{}'.format(seed))
        make_dir(logdir)
        return logdir

    @staticmethod
    def get_ppo_trajectory_dir(experiments_dir, trajectory_no):
        trajectory_dir = os.path.join(experiments_dir, 'trajectories', 'trajectory_dir_{}'.format(str(
            trajectory_no).zfill(5)))
        make_dir(trajectory_dir)
        return trajectory_dir

    ##################################################
    #################### SBRL ########################
    ##################################################
    @staticmethod
    def get_sbrl_experiment_dir(path_to_experiments, robot_name, experiment_no):
        experiment_dir = os.path.join(path_to_experiments, 'experiments', robot_name, 'sbrl',
                                      'experiment_' + str(experiment_no).zfill(2))
        make_dir(experiment_dir)
        return experiment_dir

    @staticmethod
    def get_sbrl_plot_path(experiment_dir, ext='.png', number=None, name=None):
        filename = 'sbrl_training_'+str(name) if (name is not None) else 'sbrl_training'
        filename = filename+'_'+str(int(number)).zfill(7) if (number is not None) else filename
        filename = filename+ext
        return os.path.join(experiment_dir, filename)

    @staticmethod
    def get_sbrl_buffer_savepath(experiment_dir, iter_num):
        filename = 'sbrl_'+str(iter_num)+'_buffer' if (iter_num is not None) else 'sbrl_buffer'
        filename = filename+'.sav'
        return os.path.join(experiment_dir, filename)

    @staticmethod
    def get_sbrl_rand_savepath(experiment_dir, iter_num):
        filename = 'sbrl_'+str(iter_num)+'_rand' if (iter_num is not None) else 'sbrl_rand'
        filename = filename+'.sav'
        return os.path.join(experiment_dir, filename)

    ##################################################
    ################ PROBLEM SPACE ###################
    ##################################################
    @staticmethod
    def get_problem_space_path(save_dir):
        problem_space_path = os.path.join(save_dir, 'problem_space.sav')
        return problem_space_path

    ##################################################
    ################### TRAJECTORY ###################
    ##################################################
    @staticmethod
    def get_trajectory_optimizer_dir(experiment_dir):
        traj_dir = os.path.join(experiment_dir, 'trajectory_optimizer_save')
        make_dir(traj_dir)
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
        make_dir(record_sim_dir)
        return record_sim_dir

    @staticmethod
    def get_serpenoid_curve_experiment_dir(experiments_dir, robot_name, experiment_no=None, experiment_name='experiment'):
        experiments_dir = os.path.join(experiments_dir, 'experiments', robot_name, 'serpenoid_curve')
        make_dir(experiments_dir)
        if experiment_no is None:
            experiment_no = get_max_dirno(experiments_dir, experiment_name+'_')
            if experiment_no is None: experiment_no = 0
            experiment_no += 1
        experiments_dir = os.path.join(experiments_dir, experiment_name + '_' + str(experiment_no).zfill(5))
        make_dir(experiments_dir)
        return experiments_dir

    @staticmethod
    def get_trajectory_dir(experiment_dir, trajectory_no=None, trajectory_name=None):
        trajectory_dir = os.path.join(experiment_dir, 'trajectories')
        trajectory_dir_name = 'trajectory_'+str(trajectory_name) if (trajectory_name is not None) else 'trajectory'
        if trajectory_no is not None:
            trajectory_dir_name = trajectory_dir_name + '_'+str(trajectory_no).zfill(5)
        trajectory_dir = os.path.join(trajectory_dir, trajectory_dir_name)
        make_dir(trajectory_dir)
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
        make_dir(trajectory_base_dir)
        numstr = get_max_dirno(trajectory_base_dir, trajectory_dir_name)
        if numstr is None: numstr = 0
        numstr += 1
        trajectory_dir_name = trajectory_dir_name+'_'+str(numstr).zfill(5)
        new_dir = os.path.join(trajectory_base_dir, trajectory_dir_name)
        make_dir(new_dir)
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
        make_dir(plot_dir)
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
        make_dir(render_dir)
        return render_dir

    @staticmethod
    def get_new_rendering_dir(trajectory_dir):
        rendering_base_dir = os.path.join(trajectory_dir, 'renderings')
        make_dir(rendering_base_dir)
        numstr = get_max_dirno(rendering_base_dir, 'render')
        if numstr is None: numstr = 0
        numstr += 1
        new_dir = os.path.join(rendering_base_dir, 'render_{}'.format(str(numstr).zfill(5)))
        make_dir(new_dir)
        return new_dir

    @staticmethod
    def get_rendering_plot_dir(rendering_dir):
        new_dir = os.path.join(rendering_dir, 'pngs')
        make_dir(new_dir)
        return new_dir

    @staticmethod
    def get_gui_render_path(save_dir, count):
        return os.path.join(save_dir, 'chain_gui_{}.png'.format(str(count).zfill(5)))


    ##################################################
    ################ DISTANCE MATRIX #################
    ##################################################

    @staticmethod
    def get_new_distance_matrix_dir(rrt_experiment_dir):
        base_dir = os.path.join(rrt_experiment_dir, 'distance_matrices')
        make_dir(base_dir)
        numstr = get_max_dirno(base_dir, 'distance_matrix')
        if numstr is None: numstr = 0
        numstr += 1
        new_dir = os.path.join(base_dir, 'distance_matrix_{}'.format(str(numstr).zfill(3)))
        make_dir(new_dir)
        return new_dir

    @staticmethod
    def get_distance_matrix_dir(rrt_experiment_dir, distance_matrix_no):
        distance_dir = os.path.join(rrt_experiment_dir, 'distance_matrices', 'distance_matrix_{}'.
                                    format(str(distance_matrix_no).zfill(3)))
        return distance_dir

    @staticmethod
    def get_distance_matrix_path(distance_matrix_dir):
        return os.path.join(distance_matrix_dir, 'distance_matrix.sav')

    @staticmethod
    def get_new_closest_nodes_dir(rrt_experiment_dir):
        base_dir = os.path.join(rrt_experiment_dir, 'closest_nodes')
        make_dir(base_dir)
        numstr = get_max_dirno(base_dir, 'closest_nodes')
        if numstr is None: numstr = 0
        numstr += 1
        new_dir = os.path.join(base_dir, 'closest_nodes_{}'.format(str(numstr).zfill(3)))
        make_dir(new_dir)
        return new_dir

    @staticmethod
    def get_closest_nodes_path(closest_nodes_dir):
        return os.path.join(closest_nodes_dir, 'closest_nodes_dict.sav')

    @staticmethod
    def get_closest_nodes_dir(rrt_experiment_dir, closest_nodes_dict_no):
        return os.path.join(rrt_experiment_dir, 'closest_nodes', 'closest_nodes_{}'.format(str(closest_nodes_dict_no).zfill(3)))



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
