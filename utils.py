# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 05/09/2023


# Packages to import
import os
import pickle


# Check if file exists
def check_file(path, msg):
    if os.path.exists(path):
        file = open(path, 'rb')
        results = pickle.load(file)
        file.close()
        return results
    else:
        raise RuntimeError(msg)


# Save dictionary to pickle file
def save(res, path):
    with open(path, 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)


#  Hyperparameter tuning. Gets all the possible combinations of parameters to train the base_model. Once found the
#  best combination, stop using it
def parameter_combination():
    hidden_size = [50, 100, 250, 500, 1000]
    latent_dim = [5, 10, 25, 50, 100]
    param_comb = []
    for hidden in hidden_size:
        for latent in latent_dim:
            new_params = {'hidden_size': hidden, 'latent_dim': latent}
            param_comb.append(new_params)

    return param_comb


# Function that creates output directories for each task
def create_output_dir(task, args):
    for dataset_name in args['datasets']:
        if task == 'data_preprocessing':
            os.makedirs(args['output_dir'] + dataset_name + os.sep, exist_ok=True)
        elif 'sota_sa' in task:
            for model in args['sota_models']:
                os.makedirs(args['sota_output_dir'] + dataset_name + os.sep + model + os.sep + str(
                    args['n_folds']) + '_folds' + os.sep, exist_ok=True)
        elif 'savae_sa' in task:
            for params in args['param_comb']:
                for seed in range(args['n_seeds']):
                    model_path = str(params['latent_dim']) + '_' + str(params['hidden_size']) + os.sep + 'seed_' + str(
                        seed)
                    os.makedirs(args['output_dir'] + dataset_name + os.sep + str(
                        args['n_folds']) + '_folds' + os.sep + model_path + os.sep, exist_ok=True)
        elif task == 'ablation_study':
            for params in args['param_comb']:
                for seed in range(args['n_seeds']):
                    model_path = str(params['latent_dim']) + '_' + str(params['hidden_size']) + '_' + str(
                        params['dropout_prop']) + os.sep + 'seed_' + str(seed)
                    os.makedirs(args['output_dir'] + dataset_name + os.sep + str(
                        args['n_folds']) + '_folds' + os.sep + model_path + os.sep, exist_ok=True)


# Function that sets environment configuration
def run_args(task):
    args = {}

    # Data
    datasets = []
    dataset_name = 'all'
    if dataset_name == 'all':
        datasets = ['whas', 'support', 'gbsg', 'flchain', 'nwtco', 'metabric', 'pbc', 'std', 'pneumon']
    else:
        datasets.append(dataset_name)
    args['datasets'] = datasets
    print('[INFO] Datasets: ', datasets)

    # Absolute path
    abs_path = os.path.dirname(os.path.abspath(__file__)) + os.sep

    # Depending on the task, set the arguments
    if task == 'data_preprocessing':
        args['output_dir'] = abs_path + 'data_preprocessing' + os.sep + 'data' + os.sep
        args['input_dir'] = abs_path + 'data_preprocessing' + os.sep + 'raw_data' + os.sep
    else:
        args['input_dir'] = abs_path + 'data_preprocessing' + os.sep + 'data' + os.sep

        # Training and testing configurations for savae and sota models
        args['train'] = False
        args['eval'] = False
        args['early_stop'] = True
        args['n_folds'] = 5
        args['batch_size'] = 64
        args['n_epochs'] = 3000
        args['lr'] = 1e-3
        args['time_distribution'] = ('weibull', 2)

        # SOTA models
        args['sota_output_dir'] = abs_path + 'survival_analysis' + os.sep + 'output_sota_' + str(
            args['n_folds']) + '_folds_' + str(args['batch_size']) + '_batch_size_' + args['time_distribution'][
                                      0] + os.sep
        model_name = 'all'
        args['sota_models'] = ['coxph', 'deepsurv', 'deephit'] if model_name == 'all' else [model_name]

        if task == 'ablation_study':
            args['early_stop'] = True
            args['n_threads'] = 50
            args['n_seeds'] = 10
            args[
                'output_dir'] = abs_path + 'survival_analysis' + os.sep + 'supplementary_tests' + os.sep + 'AS_output_savae_' + str(
                args['n_folds']) + '_folds_' + str(args['batch_size']) + '_batch_size_' + args['time_distribution'][
                                    0] + os.sep

            # Compare different parameter results
            latent_dim = [3, 5, 50]
            hidden_size = [10, 50, 500]
            dropout_prop = [0.0, 0.2, 0.5]
            param_comb = []
            for hidden in hidden_size:
                for latent in latent_dim:
                    for dropout in dropout_prop:
                        new_params = {'hidden_size': hidden, 'latent_dim': latent, 'dropout_prop': dropout}
                        param_comb.append(new_params)
            args['param_comb'] = param_comb
            print(args['param_comb'])

        elif task == 'sensitivity_savae_sa':
            datasets = []
            dataset_name = 'all'
            if dataset_name == 'all':
                datasets = ['whas', 'gbsg', 'flchain', 'nwtco']
            else:
                datasets.append(dataset_name)
            args['datasets'] = datasets

            # SAVAE hyperparameters
            args['n_folds'] = 2
            args['n_epochs'] = 3000
            args['n_threads'] = 1
            args['n_seeds'] = 1
            default_params = True
            args['param_comb'] = [{'hidden_size': 50, 'latent_dim': 5,
                                   'dropout_prop': 0.2}] if default_params else parameter_combination()

            # SAVAE output folders
            args[
                'output_dir'] = abs_path + 'survival_analysis' + os.sep + 'supplementary_tests' + os.sep + 'SENSITIVITY_output_savae_' + str(
                args['n_folds']) + '_folds_' + str(args['batch_size']) + '_batch_size_' + args['time_distribution'][
                                    0] + os.sep

        else:
            # SAVAE hyperparameters
            args['n_threads'] = 35
            args['n_seeds'] = 10
            default_params = True
            args['param_comb'] = [{'hidden_size': 50, 'latent_dim': 5,
                                   'dropout_prop': 0.2}] if default_params else parameter_combination()

            # SAVAE output folders
            args['output_dir'] = abs_path + 'survival_analysis' + os.sep + 'output_savae_' + str(
                args['n_folds']) + '_folds_' + str(args['batch_size']) + '_batch_size_' + args['time_distribution'][
                                     0] + os.sep

    return args
