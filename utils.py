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
            os.makedirs(args['output_dir'] + dataset_name + '/', exist_ok=True)
        elif task == 'sota_survival_analysis':
            for model in args['sota_models']:
                os.makedirs(
                    args['sota_output_dir'] + dataset_name + '/' + model + '/' + str(args['n_folds']) + '_folds/',
                    exist_ok=True)
        elif task == 'savae_survival_analysis':
            for params in args['param_comb']:
                for seed in range(args['n_seeds']):
                    model_path = str(params['latent_dim']) + '_' + str(params['hidden_size']) + '/seed_' + str(seed)
                    os.makedirs(args['output_dir'] + dataset_name + '/' + str(args['n_folds']) + '_folds/' +
                                model_path + '/', exist_ok=True)


# Function that sets environment configuration
def run_args(task='savae_survival_analysis'):
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

    # The following lines are used to run the code both in pycharm and in the terminal
    project_name = 'savae'  # TODO: change if project name changes
    divided_path = os.getcwd().split('/')
    path = []
    for parts in divided_path:
        if project_name in parts:
            path.append(parts)
            break
        path.append(parts)
    abs_path = '/'.join(path)

    # Depending on the task, set the arguments
    if task == 'data_preprocessing':
        args['output_dir'] = abs_path + '/data_preprocessing/data/'
        args['input_dir'] = abs_path + '/data_preprocessing/raw_data/'
    else:
        args['input_dir'] = abs_path + '/data_preprocessing/data/'

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
        args['sota_output_dir'] = abs_path + '/survival_analysis/output_sota_' + str(args['n_folds']) + '_folds_' + str(
            args['batch_size']) + '_batch_size_' + args['time_distribution'][0] + '/'
        model_name = 'all'
        args['sota_models'] = ['coxph', 'deepsurv', 'deephit'] if model_name == 'all' else [model_name]

        # SAVAE hyperparameters
        args['n_threads'] = 65
        args['n_seeds'] = 10
        default_params = True
        args['param_comb'] = [{'hidden_size': 50, 'latent_dim': 5}] if default_params else parameter_combination()

        # SAVAE output folders
        args['output_dir'] = abs_path + '/survival_analysis/output_savae_' + str(args['n_folds']) + '_folds_' + str(
            args['batch_size']) + '_batch_size_' + args['time_distribution'][0] + '/'

    return args
