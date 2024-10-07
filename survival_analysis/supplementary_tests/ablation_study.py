# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 10/09/2024


# Packages to import
import os
import sys

import torch

import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..' + os.sep + '..'))
sys.path.append(parent_dir)
from tabulate import tabulate
from data import split_cv_data
from colorama import Fore, Style
from joblib import Parallel, delayed
from validation import plot_model_losses
from survival_analysis.savae import SAVAE
from utils import run_args, create_output_dir, save, check_file


# Choose best results for each fold
def get_fold_best_seed_results(results, params, n_seeds, n_folds, seeds_eval=3):
    best_results = {'avg_ci': 0.0, 'avg_ibs': 0.0, 'best_cis': [], 'param_comb': ''}

    model_params = str(params['latent_dim']) + '_' + str(params['hidden_size']) + '_' + str(params['dropout_prop'])
    fold_results = []
    for fold in range(n_folds):
        # Average results from folds
        ci_per_seed = [results[model_params][seed][fold]['ci'][-1] for seed in range(n_seeds)]
        ibs_per_seed = [results[model_params][seed][fold]['ibs'][-1] for seed in range(n_seeds)]

        # Select based on both metrics
        differences = []
        best_idx = []
        for i in range(len(ci_per_seed)):
            diff = ci_per_seed[i][1] - ibs_per_seed[i][1]
            if seeds_eval > len(differences):
                best_idx.append(i)
                differences.append(diff)
            else:
                min_dif_idx = np.argmin(differences)
                if diff > differences[min_dif_idx]:
                    differences[min_dif_idx] = diff
                    best_idx[min_dif_idx] = i
        fold_results.append((fold, np.mean(np.array([ci[1] for ci in ci_per_seed])[best_idx]),
                             [ci_per_seed[idx] for idx in best_idx],
                             np.mean(np.array([ibs[1] for ibs in ibs_per_seed])[best_idx]),
                             [ibs_per_seed[idx] for idx in best_idx]))

    avg_ci = sum([x[1] for x in fold_results]) / n_folds
    best_results['avg_ci'] = avg_ci
    best_results['param_comb'] = model_params
    best_results['best_cis'] = [x[2] for x in fold_results]
    best_results['best_ibs'] = [x[4] for x in fold_results]
    best_results['avg_ibs'] = sum([x[3] for x in fold_results]) / n_folds
    return best_results


def train(data, feat_distributions_no_sa, params, seed, output_dir, args, fold):
    # Prepare savae params
    max_t = max([np.amax(data[0].iloc[:, -2]), np.amax(data[2].iloc[:, -2])])
    model_params = {'feat_distributions': feat_distributions_no_sa,
                    'latent_dim': params['latent_dim'],
                    'hidden_size': params['hidden_size'],
                    'dropout_prop': params['dropout_prop'],
                    'input_dim': len(feat_distributions_no_sa),
                    'max_t': max_t,
                    'time_dist': args['time_distribution'],
                    'early_stop': args['early_stop']}
    model = SAVAE(model_params)
    model_path = str(model_params['latent_dim']) + '_' + str(model_params['hidden_size']) + '_' + str(
        model_params['dropout_prop'])
    seed_path = model_path + os.sep + 'seed_' + str(seed)
    log_name = output_dir + seed_path + os.sep + 'model_fold_' + str(fold)

    # Train model
    train_params = {'n_epochs': args['n_epochs'], 'batch_size': args['batch_size'], 'device': torch.device('cpu'),
                    'lr': args['lr'], 'path_name': log_name}
    training_results = model.fit(data, train_params)

    # Plot losses
    path_name = str(train_params['path_name'])
    for loss in ['loss_', 'll_cov_', 'kl_', 'll_time_']:
        plot_model_losses(training_results[loss + 'tr'], training_results[loss + 'va'],
                          path_name + '_' + loss + 'losses.png', 'Train and Validation ' + loss + 'losses')

    # Save base_model information
    model.save(log_name + '.pt')
    model_params.update(train_params)
    model_params.update(training_results)
    save(model_params, log_name + '.pickle')

    # Load validation results
    ci_results = model_params['ci_va']
    ibs_results = model_params['ibs_va']

    return ci_results, ibs_results, params, seed, fold


def main():
    print(Fore.RED + '\n\n-------- ABLATION STUDY - SURVIVAL ANALYSIS  --------' + Style.RESET_ALL)

    # Environment configuration
    task = 'ablation_study'
    args = run_args(task)
    create_output_dir(task, args)

    if args['train']:
        print('\n----SURVIVAL ANALYSIS TRAINING----')
        for dataset_name in args['datasets']:
            print('\n\nDataset: ' + Fore.CYAN + dataset_name + Style.RESET_ALL)

            # Load data
            input_dir = args['input_dir'] + dataset_name
            real_df = pd.read_csv(input_dir + os.sep + '/data.csv')
            output_dir = args['output_dir'] + dataset_name + os.sep + str(args['n_folds']) + '_folds' + os.sep

            # Prepare data
            cv_data, feat_distributions = split_cv_data(real_df, args['n_folds'], time_dist=args['time_distribution'])

            # Train model
            models_results = Parallel(n_jobs=args['n_threads'], verbose=10)(
                delayed(train)(cv_data[fold], feat_distributions[:-2], params, seed, output_dir, args, fold) for params
                in args['param_comb'] for seed in range(args['n_seeds']) for fold in range(args['n_folds']))

            # Create dictionary to save results
            results = {}
            for params in args['param_comb']:
                model_params = str(params['latent_dim']) + '_' + str(params['hidden_size']) + '_' + str(
                    params['dropout_prop'])
                results[model_params] = {}
                for seed in range(args['n_seeds']):
                    results[model_params][seed] = {}
                    for fold in range(args['n_folds']):
                        results[model_params][seed][fold] = {'ci': [], 'ibs': []}

            for res in models_results:
                params = res[2]
                seed = res[3]
                fold = res[4]
                model_params = str(params['latent_dim']) + '_' + str(params['hidden_size']) + '_' + str(
                    params['dropout_prop'])
                results[model_params][seed][fold]['ci'] = res[0]
                results[model_params][seed][fold]['ibs'] = res[1]

            # Save results
            save(results, output_dir + 'results.pkl')

    # Show results
    # TODO: Refactor this code to avoid code repetition
    print('\n\n----SURVIVAL ANALYSIS ABLATION RESULTS----')
    for dataset_name in args['datasets']:
        output_dir = args['output_dir'] + dataset_name + os.sep + str(args['n_folds']) + '_folds' + os.sep
        all_results = {}
        for params in args['param_comb']:
            params_output_dir = output_dir + str(params['latent_dim']) + '_' + str(params['hidden_size']) + '_' + str(
                params['dropout_prop']) + os.sep
            for seed in range(args['n_seeds']):
                seed_output_dir = params_output_dir + 'seed_' + str(seed) + os.sep
                for fold in range(args['n_folds']):
                    fold_output_dir = seed_output_dir + 'model_fold_' + str(fold) + '.pickle'
                    fold_results = check_file(fold_output_dir,
                                              'Results file does not exist for ' + seed_output_dir + 'model_fold_' + str(
                                                  fold) + '.')
                    model_params = str(params['latent_dim']) + '_' + str(params['hidden_size']) + '_' + str(
                        params['dropout_prop'])
                    if model_params not in all_results:
                        all_results[model_params] = {}
                    if seed not in all_results[model_params]:
                        all_results[model_params][seed] = {}
                    if fold not in all_results[model_params][seed]:
                        all_results[model_params][seed][fold] = {'ci': [], 'ibs': []}
                    all_results[model_params][seed][fold]['ci'] = fold_results['ci_va']
                    all_results[model_params][seed][fold]['ibs'] = fold_results['ibs_va']

        # Save all results
        save(all_results, output_dir + 'all_results.pkl')

    for dataset_name in args['datasets']:
        table_data = []
        print('\nDataset: ' + Fore.CYAN + dataset_name + Style.RESET_ALL)
        output_dir = args['output_dir'] + dataset_name + os.sep + str(args['n_folds']) + '_folds' + os.sep
        results_path = output_dir + 'all_results.pkl'
        results = check_file(results_path, 'Results file for model does not exist.')
        for params in args['param_comb']:
            best_params_results = get_fold_best_seed_results(results, params, args['n_seeds'], args['n_folds'])

            # Save best results
            save(best_params_results,
                 output_dir + str(params['latent_dim']) + '_' + str(params['hidden_size']) + '_' + str(
                     params['dropout_prop']) + 'best_params_results.pkl')

            # Display results
            print('Parameters combination: ' + str(best_params_results['param_comb']))
            print('Average C-index from best seeds: ' + str(format(best_params_results['avg_ci'], '.3f')))
            print('Average IBS from best seeds: ' + str(format(best_params_results['avg_ibs'], '.3f')))
            print('\n')

            # Add results to table
            table_data.append([params['latent_dim'], params['hidden_size'], params['dropout_prop'],
                               format(best_params_results['avg_ci'], '.3f'),
                               format(best_params_results['avg_ibs'], '.3f')])

        # Print results
        headers = ['Latent Space Dimension', '# Hidden Neurons', 'Dropout Percentage', 'AVG CI', 'AVG IBS']
        table_data.insert(0, headers)
        print(tabulate(table_data, headers='firstrow', tablefmt='grid'))
        print(tabulate(table_data, headers='firstrow', tablefmt='latex'))
        print('\n\n')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
