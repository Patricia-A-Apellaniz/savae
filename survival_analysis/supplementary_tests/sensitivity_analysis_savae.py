# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 12/09/2024


# Packages to import
import os
import sys

import matplotlib.pyplot as plt
import shap
import torch

import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..' + os.sep))
sys.path.append(parent_dir)
from data import split_cv_data
from colorama import Fore, Style
from joblib import Parallel, delayed
from validation import plot_model_losses
from survival_analysis.savae import SAVAE
from utils import run_args, create_output_dir, save, check_file


# Choose best results for each fold
def get_fold_best_seed_results(results, param_comb, n_seeds, n_folds, seeds_eval=3):
    best_results = {'avg_ci': 0.0, 'avg_ibs': 0.0, 'best_cis': [], 'param_comb': ''}
    best_ci = 0.0
    for params in param_comb:
        model_params = str(params['latent_dim']) + '_' + str(params['hidden_size'])
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
        if avg_ci > best_ci:
            best_ci = avg_ci
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
    model_path = str(model_params['latent_dim']) + '_' + str(model_params['hidden_size'])
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

    # Sensitivity analysis
    # Configure input data and missing data mask
    x_train, _, x_test, _ = data
    train_cov = x_train.drop(['time', 'event'], axis=1)
    test_cov = x_test.drop(['time', 'event'], axis=1)
    # 1. Fits the explainer
    test_explainer = shap.Explainer(model.predict_just_time, test_cov, feature_names=test_cov.columns)
    # 2. Calculates the SHAP values - It takes some time
    test_shap_values = test_explainer(test_cov)
    # 3. Evaluate SHAP values
    # shap_values = explainer.shap_values(train_cov)
    # 4. If we simply want the feature importances as determined by SHAP algorithm, we need to take the mean average value for each feature.
    from scipy.special import softmax
    def print_feature_importances_shap_values(shap_values, features):
        '''
        Prints the feature importances based on SHAP values in an ordered way
        shap_values -> The SHAP values calculated from a shap.Explainer object
        features -> The name of the features, on the order presented to the explainer
        '''
        # Calculates the feature importance (mean absolute shap value) for each feature
        importances = []
        for i in range(shap_values.values.shape[1]):
            importances.append(np.mean(np.abs(shap_values.values[:, i])))
        # Calculates the normalized version
        importances_norm = softmax(importances)
        # Organize the importances and columns in a dictionary
        feature_importances = {fea: imp for imp, fea in zip(importances, features)}
        feature_importances_norm = {fea: imp for imp, fea in zip(importances_norm, features)}
        # Sorts the dictionary
        feature_importances = {k: v for k, v in
                               sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)}
        feature_importances_norm = {k: v for k, v in
                                    sorted(feature_importances_norm.items(), key=lambda item: item[1], reverse=True)}
        # Prints the feature importances
        for k, v in feature_importances.items():
            print(f"{k} -> {v:.4f} (softmax = {feature_importances_norm[k]:.4f})")

    print_feature_importances_shap_values(test_shap_values, test_cov.columns)
    # 5. For analysis of the global effect of the features we can use the following plots.
    shap.plots.bar(test_shap_values, show=False)
    plt.savefig(log_name + '_shap_bar_test.png')
    plt.close()
    shap.plots.beeswarm(test_shap_values, show=False)
    plt.savefig(log_name + '_shap_beeswarm_test.png')
    plt.close()
    shap.summary_plot(test_shap_values, plot_type='violin', show=False)
    plt.savefig(log_name + '_shap_summary_test.png')
    plt.close()
    # 6. For local analysis
    # shap.plots.bar(test_shap_values[0])
    # shap.plots.waterfall(test_shap_values[0])
    # shap.plots.force(test_shap_values[0])

    # Do the same for training
    train_explainer = shap.Explainer(model.predict_just_time, train_cov, feature_names=train_cov.columns)
    train_shap_values = train_explainer(train_cov)
    print_feature_importances_shap_values(train_shap_values, train_cov.columns)
    shap.plots.bar(train_shap_values, show=False)
    plt.savefig(log_name + '_shap_bar_train.png')
    plt.close()
    shap.plots.beeswarm(train_shap_values, show=False)
    plt.savefig(log_name + '_shap_beeswarm_train.png')
    plt.close()
    shap.summary_plot(train_shap_values, show=False, plot_type='violin')
    plt.savefig(log_name + '_shap_summary_train.png')
    plt.close()

    return ci_results, ibs_results, params, seed, fold


def main():
    print(Fore.RED + '\n\n-------- SURVIVAL ANALYSIS  --------' + Style.RESET_ALL)

    # Environment configuration
    task = 'sensitivity_savae_sa'
    args = run_args(task)
    create_output_dir(task, args)

    if args['train']:
        print('\n----SURVIVAL ANALYSIS TRAINING----')
        for dataset_name in args['datasets']:
            print('\n\nDataset: ' + Fore.CYAN + dataset_name + Style.RESET_ALL)

            # Load data
            input_dir = args['input_dir'] + dataset_name
            real_df = pd.read_csv(input_dir + os.sep + 'data.csv')
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
                model_params = str(params['latent_dim']) + '_' + str(params['hidden_size'])
                results[model_params] = {}
                for seed in range(args['n_seeds']):
                    results[model_params][seed] = {}
                    for fold in range(args['n_folds']):
                        results[model_params][seed][fold] = {'ci': [], 'ibs': []}

            for res in models_results:
                params = res[2]
                seed = res[3]
                fold = res[4]
                model_params = str(params['latent_dim']) + '_' + str(params['hidden_size'])
                results[model_params][seed][fold]['ci'] = res[0]
                results[model_params][seed][fold]['ibs'] = res[1]

            # Save results
            save(results, output_dir + 'results.pkl')

    # Show results
    print('\n\n----SURVIVAL ANALYSIS RESULTS----')
    for dataset_name in args['datasets']:
        print('\nDataset: ' + Fore.CYAN + dataset_name + Style.RESET_ALL)
        output_dir = args['output_dir'] + dataset_name + os.sep + str(args['n_folds']) + '_folds' + os.sep
        results_path = output_dir + 'results.pkl'
        results = check_file(results_path, 'Results file for model does not exist.')
        best_results = get_fold_best_seed_results(results, args['param_comb'], args['n_seeds'], args['n_folds'])

        # Save best results
        save(best_results, output_dir + 'best_results.pkl')

        # Display results
        print('Best hyperparameters: ' + str(best_results['param_comb']))
        print('Average C-index from best seeds: ' + str(format(best_results['avg_ci'], '.2f')))
        print('Average IBS from best seeds: ' + str(format(best_results['avg_ibs'], '.3f')))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
