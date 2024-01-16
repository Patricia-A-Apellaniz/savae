# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 03/10/2023


# Packages to import
import os
import sys
import torch
import pickle

import numpy as np
import pandas as pd
import torchtuples as tt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from data import split_cv_data
from colorama import Fore, Style
from pycox.evaluation import EvalSurv
from validation import bern_conf_interval
from pycox.models import CoxPH, DeepHitSingle
from utils import run_args, create_output_dir, check_file


# REPOSITORY: https://nbviewer.org/github/havakv/pycox/blob/master/examples/cox-ph.ipynb
def train(args, df_train, df_test, get_target, x_train, y_train, x_test, y_test, output_dir, model_name, fold,
          early_stop=False):
    # Parameters
    in_features = x_train.shape[1]

    if model_name == 'deephit':
        # Data transformation
        # When it comes to the choice of num_durations, you can replace it with your own defined grid,
        # if you have some knowledge what a good discretization grid would be. Or if you have discrete event times,
        # you might just want to use those.
        if np.all(y_train[0] == y_train[0].astype(int)):
            num_durations = np.unique(y_train[0])
        else:
            num_durations = len(np.unique(y_train[0]))
        labtrans = DeepHitSingle.label_transform(num_durations)
        y_train = labtrans.fit_transform(*get_target(df_train))
        y_train = (y_train[0], y_train[1])
        y_test = (y_test[0], y_test[1])

        # Network: 2 layers 32 nodes
        num_nodes = [32, 32]
        out_features = labtrans.out_features
        batch_norm = True
        dropout = 0.2
        output_bias = False
        net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout,
                                      output_bias=output_bias)
        model = DeepHitSingle(net, tt.optim.Adam, alpha=0.2, sigma=0.1, duration_index=labtrans.cuts)
    else:
        if model_name == 'coxph':
            net = torch.nn.Linear(in_features, 1)
        elif model_name == 'deepsurv':
            num_nodes = [32, 32]
            out_features = 1
            batch_norm = True
            dropout = 0.1
            output_bias = False
            net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                                          dropout, output_bias=output_bias)
        else:
            raise RuntimeError('State-of-the-art model not recognized')

        model = CoxPH(net, tt.optim.Adam)

    batch_size = args['batch_size']
    model.optimizer.set_lr(0.01)
    epochs = args['n_epochs']
    verbose = True
    if early_stop:
        callbacks = [tt.callbacks.EarlyStopping(patience=30)]
        if model_name == 'deephit':
            y_val = labtrans.fit_transform(*get_target(df_test))
            x_val = x_test
            val = x_val, y_val
        else:
            val = x_test, y_test
        _ = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose, val_data=val,
                      val_batch_size=batch_size)
    else:
        callbacks = None
        _ = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose)

    # As CoxPH is semi-parametric, we first need to get the non-parametric baseline hazard estimates with
    # compute_baseline_hazards.
    if model_name == 'coxph' or model_name == 'deepsurv':
        _ = model.compute_baseline_hazards()
    surv = model.predict_surv_df(x_test)

    # Evaluation
    ev = EvalSurv(surv, y_test[0], y_test[1], censor_surv='km')
    ci = ev.concordance_td()
    time_grid = np.linspace(y_test[0].min(), y_test[0].max() - 1, 1000)
    ibs = ev.integrated_brier_score(time_grid)

    # Obtain Confidence Intervals for CI results
    conf_int_ci = bern_conf_interval(len(y_test[0]), ci)
    conf_int_ibs = bern_conf_interval(len(y_test[0]), ibs, ibs=True)

    # Save results
    results = {'ci': conf_int_ci, 'ibs': conf_int_ibs}
    with open(output_dir + 'results_fold_' + str(fold) + '.pkl', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    print(Fore.RED + '\n\n-------- SURVIVAL ANALYSIS - STATE OF THE ART MODELS --------' + Style.RESET_ALL)

    # Environment configuration
    task = 'sota_sa'
    args = run_args(task)
    create_output_dir(task, args)

    # Set seed for reproducibility
    np.random.seed(1234)
    _ = torch.manual_seed(1234)
    if args['train']:
        print('\n----SURVIVAL ANALYSIS SOTA TRAINING----')
        for dataset_name in args['datasets']:
            print('\n\nDataset: ' + Fore.CYAN + dataset_name + Style.RESET_ALL)
            # Load and prepare data
            input_dir = args['input_dir'] + dataset_name
            real_df = pd.read_csv(input_dir + '/data.csv')
            cv_data, _ = split_cv_data(real_df, args['n_folds'], time_dist=args['time_distribution'])

            for fold in range(args['n_folds']):
                # Prepare cv data
                data = cv_data[fold]
                df_train = data[0].astype('float32')
                df_test = data[2].astype('float32')
                get_target = lambda df: (df['time'].values, df['event'].values)
                x_train = np.array(df_train.drop(['time', 'event'], axis=1))
                y_train = get_target(df_train)
                x_test = np.array(df_test.drop(['time', 'event'], axis=1))
                y_test = get_target(df_test)

                for model in args['sota_models']:
                    output_dir = args['sota_output_dir'] + dataset_name + '/' + model + '/' + str(args['n_folds']) \
                                 + '_folds/'

                    # Train model
                    print('\nModel: ' + Fore.RED + model + Style.RESET_ALL)
                    train(args, df_train, df_test, get_target, x_train, y_train, x_test, y_test, output_dir, model,
                          fold, args['early_stop'])

    # Show results
    print('\n\n----SURVIVAL ANALYSIS SOTA RESULTS----')
    for dataset_name in args['datasets']:
        print('\nDataset: ' + Fore.CYAN + dataset_name + Style.RESET_ALL)
        for model in args['sota_models']:
            ci_sum = 0.0
            ibs_sum = 0.0
            print('Model: ' + Fore.RED + model + Style.RESET_ALL)
            for fold in range(args['n_folds']):
                # Check if validation has been done and saved
                output_dir = args['sota_output_dir'] + dataset_name + '/' + model + '/' + str(
                    args['n_folds']) + '_folds/' + 'results_fold_' + str(fold) + '.pkl'
                results = check_file(output_dir, 'Results file for model does not exist.')
                ci_sum += results['ci'][1]
                ibs_sum += results['ibs'][1]

            print('Average C-index from folds: ' + str(format(ci_sum / args['n_folds'], '.2f')))
            print('Average IBS from folds: ' + str(format(ibs_sum / args['n_folds'], '.3f')))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
