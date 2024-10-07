# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 29/11/2023


# Packages to import
import os
import sys

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from tabulate import tabulate
from colorama import Fore, Style
from scipy.stats import rankdata
from scipy.stats import ttest_1samp
from utils import run_args, check_file
from statsmodels.stats.multitest import multipletests


def main():
    print(Fore.RED + '\n\n-------- SURVIVAL ANALYSIS - RESULTS --------' + Style.RESET_ALL)

    task = 'display_sa_results'
    args = run_args(task)
    msg = 'Results file not found. Please, run survival_analysis/main_savae.py and ' \
          'survival_analysis/main_sota.py (coxph, deepsurv and deephit) first.'

    for metric in ['ci', 'ibs']:
        mrr = []
        min_max_table = []
        p_values_table = []
        holm_p_values_table = []
        round_format = '.2f' if metric == 'ci' else '.3f'
        for dataset_name in args['datasets']:
            mrr_dataset = []
            min_max_dataset = [dataset_name]
            p_values_dataset = [dataset_name]
            sota_folds = []

            # Obtain sota avg metrics
            for model in args['sota_models']:
                # Load results
                sota_model_dir = args['sota_output_dir'] + dataset_name + '/' + model + '/'
                folds = []
                for fold in range(args['n_folds']):
                    fold_dir = sota_model_dir + str(args['n_folds']) + '_folds/results_fold_' + str(fold) + '.pkl'
                    fold_results = check_file(fold_dir, msg)
                    folds.append(fold_results[metric])
                sota_folds.append(folds)
                mrr_dataset.append(np.mean([folds[i][1] for i in range(args['n_folds'])]))

                # Min-max intervals
                min_values = [folds[i][0] for i in range(args['n_folds'])]
                mean_values = [folds[i][1] for i in range(args['n_folds'])]
                max_values = [folds[i][2] for i in range(args['n_folds'])]
                min_max_dataset.extend(['(' + str(format(np.min(min_values), round_format)) + ' - ' + str(
                    format(np.mean(mean_values), round_format)) + ' - ' + str(
                    format(np.max(max_values), round_format)) + ')'])

            # Obtain savae metrics confidence intervals
            file_dir = args['output_dir'] + dataset_name + '/' + str(args['n_folds']) + '_folds/best_results.pkl'
            savae_results = check_file(file_dir, msg)

            # Min-max intervals
            best_values = []
            for fold in range(args['n_folds']):
                dict_key = 'best_' + metric + 's' if metric == 'ci' else 'best_' + metric
                best_values.extend(
                    [savae_results[dict_key][fold][seed] for seed in range(len(savae_results[dict_key][fold]))])

            # Min-max intervals
            min_values = [best_values[i][0] for i in range(len(best_values))]
            mean_values = [best_values[i][1] for i in range(len(best_values))]
            max_values = [best_values[i][2] for i in range(len(best_values))]
            min_max_dataset.extend(['(' + str(format(np.min(min_values), round_format)) + ' - ' + str(
                format(np.mean(mean_values), round_format)) + ' - ' + str(
                format(np.max(max_values), round_format)) + ')'])

            mrr_dataset.append(np.mean(mean_values))

            # ttest_1samp to compare savae results with sota means
            for i, model in enumerate(args['sota_models']):
                sota_mean = np.mean([sota_folds[i][fold][1] for fold in range(args['n_folds'])])
                t_test_results = ttest_1samp(mean_values, popmean=sota_mean, alternative='greater')
                # Threshold 0.05. If p_value obtained is less than threshold,
                # we reject null hypothesis: mean of savae results is equal to sota mean
                p_values_dataset.extend([t_test_results.pvalue])

            mrr.append(mrr_dataset)
            min_max_table.append(min_max_dataset)
            p_values_table.append(p_values_dataset)
            holm_p_values_table.append([p_values_dataset[0]] + list(multipletests(p_values_dataset[1:], alpha=0.05, method='holm')[1]))


        # Show metric (min-max) table results
        headers = ['DATASET', 'COXPH', 'DEEPSURV', 'DEEPHIT', 'SAVAE']
        min_max_table.insert(0, headers)
        print('\n\n')
        print(metric + ' (min-avg-max) RESULTS')
        print(tabulate(min_max_table, headers='firstrow', tablefmt='grid'))

        # Show metric (p_values) table results
        headers = ['DATASET', 'COXPH', 'DEEPSURV', 'DEEPHIT']
        p_values_table.insert(0, headers)
        print('\n\n')
        print(metric + ' (p_values) RESULTS')
        print(tabulate(p_values_table, headers='firstrow', tablefmt='grid'))

        # Show p_values adjustment based on Holm method table results
        headers = ['DATASET', 'COXPH', 'DEEPSURV', 'DEEPHIT']
        holm_p_values_table.insert(0, headers)
        print('\n\n')
        print(metric + ' (adjusted p_values) RESULTS')
        print(tabulate(holm_p_values_table, headers='firstrow', tablefmt='grid'))
        print(tabulate(holm_p_values_table, headers='firstrow', tablefmt='latex'))

        # Calculate MRR
        ranks = []
        for row in mrr:
            if metric == 'ci':
                ranks.append(rankdata(-np.array([np.round(value, 2) for value in row]), method='dense'))
            else:
                ranks.append(rankdata(np.array([np.round(value, 3) for value in row]), method='dense'))
        # Second, calculate MRR for each model
        mrr_vals = []
        for model in range(len(ranks[0])):
            mrr_vals.append(0.0)
            for ranking in range(len(ranks)):
                mrr_vals[model] += 1 / (ranks[ranking][model])
            mrr_vals[model] /= len(ranks)

        # Print MRR
        headers = ['COXPH', 'DEEPSURV', 'DEEPHIT', 'SAVAE']
        print('\n\n')
        print(metric + ' MRR RESULTS')
        print(tabulate([mrr_vals], headers=headers, tablefmt='grid'))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
