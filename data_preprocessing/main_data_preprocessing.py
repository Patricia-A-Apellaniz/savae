# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 05/09/2023


# Packages to import
import os
import sys

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from utils import run_args, create_output_dir
from sa_datasets import preprocess_gbsg, preprocess_whas, preprocess_flchain, preprocess_nwtco, \
    preprocess_metabric, preprocess_pneumon, preprocess_std, preprocess_pbc, preprocess_support


def preprocess_data(dataset_name, args):
    # Deepsurv and Deephit datasets
    if dataset_name == 'whas':
        data = preprocess_whas(args)
    elif dataset_name == 'support':
        data = preprocess_support(args)
    elif dataset_name == 'gbsg':
        data = preprocess_gbsg()
    elif dataset_name == 'metabric':
        data = preprocess_metabric(args)
    # R datasets
    elif dataset_name == 'flchain':
        data = preprocess_flchain()
    elif dataset_name == 'nwtco':
        data = preprocess_nwtco()
    elif dataset_name == 'std':
        data = preprocess_std(args)
    elif dataset_name == 'pbc':
        data = preprocess_pbc(args)
    elif dataset_name == 'pneumon':
        data = preprocess_pneumon(args)
    else:
        raise RuntimeError('Dataset not recognized')
    return data


def main():
    print('\n\n-------- DATA PREPROCESSING  --------')

    # Environment configuration
    task = 'data_preprocessing'
    args = run_args(task)
    create_output_dir(task, args)

    # Preprocess data
    for dataset_name in args['datasets']:
        # Load dataset
        data = preprocess_data(dataset_name, args)

        # Save data
        pd.DataFrame(data, columns=data.columns).to_csv(args['output_dir'] + dataset_name + '/' + 'data.csv',
                                                        index=False)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
