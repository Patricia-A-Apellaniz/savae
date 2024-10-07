# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 06/09/2023


# Packages to import
import math

import numpy as np
import pandas as pd

from scipy import stats
from sklearn.svm import SVR
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split, KFold


# -----------------------------------------------------------
#                   DATA IMPUTATION
# -----------------------------------------------------------

def zero_imputation(data):
    imp_data = data.copy()
    imp_data = imp_data.fillna(0)
    return imp_data


def mice_imputation(data, model='bayesian'):
    imp_data = data.copy()
    if model == 'bayesian':
        clf = BayesianRidge()
    elif model == 'svr':
        clf = SVR()
    else:
        raise RuntimeError('MICE imputation base_model not recognized')
    imp = IterativeImputer(estimator=clf, verbose=2, max_iter=30, tol=1e-10, imputation_order='roman')
    imp_data.iloc[:, :] = imp.fit_transform(imp_data)
    return imp_data


def statistics_imputation(data):
    imp_data = data.copy()
    n_samp, n_feat = imp_data.shape
    for i in range(n_feat):
        values = data.iloc[:, i].values
        if any(pd.isnull(values)):
            no_nan_values = values[~pd.isnull(values)]
            if values.dtype in [object, str] or no_nan_values.size <= 2 or np.amin(
                    np.equal(np.mod(no_nan_values, 1), 0)):
                stats_value = stats.mode(no_nan_values, keepdims=True)[0][0]
            else:
                stats_value = no_nan_values.mean()
            imp_data.iloc[:, i] = [stats_value if pd.isnull(x) else x for x in imp_data.iloc[:, i]]

    return imp_data


def impute_data(df, mode='stats'):
    # If missing data exists, impute it
    if df.isna().any().any():
        # Data imputation
        if mode == 'zero':
            imp_df = zero_imputation(df)
        elif mode == 'stats':
            imp_df = statistics_imputation(df)
        else:
            imp_df = mice_imputation(df)

        # Generate missing data mask. Our model uses it to ignore missing data during training,
        # although it has been imputed
        nans = df.isna()
        mask = nans.replace([True, False], [0, 1])
    else:
        imp_df = df.copy()
        mask = np.ones((df.shape[0], df.shape[1]))
        mask = (pd.DataFrame(mask, columns=imp_df.columns)).astype(int)
    return imp_df, mask


# -----------------------------------------------------------
#                   DATA NORMALIZATION
# -----------------------------------------------------------

def get_feat_distributions(df, time=None):
    n_feat = df.shape[1]
    feat_dist = []
    for i in range(n_feat):
        if time is not None and i == n_feat - 2:  # Force time distribution
            feat_dist.append(time)
            continue
        values = df.iloc[:, i].unique()
        if len(values) == 1 and math.isnan(values[0]):
            values = np.zeros((1,))
        no_nan_values = values[~pd.isnull(values)]
        if no_nan_values.size <= 2 and all(np.sort(no_nan_values).astype(int) == np.array(
                range(no_nan_values.min().astype(int), no_nan_values.min().astype(int) + len(no_nan_values)))):
            feat_dist.append(('bernoulli', 1))
        elif np.amin(np.equal(np.mod(no_nan_values, 1), 0)):
            # Check if values are floats but don't have decimals and transform to int. They are floats because of NaNs
            if no_nan_values.dtype == 'float64':
                no_nan_values = no_nan_values.astype(int)
            if np.unique(no_nan_values).size < 30 and np.amin(no_nan_values) == 0:
                feat_dist.append(('categorical', np.max(no_nan_values) + 1))
            else:
                feat_dist.append(('gaussian', 2))
        else:
            feat_dist.append(('gaussian', 2))

    return feat_dist


def transform_data(raw_df, feat_distributions, norm_df=None):
    transf_df = norm_df.copy() if norm_df is not None else raw_df.copy()
    for i in range(raw_df.shape[1]):
        dist = feat_distributions[i][0]
        values = raw_df.iloc[:, i]
        if len(values) == 1 and math.isnan(values[0]):
            values = np.zeros((1,))
        no_nan_values = values[~pd.isnull(values)].values
        if dist == 'gaussian':
            loc = np.mean(no_nan_values)
            scale = np.std(no_nan_values)
        elif dist == 'bernoulli':
            loc = np.amin(no_nan_values)
            scale = np.amax(no_nan_values) - np.amin(no_nan_values)
        elif dist == 'categorical':
            loc = np.amin(no_nan_values)
            scale = 1  # Do not scale
        elif dist == 'weibull':
            loc = -1 if 0 in no_nan_values else 0
            scale = 0
        else:
            raise NotImplementedError('Distribution ', dist, ' not normalized!')

        if norm_df is not None:  # Denormalize
            transf_df.iloc[:, i] = (
                norm_df.iloc[:, i] * scale + loc if scale != 0
                else norm_df.iloc[:, i] + loc).astype(raw_df.iloc[:, i].dtype)
        else:  # Normalize
            transf_df.iloc[:, i] = (raw_df.iloc[:, i] - loc) / scale if scale != 0 else raw_df.iloc[:, i] - loc

    return transf_df


# -----------------------------------------------------------
#                   DATA SPLITTING
# -----------------------------------------------------------

def append_data(train_data, test_data, train_mask, test_mask, cv_data):
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    train_mask = train_mask.reset_index(drop=True)
    test_mask = test_mask.reset_index(drop=True)
    cv_data.append((train_data, train_mask, test_data, test_mask))
    return cv_data


# Function that receives data and performs a split of a specific number of folds for cross-validation.
# It returns a list of tuples with the train and test data and the mask for each fold
def split_cv_data(real_df, n_folds, time_dist=None):
    # First, impute all together
    imp_data, mask = impute_data(real_df)

    # Then, normalize all together
    feat_distributions = get_feat_distributions(imp_data, time=time_dist)
    norm_data = transform_data(imp_data, feat_distributions)

    # Finally, split
    cv_data = []
    if n_folds < 2:
        train_data, test_data, train_mask, test_mask = train_test_split(norm_data, mask, test_size=0.2,
                                                                        random_state=1234)
        cv_data = append_data(train_data, test_data, train_mask, test_mask, cv_data)
    else:
        kf = KFold(n_splits=n_folds, random_state=1234, shuffle=True)
        for train_index, test_index in kf.split(real_df):
            train_data = norm_data.iloc[train_index]
            test_data = norm_data.iloc[test_index]
            train_mask = mask.iloc[train_index]
            test_mask = mask.iloc[test_index]
            cv_data = append_data(train_data, test_data, train_mask, test_mask, cv_data)

    return cv_data, feat_distributions
