# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 06/09/2023


# Packages to import
import numpy as np

import matplotlib.pyplot as plt

from pycox.evaluation import EvalSurv
from statsmodels.stats.proportion import proportion_confint

# This warning type is removed due to pandas future warnings
# https://github.com/havakv/pycox/issues/162. Incompatibility between pycox and pandas' new version
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


# ------------------------------------------------------------------------------------------------------
#                                     UTILS FUNCTIONS
# ------------------------------------------------------------------------------------------------------

def bern_conf_interval(n, mean, ibs=False):
    # Confidence interval
    ci_bot, ci_top = proportion_confint(count=mean * n, nobs=n, alpha=0.1, method='beta')
    if mean < 0.5 and not ibs:
        ci_bot_2 = 1 - ci_top
        ci_top = 1 - ci_bot
        ci_bot = ci_bot_2
        mean = 1 - mean

    return np.round(ci_bot, 4), mean, np.round(ci_top, 4)


def plot_model_losses(train_loss, val_loss, fig_path, title, x_label='Epochs'):
    plt.figure(figsize=(15, 15))
    plt.semilogy(train_loss, label='Train')
    plt.semilogy(val_loss, label='Valid')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.xlabel(x_label)
    plt.savefig(fig_path)
    # plt.show()
    plt.close()


# ------------------------------------------------------------------------------------------------------
#                               VALIDATION FUNCTIONS
# ------------------------------------------------------------------------------------------------------
def obtain_c_index(surv_f, time, censor):
    # Evaluate using PyCox c-index
    ev = EvalSurv(surv_f, time.flatten(), censor.flatten(), censor_surv='km')
    ci = ev.concordance_td()

    # Obtain also ibs
    time_grid = np.linspace(time.min(), time.max(), 100)
    ibs = ev.integrated_brier_score(time_grid)
    return ci, ibs
