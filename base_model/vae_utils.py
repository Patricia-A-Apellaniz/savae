# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 16/03/2023

# Import libraries
import torch

import numpy as np


# -----------------------------------------------------------
#                      TRAINING PROCESS
# -----------------------------------------------------------

def check_nan_inf(values, log):
    if torch.isnan(values).any().detach().cpu().tolist() or torch.isinf(values).any().detach().cpu().tolist():
        raise RuntimeError('NAN DETECTED. ' + str(log))
    return


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.stop = False

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
        if self.counter >= self.patience:
            self.stop = True


def get_dim_from_type(feat_dists):
    return sum(d[1] for d in feat_dists)  # Returns the number of parameters needed
    # to base_model the distributions in feat_dists


def get_activations_from_types(x, feat_dists, min_val=1e-3, max_std=10.0, max_alpha=2, max_k=1000.0):
    # Ancillary function that gives the correct torch activations for each data distribution type
    # Example of type list: [('bernoulli', 1), ('gaussian', 2), ('categorical', 5)]
    # (distribution, number of parameters needed for it)
    index_x = 0
    out = []
    for index_type, type in enumerate(feat_dists):
        dist, num_params = type
        if dist == 'gaussian':
            out.append(torch.tanh(x[:, index_x, np.newaxis]) * 5)  # Mean: from -inf to +inf
            out.append((torch.sigmoid(x[:, index_x + 1, np.newaxis]) * (max_std - min_val) + min_val) / (10 * max_std))
        elif dist == 'weibull':
            out.append((torch.sigmoid(x[:, index_x, np.newaxis]) * (max_std - min_val) + min_val) / max_std * max_alpha)
            out.append(
                (torch.sigmoid(x[:, index_x + 1, np.newaxis]) * (max_std - min_val) + min_val) / max_std * max_k)
            # K: (min_val, max_k + min_val)
        elif dist == 'bernoulli':
            out.append(torch.sigmoid(x[:, index_x, np.newaxis]) * (1.0 - 2 * min_val) + min_val)
            # p: (min_val, 1-min_val)
        elif dist == 'categorical':  # Softmax activation: NANs appear if values are close to 0,
            # so use min_val to prevent that
            vals = torch.tanh(x[:, index_x: index_x + num_params]) * 10.0  # Limit the max values
            out.append(torch.softmax(vals, dim=1))  # probability of each categorical value
            check_nan_inf(out[-1], 'Categorical distribution')
        else:
            raise NotImplementedError('Distribution ' + dist + ' not implemented')
        index_x += num_params
    return torch.cat(out, dim=1)


def linear_rate(epoch, n_epochs, ann_prop):
    # Adjust the KL parameter with a constant annealing rate
    if epoch >= n_epochs * ann_prop - 1:
        factor = 1
    else:
        factor = 1 / (ann_prop * n_epochs) * epoch  # Linear increase
    return factor


def cyclic_rate(epoch, n_epochs, n_cycles, ann_prop=0.5):
    # Based on the paper: Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing
    epochs_per_cycle = int(np.ceil(n_epochs / n_cycles))
    return linear_rate(epoch % epochs_per_cycle, epochs_per_cycle, ann_prop)


def triangle_rate(epoch, n_epochs_total, n_epochs_init, init_val, ann_prop):
    if epoch <= n_epochs_init:
        return init_val
    else:
        return linear_rate(epoch - n_epochs_init, n_epochs_total - n_epochs_init, ann_prop)


# -----------------------------------------------------------
#                      RECONSTRUCTION PROCESS
# -----------------------------------------------------------

def sample_from_dist(params, feat_dists):  # Get samples from the base_model
    i = 0
    out_vals = []
    for type in feat_dists:
        dist, num_params = type
        if dist == 'gaussian':
            x = np.random.normal(loc=params[:, i], scale=params[:, i + 1])
            out_vals.append(x)
        elif dist == 'weibull':
            out_vals.append(np.random.weibull(a=params[:, i]) * params[:, i + 1])
        elif dist == 'bernoulli':
            out_vals.append(np.random.binomial(n=np.ones_like(params[:, i]).astype(int), p=params[:, i]))
        elif dist == 'categorical':
            aux = np.zeros((params.shape[0],))
            for j in range(params.shape[0]):
                aux[j] = np.random.choice(np.arange(num_params), p=params[j, i: i + num_params])  # Choice
                # takes p as vector only: we must proceed one by one
            out_vals.append(aux)
        i += num_params
    return np.array(out_vals).T
