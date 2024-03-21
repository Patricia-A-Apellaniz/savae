# Author: Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 13/09/2023


# Packages to import
import torch

import numpy as np
import pandas as pd

from base_model.vae_model import VariationalAutoencoder
from validation import obtain_c_index, bern_conf_interval
from base_model.vae_utils import check_nan_inf, sample_from_dist, triangle_rate
from base_model.vae_modules import Decoder, LogLikelihoodLossWithCensoring


class SAVAE(VariationalAutoencoder):
    def __init__(self, params):
        # Initialize Savae parameters and modules
        super(SAVAE, self).__init__(params)

        # Parameters
        self.time_dist = params['time_dist']
        self.max_t = 2 * params['max_t']

        # Add another decoder to architecture (time prediction module)
        self.Time_Decoder = Decoder(latent_dim=self.latent_dim,
                                    hidden_size=self.hidden_size,
                                    feat_dists=[self.time_dist],
                                    max_k=self.max_t)

        # Define losses
        self.time_loss = LogLikelihoodLossWithCensoring(self.time_dist)

    def feed_forward(self, input_data):
        out_params = self(input_data)
        time_params = self.Time_Decoder(out_params['z'])
        check_nan_inf(time_params, 'Time Decoder')
        out = {'z': out_params['z'], 'cov_params': out_params['cov_params'], 'time_params': time_params,
               'latent_params': out_params['latent_params']}
        return out

    def predict_time(self, x, device=torch.device('cpu')):
        cov = x.drop(['time', 'event'], axis=1)
        out_params = self.predict(cov, device)
        time_params = self.Time_Decoder(out_params['z'])

        # Sample covariate and time values
        cov_params = out_params['cov_params'].detach().cpu().numpy()
        cov_samples = sample_from_dist(cov_params, self.feat_distributions)
        time_params = time_params.detach().cpu().numpy()
        time_samples = sample_from_dist(time_params, [self.time_dist])

        out_data = {'z': out_params['z'].detach().cpu().numpy(),
                    'cov_params': cov_params,
                    'cov_samples': cov_samples,
                    'time_params': time_params,
                    'time_samples': time_samples,
                    'latent_params': [l.detach().cpu().numpy() for l in out_params['latent_params']]}

        return out_data

    def calculate_risk(self, time_train, x_val, censor_val):
        # Calculate risk (CDF) at the end of batch training
        times = np.unique(time_train)
        out_data = self.predict_time(x_val)
        pred_risk = np.zeros([out_data['time_params'].shape[0], len(times)])
        for sample in range(pred_risk.shape[0]):
            if self.time_dist[0] == 'weibull':
                alpha = out_data['time_params'][sample, 0]
                lam = out_data['time_params'][sample, 1]
                pred_risk[sample, :] = 1 - np.exp(-np.power(times / lam, alpha))
            else:
                raise RuntimeError('Unknown time distribution to compute risk')

        # Compute C-index and IBS
        ci, ibs = obtain_c_index(pd.DataFrame(1 - pred_risk.T, index=times), np.array(x_val['time']), censor_val)
        ci_confident_intervals = bern_conf_interval(len(np.array(x_val['time'])), ci)
        ibs_confident_intervals = bern_conf_interval(len(np.array(x_val['time'])), ibs, ibs=True)

        return ci_confident_intervals, ibs_confident_intervals

    def fit_epoch(self, data, optimizer, batch_size=64, device=torch.device('cpu')):
        epoch_results = {'loss_tr': 0.0, 'loss_va': 0.0, 'kl_tr': 0.0, 'kl_va': 0.0, 'll_cov_tr': 0.0, 'll_cov_va': 0.0,
                         'll_time_tr': 0.0, 'll_time_va': 0.0, 'ci_va': 0.0, 'ibs_va': 0.0}
        cov_train, mask_train, time_train, censor_train, cov_val, mask_val, time_val, censor_val = data

        # Train epoch
        cov_val = torch.from_numpy(cov_val).to(device).float()
        mask_val = torch.from_numpy(mask_val).to(device).float()
        time_val = torch.from_numpy(time_val).to(device).float()
        censor_val = torch.from_numpy(censor_val).to(device).float()
        n_batches = int(np.ceil(cov_train.shape[0] / batch_size).item())
        for batch in range(n_batches):
            # Get (X, y) of the current mini batch/chunk
            index_init = batch * batch_size
            index_end = min(((batch + 1) * batch_size, cov_train.shape[
                0]))  # Use the min to prevent errors due to samples being smaller than batch_size
            cov_train_batch = cov_train[index_init: index_end]
            mask_train_batch = mask_train[index_init: index_end]
            time_train_batch = time_train[index_init: index_end]
            censor_train_batch = censor_train[index_init: index_end]

            self.train()
            cov_train_batch = torch.from_numpy(cov_train_batch).to(device).float()
            mask_train_batch = torch.from_numpy(mask_train_batch).to(device).float()
            time_train_batch = torch.from_numpy(time_train_batch).to(device).float()
            censor_train_batch = torch.from_numpy(censor_train_batch).to(device).float()

            # Generate output params
            out = self.feed_forward(cov_train_batch)

            # Compute losses
            optimizer.zero_grad()
            loss_kl = self.latent_space.kl_loss(out['latent_params'])
            loss_cov = self.rec_loss(out['cov_params'], cov_train_batch, mask_train_batch)
            loss_time = self.time_loss(out['time_params'], time_train_batch, censor_train_batch)
            loss = loss_kl + loss_cov + loss_time
            loss.backward()
            optimizer.step()

            # Save data
            epoch_results['loss_tr'] += loss.item()
            epoch_results['kl_tr'] += loss_kl.item()
            epoch_results['ll_cov_tr'] += loss_cov.item()
            epoch_results['ll_time_tr'] += loss_time.item()

            # Validation step
            self.eval()
            with torch.no_grad():
                out = self.feed_forward(cov_val)
                loss_kl = self.latent_space.kl_loss(out['latent_params'])
                loss_cov = self.rec_loss(out['cov_params'], cov_val, mask_val)
                loss_time = self.time_loss(out['time_params'], time_val, censor_val)
                loss = loss_kl + loss_cov + loss_time

                # Save data
                epoch_results['loss_va'] += loss.item()
                epoch_results['kl_va'] += loss_kl.item()
                epoch_results['ll_cov_va'] += loss_cov.item()
                epoch_results['ll_time_va'] += loss_time.item()

        if self.early_stop:
            self.early_stopper.early_stop(epoch_results['loss_va'])

        return epoch_results

    def fit(self, data, train_params):
        training_stats = {'loss_tr': [], 'loss_va': [], 'kl_tr': [], 'kl_va': [], 'll_cov_tr': [], 'll_cov_va': [],
                          'll_time_tr': [], 'll_time_va': [], 'ci_va': [], 'ibs_va': []}

        optimizer = torch.optim.Adam(self.parameters(), lr=train_params['lr'])

        epochs_ci = []
        epochs_ibs = []
        train_losses = []
        valid_losses = []
        best_model = None
        best_loss = np.inf
        best_ci = 0.0
        best_ibs = 1.0
        for epoch in range(train_params['n_epochs']):
            self.kl_w = triangle_rate(epoch % 50, n_epochs_total=50, n_epochs_init=25, init_val=1 / 10000, ann_prop=0.5)
            self.cov_w = triangle_rate(epoch % 50, n_epochs_total=50, n_epochs_init=15, init_val=1 / 10000,
                                       ann_prop=0.5)

            # Configure input data and missing data mask
            x_train, mask_train, x_val, mask_val = data
            time_train = np.array(x_train.loc[:, 'time'])
            time_val = np.array(x_val.loc[:, 'time'])
            censor_train = np.array(x_train.loc[:, 'event'])
            censor_val = np.array(x_val.loc[:, 'event'])
            cov_train = np.array(x_train.drop(['time', 'event'], axis=1))
            cov_val = np.array(x_val.drop(['time', 'event'], axis=1))
            mask_train = np.array(mask_train.drop(['time', 'event'], axis=1))
            mask_val = np.array(mask_val.drop(['time', 'event'], axis=1))
            assert mask_train.shape == cov_train.shape
            assert mask_val.shape == cov_val.shape
            ep_data = cov_train, mask_train, time_train, censor_train, cov_val, mask_val, time_val, censor_val

            epoch_results = self.fit_epoch(ep_data, optimizer, batch_size=train_params['batch_size'],
                                           device=train_params['device'])

            # Calculate metrics
            epoch_results['ci_va'], epoch_results['ibs_va'] = self.calculate_risk(time_train, x_val, censor_val)

            # Save training stats
            train_losses.append(epoch_results['loss_tr'])
            valid_losses.append(epoch_results['loss_va'])
            epochs_ci.append(epoch_results['ci_va'])
            epochs_ibs.append(epoch_results['ibs_va'])
            for key in epoch_results.keys():
                training_stats[key].append(epoch_results[key])

            if epoch % 50 == 0:
                print('Iteration = ', epoch,
                      '; train loss = ', '{:.2f}'.format(epoch_results['loss_tr']),
                      '; val loss = ', '{:.2f}'.format(epoch_results['loss_va']),
                      '; ll_cov_tr = ', '{:.2f}'.format(np.sum(epoch_results['ll_cov_tr'])),
                      '; ll_cov_va = ', '{:.2f}'.format(np.sum(epoch_results['ll_cov_va'])),
                      '; kl_tr = ', '{:.2f}'.format(epoch_results['kl_tr']),
                      '; kl_va = ', '{:.2f}'.format(epoch_results['kl_va']),
                      '; ll_time_tr = ', '{:.2f}'.format(np.sum(epoch_results['ll_time_tr'])),
                      '; ll_time_va = ', '{:.2f}'.format(np.sum(epoch_results['ll_time_va'])),
                      '; C-index = ', '{:.4f}'.format(np.mean(epoch_results['ci_va'])))

            # Save best model
            if epoch_results['loss_va'] < best_loss:
                best_loss = epoch_results['loss_va']
                best_model = self.state_dict()
                best_ci = epoch_results['ci_va']
                best_ibs = epoch_results['ibs_va']

            if self.early_stop and self.early_stopper.stop:
                print('[INFO] Training early stop')
                break

        # Set the best model as the current model
        self.load_state_dict(best_model)
        # Set the best ci and ibs in the last epoch of the validation, just to ensure that they are taken into account
        # when selecting the best seed after
        training_stats['ci_va'].append(best_ci)
        training_stats['ibs_va'].append(best_ibs)

        return training_stats
