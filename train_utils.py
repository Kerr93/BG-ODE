from cgi import test
from cmath import isnan
import copy
import tqdm
import logging

import pickle

import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd

from data_utils import state_dim

from model import Model
import ipdb


def _get_loss_with_nan(y, y_hat):
    mask = 1 - torch.isnan(y).float()

    y = torch.nan_to_num(y)
    loss = F.mse_loss(y, y_hat, reduction='none')
    loss = torch.sum(loss * mask) / (torch.sum(mask) + 1e-3)

    return loss


def run_one_epoch(model, dataloader, optimizer=None, desc='', do_interpolation=False, use_state=False, **kwargs):
    running_loss = 0.0

    res = {
        'df': []
    }

    def _save_to_dataframe(xy, predictions, label, interpolations, patient_id):
        ts = xy.select(-1, 0).squeeze(dim=0)
        ts = ts.data.cpu().numpy()

        predictions = predictions
        label = label.squeeze(dim=0)

        label = label.data.cpu().numpy()
        predictions = predictions.data.cpu().numpy()
        interpolations = interpolations.data.cpu().numpy()

        infer_start_tm = ts[kwargs['lookback']]

        ts = ts[1:]

        finger_df = pd.DataFrame(
            {
                'ts': ts,
                'label': label,
                'prediction': predictions
            }
        )

        cgm_df = pd.DataFrame(
            {
                'ts': np.arange(ts[0], ts[-1] + 1),
                'cgm': interpolations
            }
        )

        df = cgm_df.merge(finger_df, on='ts', how='left')
        df['infer'] = df['ts'].map(lambda x: 1 if x >= infer_start_tm else 0)
        df['patient_id'] = patient_id

        res['df'].append(df)

    with tqdm.tqdm(dataloader) as bar:
        for bid, (x, y) in enumerate(bar):
            bar.set_description(desc)

            x, y = x.cuda(), y.cuda()

            xy = torch.cat([x, y], dim=1)

            if use_state:
                xy, states = xy[:, :, :-state_dim], xy[:, :, -state_dim:]
            else:
                states = None

            # filename
            patient_id = xy.select(-1, -1).mean().item()

            xy = xy[:, :, :-1]

            # mask all the NaN values for finger_stick
            mask = 1 - torch.isnan(xy.select(-1, 1)).float()
            mask[:, x.size(1):] = 0

            label = xy.squeeze(dim=0)[:, 1]
            # remove the first point
            label = label[1:]

            if use_state:
                states = states.squeeze(dim=0)
                states = states[1:]

            if do_interpolation:
                predictions, states_hat, interpolations = model(xy, mask, True)
                _save_to_dataframe(xy, predictions, label,
                                   interpolations, patient_id)
            else:
                predictions, states_hat = model(xy, mask)

            loss = _get_loss_with_nan(label, predictions)

            if use_state and (~torch.isnan(states)).sum() > 0:
                state_loss = _get_loss_with_nan(states, states_hat)
                # loss = loss + state_loss * 0.1
                loss = loss + state_loss * 5e-3

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

            bar.set_postfix(loss=running_loss/(bid+1))

    res['loss'] = running_loss

    return res


def train(args, input_dim, train_dataloader, test_dataloader, use_state=False, base_model=None, disable_train=False):
    # input_dim: fea_cols + filename
    if base_model is None:
        model = Model(input_dim=input_dim, hid_dim=args.latent_dim)
        model.cuda()
    else:
        model = copy.deepcopy(base_model)
        model.cuda()

    # model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_loss = np.inf
    best_model = None
    best_res = None

    for eid in range(args.epochs):
        print('Epoch {}'.format(eid))

        if not disable_train:
            model.train()
            train_res = run_one_epoch(
                model, train_dataloader, optimizer, desc='Train', use_state=use_state, lookback=args.lookback
            )

        model.eval()
        test_res = run_one_epoch(
            model, test_dataloader, optimizer=None, desc='Test', do_interpolation=True, use_state=use_state, lookback=args.lookback
        )

        if disable_train:
            best_model = copy.deepcopy(model)
            best_res = test_res

            torch.save(best_model, f'./results/model/{args.exp}_{int(args.frac)}.model')
            pickle.dump(
                best_res,
                open(f'./results/pickles/result_{args.exp}_{int(args.frac)}.pkl', 'wb')
            )
            break

        if test_res['loss'] < best_loss:
            best_loss = test_res['loss']
            best_model = copy.deepcopy(model)
            best_res = test_res
            if args.exp == 'expert':
                torch.save(best_model, f'./results/model/{args.exp}.model')
                pickle.dump(
                    best_res,
                    open(f'./results/pickles/result_{args.exp}.pkl', 'wb')
                )
            else:
                torch.save(best_model, f'./results/model/{args.exp}_{int(args.frac)}.model')
                pickle.dump(
                    best_res,
                    open(f'./results/pickles/result_{args.exp}_{int(args.frac)}.pkl', 'wb')
                )

    return best_model, best_res


def run_one_epoch_union(model, dataloader, optimizer=None, desc='', do_interpolation=False, **kwargs):
    model, expert_model = model

    running_loss = 0.0

    res = {
        'df': []
    }

    def _save_to_dataframe(xy, predictions, label, interpolations, patient_id):
        ts = xy.select(-1, 0).squeeze(dim=0)
        ts = ts.data.cpu().numpy()

        predictions = predictions
        label = label.squeeze(dim=0)

        label = label.data.cpu().numpy()
        predictions = predictions.data.cpu().numpy()
        interpolations = interpolations.data.cpu().numpy()

        infer_start_tm = ts[kwargs['lookback']]

        ts = ts[1:]

        finger_df = pd.DataFrame(
            {
                'ts': ts,
                'label': label,
                'prediction': predictions
            }
        )

        cgm_df = pd.DataFrame(
            {
                'ts': np.arange(ts[0], ts[-1] + 1),
                'cgm': interpolations
            }
        )

        df = cgm_df.merge(finger_df, on='ts', how='left')
        df['infer'] = df['ts'].map(lambda x: 1 if x >= infer_start_tm else 0)
        df['patient_id'] = patient_id

        res['df'].append(df)

    with tqdm.tqdm(dataloader) as bar:
        for bid, (x, y) in enumerate(bar):
            bar.set_description(desc)

            x, y = x.cuda(), y.cuda()

            xy = torch.cat([x, y], dim=1)

            patient_id = xy.select(-1, -1).mean().item()

            xy = xy[:, :, :-1]

            # mask all the NaN values
            mask = 1 - torch.isnan(xy.select(-1, 1)).float()
            mask[:, x.size(1):] = 0

            label = xy.squeeze(dim=0)[:, 1]
            # remove the first point
            label = label[1:]

            if do_interpolation:
                predictions, states, interpolations = model(xy, mask, True)
                _save_to_dataframe(xy, predictions, label,
                                   interpolations, patient_id)
            else:
                predictions, states = model(xy, mask)
                predictions_e, states_e = expert_model(xy, mask)

            loss = _get_loss_with_nan(label, predictions)

            if optimizer is not None:
                loss_aux = _get_loss_with_nan(
                    predictions, predictions_e.detach()
                )
                loss_state = _get_loss_with_nan(states, states_e.detach())

                disc_loss = loss_aux + loss_state * 5e-3
                # disc_loss = loss_aux * 5e-3 + loss_state * 5e-3
                loss = loss + disc_loss

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

            bar.set_postfix(loss=running_loss/(bid+1))

    res['loss'] = running_loss

    return res


def train_union(args, input_dim, train_dataloader, test_dataloader, expert_model):
    model = Model(input_dim=input_dim, hid_dim=args.latent_dim)
    model.cuda()

    expert_model.cuda()
    expert_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_loss = np.inf
    best_model = None
    best_res = None

    for eid in range(args.epochs):
        print('Epoch {}'.format(eid))

        model.train()
        train_res = run_one_epoch_union(
            (model, expert_model), train_dataloader, optimizer, desc='Train', lookback=args.lookback
        )

        model.eval()
        test_res = run_one_epoch_union(
            (model, expert_model), test_dataloader, optimizer=None, desc='Test', do_interpolation=True, lookback=args.lookback
        )

        if test_res['loss'] < best_loss:
            best_loss = test_res['loss']
            best_model = copy.deepcopy(model)
            best_res = test_res

            torch.save(best_model, f'./results/model/{args.exp}_{int(args.frac)}.model')
            pickle.dump(
                best_res,
                open(f'./results/pickles/result_{args.exp}_{int(args.frac)}.pkl', 'wb')
            )

    return best_model, best_res
