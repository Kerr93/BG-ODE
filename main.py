import tqdm
import pickle
import copy
import data_utils
import argparse

from sklearn.utils import shuffle

import simulation
import simulation.data_utils

import real
import real.data_utils

import numpy as np

from model import Model

import torch
import torch.optim as optim
import torch.nn.functional as F

import pandas as pd

from data_utils import state_dim

from train_utils import train, train_union

import ipdb
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
# DEVICE = 'cuda:0'

fea_cols = [
    'finger_stick', 'meal', 'filled_meal', 'basal', 'bolus'
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lookback', type=int, default=12)
    parser.add_argument('--lookahead', type=int, default=6)
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--frac', type=float, default=1.0,
                        help='fraction of real-world data')
    args = parser.parse_args()

    # data-driven model & expert model
    if args.exp in ['expert', 'neural']:
        if args.exp == 'expert':
            inputs, outputs = simulation.data_utils._get_inputs_outputs(
                lookback=args.lookback,
                lookahead=args.lookahead,
                fea_cols=fea_cols,
                filepath='./simulation/data'
            )
            input_dim = inputs.shape[-1] - 1 - state_dim
            use_state = True

        elif args.exp == 'neural':
            inputs, outputs = real.data_utils._get_inputs_outputs(
                lookback=args.lookback,
                lookahead=args.lookahead,
                fea_cols=fea_cols,
                filepath='./real/data/processed',
                frac=args.frac
            )
            input_dim = inputs.shape[-1] - 1
            use_state = False

        train_dataloader, test_dataloader = data_utils._build_data_loader(
            inputs, outputs
        )

        model, res = train(
            args, input_dim, train_dataloader, test_dataloader, use_state=use_state
        )

    elif args.exp == 'data_aug':
        inputs, outputs = real.data_utils._get_inputs_outputs(
            lookback=args.lookback,
            lookahead=args.lookahead,
            fea_cols=fea_cols,
            filepath='./real/data/processed',
            frac=args.frac
        )

        inputs_aug, outputs_aug = simulation.data_utils._get_inputs_outputs(
            lookback=args.lookback,
            lookahead=args.lookahead,
            fea_cols=fea_cols,
            filepath='./simulation/data'
        )

        input_dim = inputs.shape[-1] - 1

        train_dataloader, test_dataloader = data_utils._build_data_loader_aug(
            inputs=inputs,
            outputs=outputs,
            inputs_aug=inputs_aug,
            outputs_aug=outputs_aug
        )

        model, res = train(
            args, input_dim, train_dataloader, test_dataloader, use_state=True
        )

    elif args.exp == 'fine_tune':
        expert_model = torch.load(f'./results/model/expert.model')

        inputs, outputs = real.data_utils._get_inputs_outputs(
            lookback=args.lookback,
            lookahead=args.lookahead,
            fea_cols=fea_cols,
            filepath='./real/data/processed',
            frac=args.frac
        )

        input_dim = inputs.shape[-1] - 1

        train_dataloader, test_dataloader = data_utils._build_data_loader(
            inputs=inputs,
            outputs=outputs,
        )

        model, res = train(
            args, input_dim, train_dataloader, test_dataloader, use_state=False, base_model=expert_model, disable_train=True
        )

    elif args.exp == 'union':
        expert_model = torch.load(f'./results/model/expert.model')

        inputs, outputs = real.data_utils._get_inputs_outputs(
            lookback=args.lookback,
            lookahead=args.lookahead,
            fea_cols=fea_cols,
            filepath='./real/data/processed',
            frac=args.frac
        )

        input_dim = inputs.shape[-1] - 1

        train_dataloader, test_dataloader = data_utils._build_data_loader(
            inputs=inputs,
            outputs=outputs,
        )

        model, res = train_union(
            args, input_dim, train_dataloader, test_dataloader, expert_model=expert_model
        )

