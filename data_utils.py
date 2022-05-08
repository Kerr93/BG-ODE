import torch
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

from sklearn.model_selection import train_test_split

state_dim = 13


def _build_data_loader(inputs, outputs, batch_size=1):
    def _to_dataloader(_inputs, _outputs, shuffle):
        dataset = TensorDataset(
            torch.from_numpy(_inputs.astype('float32')),
            torch.from_numpy(_outputs.astype('float32'))
        )

        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader

    train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(
        inputs, outputs, test_size=0.3, shuffle=False
    )

    trainloader = _to_dataloader(train_inputs, train_outputs, shuffle=True)
    testloader = _to_dataloader(test_inputs, test_outputs, shuffle=False)

    return trainloader, testloader


def _build_data_loader_aug(inputs, outputs, inputs_aug, outputs_aug, batch_size=1, shuffle=False):
    def _to_dataloader(_inputs, _outputs, shuffle):
        dataset = TensorDataset(
            torch.from_numpy(_inputs.astype('float32')),
            torch.from_numpy(_outputs.astype('float32'))
        )

        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )

        return dataloader

    inputs_nan = np.empty((inputs.shape[0], inputs.shape[1], state_dim))
    inputs_nan[:] = np.nan

    outputs_nan = np.empty((outputs.shape[0], outputs.shape[1], state_dim))
    outputs_nan[:] = np.nan

    inputs = np.concatenate([inputs, inputs_nan], axis=-1)
    outputs = np.concatenate([outputs, outputs_nan], axis=-1)

    train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(
        inputs, outputs, test_size=0.3, shuffle=False
    )

    # inputs_aug = inputs_aug[:, :, :-state_dim]
    # outputs_aug = outputs_aug[:, :, :-state_dim]

    train_inputs = np.concatenate([train_inputs, inputs_aug], axis=0)
    train_outputs = np.concatenate([train_outputs, outputs_aug], axis=0)

    trainloader = _to_dataloader(train_inputs, train_outputs, shuffle=True)
    testloader = _to_dataloader(test_inputs, test_outputs, shuffle=False)

    return trainloader, testloader
