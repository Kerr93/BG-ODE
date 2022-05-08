import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint

import numpy as np

from data_utils import state_dim
import ipdb


def _get_loss_with_nan(y, y_hat):
    mask = 1 - torch.isnan(y).float()

    y = torch.nan_to_num(y)
    loss = F.mse_loss(y, y_hat, reduction='none')
    loss = torch.sum(loss * mask) / torch.sum(mask)

    return loss


class ODEFunc(nn.Module):
    def __init__(self, hid_dim): 
        super().__init__()

        self.f = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, hid_dim)
        )

    def forward(self, t, x):
        return self.f(x)


class Model(nn.Module):
    def __init__(self, input_dim, hid_dim):
        super().__init__()

        self.input_dim = input_dim - 1
        self.hid_dim = hid_dim

        self.rnn = nn.GRUCell(self.input_dim, self.hid_dim)

        self.ode_func = ODEFunc(self.hid_dim)

        self.g = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, 1)
        )

        self.s = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, state_dim)
        )

    def forward(self, inputs, mask, do_interpolation=False): 
        if do_interpolation:
            assert(not self.training)
            interpolations = []

        ts = inputs.select(-1, 0).squeeze(dim=0)

        start_time = ts[0]
        ts = (ts - start_time) / 1000

        inputs = inputs.clone()
        inputs = torch.nan_to_num(inputs)

        inputs = inputs[:, :, 1:]

        hid = torch.zeros(inputs.size(0), self.hid_dim).cuda()

        predictions = []
        states = []

        for t in range(inputs.size(1)):
            prediction = self.g(hid.squeeze(dim=1)).squeeze(dim=-1) #(1)
            state = self.s(hid.squeeze(dim=1)) #(1, 13)

            if t > 0:
                # no predictio for t = 0 since we have no prior knowledge
                predictions.append(prediction)
                states.append(state)

            x = inputs.select(1, t)
            m = mask.select(1, t)

            blood, others = x.split([1, self.input_dim - 1], dim=-1)
            # replace blood with prediction for auto-regressive prediction
            blood = m * blood + (1 - m) * prediction

            x = torch.cat([blood, others], dim=-1)

            hid = self.rnn(x, hid)

            if t < inputs.size(1) - 1:
                if do_interpolation:
                    ts_span = ts[[t, t + 1]]
                    steps = np.round((ts[t + 1] - ts[t]).item() * 1e3) + 1
                    ts_span = torch.linspace(
                        ts[t], ts[t + 1], steps=int(steps)
                    )
                else:
                    ts_span = ts[[t, t + 1]]

                hid = odeint(
                    self.ode_func, hid, ts_span,
                    rtol=1e-3, atol=1e-3,
                    adjoint_options=dict(norm='seminorm')
                )

                if do_interpolation:
                    _interpolation = self.g(hid.squeeze(dim=1)).squeeze(dim=-1)
                    if t == 0:
                        _interpolation = _interpolation[[-1]]
                    else:
                        # eliminate the first point since we already know the input (i.e., x)
                        _interpolation = _interpolation[1:]

                    interpolations.append(_interpolation)
                    
                hid = hid[-1]

        predictions = torch.cat(predictions, dim=0)
        states = torch.cat(states, dim=0)

        if do_interpolation:
            interpolations = torch.cat(interpolations, dim=-1)
            return predictions, states, interpolations
        else:
            return predictions, states
