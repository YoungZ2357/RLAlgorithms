# -*- coding: utf-8 -*-
# @Time    : 2024/5/26 20:53
# @Author  : Qingyang Zhang
# @File    : basic_network.py
# @Project : RLAlgorithms
import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(
            self,
            layer_sizes: list,
            activations
    ):
        super(MLP, self).__init__()
        self.fc = self._create_layers(layer_sizes, activations)

    @staticmethod
    def _create_layers(layer_sizes, activations):
        layers = list()
        n_hidden = len(layer_sizes) - 2
        if isinstance(activations, nn.Module):
            activations = [activations] * n_hidden

        if len(activations) != n_hidden:
            raise ValueError("The number of activations must be equal to the number of hidden layers")
        for idx in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[idx], layer_sizes[idx + 1]))
            if idx < len(layer_sizes) - 2:
                layers.append(activations[idx])
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


class LSTM(nn.Module):
    def __init__(
            self,
            n_layers,
            input_dim,
            hidden_dim,
            output_dim
    ):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_dim=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True
        )
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        tmp, _ = self.rnn(x)
        out = self.fc(tmp[:, -1, :])
        return out


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

    def forward(self):
        pass
# if __name__ == "__main__":
#     sizes = [2, 16, 16, 16, 4]
#     act = nn.SiLU()
#     net = MLP(layer_sizes=sizes, activations=act)
#     sample = [12, 21]
#     sample = torch.FloatTensor(sample)
#     r = net.forward(sample)
#     print(r)
#     print(net)
