# -*- coding: utf-8 -*-
# @Time    : 2024/5/26 20:49
# @Author  : Young Zhang
# @File    : DQN.py
# @Project : RLAlgorithms
import torch.nn as nn


class DQNAgent:
    def __init__(
            self,
            fitter: nn.Module,
            env,
            epsilon,
            discount_factor,

    ):
        self.fitter = fitter
        self.env = env
        self.epsilon = epsilon
        self.discount_factor = discount_factor


    def choose_action(self, state):
        pass

    def learn(self):
        pass
