# -*- coding: utf-8 -*-
# @Time    : 2024/5/25 9:48
# @Author  : Young Zhang
# @File    : q_learning.py
# @Project : RLAlgorithms
import numpy as np


class QLearningTabular:
    def __init__(
            self,
            n_states: int,
            n_actions: int,
            discount_factor: float,
            epsilon: float,
            fitter
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.fitter = fitter
        pass

    def choose_action(self, state):
        """ do epsilon-greedy policy

        for example: if epsilon=0.1, the agent has probability of 0.9 to take actions with the highest reward expectation
        :param state: current state of the environment
        :return: action
        """
        flag = np.random.rand()
        if flag < self.epsilon:
            action = np.random.choice(self.n_actions)
        else:
            action = np.argmax(self.fitter[state, :])

    def update(self, state, action, reward, next_state):
        # set the next_state with the highest expectation as the gole for this update
        target = reward + self.discount_factor * np.max(self.fitter[next_state, :])
        self.fitter[state, action] = (1 - self.al)
