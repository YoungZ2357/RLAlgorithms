# -*- coding: utf-8 -*-
# @Time    : 2024/5/25 9:48
# @Author  : Young Zhang
# @File    : q_learning.py
# @Project : RLAlgorithms
import gymnasium
import numpy as np


class QLearningTabular:
    def __init__(
            self,
            env: gymnasium.Env,
            discount_factor: float,
            lr: float,
            epsilon: float
    ):

        self.env = env
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n

        self.fitter = np.zeros(shape=(self.n_states, self.n_actions))
        self.discount_factor = discount_factor
        self.lr = lr
        self.epsilon = epsilon



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
        return action

    def update(self, state, action, reward, next_state) -> None:
        best_action = np.argmax(self.fitter[next_state][:])
        target = float(reward) + self.discount_factor * self.fitter[next_state][best_action]

        self.fitter[state][action] += self.lr * (target - self.fitter[state][action])

    def train(self, n_episodes: int):
        for episode in range(n_episodes):
            state = self.env.reset()
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                self.update(state, action, reward, next_state)
                state = next_state

    def evaluate(self, n_episodes: int):
        total_rewards = 0
        for episode in range(n_episodes):
            state = self.env.reset()
            terminated = False
            truncated = False
            episode_reward = 0.
            while not (terminated or truncated):
                action = np.argmax(self.fitter[state])
                state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
            total_rewards += episode_reward
        return total_rewards / n_episodes


