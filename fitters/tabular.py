# -*- coding: utf-8 -*-
# @Time    : 2024/5/25 10:11
# @Author  : Young Zhang
# @File    : tabular.py
# @Project : RLAlgorithms
import numpy as np
from typing import Literal

INIT_METHOD_TYPE = Literal["normal", "zero"]


class Table:
    def __init__(self, n_state, n_action, init_method: INIT_METHOD_TYPE):
        self.n_state = n_state
        self.n_action = n_action
        self.init_method = init_method
        self.handler = {
            "normal": np.random.normal(loc=0, scale=1, size=(n_state, n_action)),
            "zero": np.zeros(shape=(n_state, n_action))
        }

    def get_fitter(self):
        return self.handler[self.init_method]
