# -*- coding: utf-8 -*-
# @Time    : 2024/5/26 16:26
# @Author  : Young Zhang
# @File    : env_namelist.py
# @Project : RLAlgorithms
import gymnasium

myenvs = list(gymnasium.envs.registry.keys())
atari = [x for x in myenvs if "ALE" in x]
print(atari)
