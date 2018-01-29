import random
import math
import sys

import numpy as np
import pandas as pd
import pickle as pkl

from pysc2.agents import base_agent
from actions.actions import *

KILL_UNIT_REWARD = 1

# Stolen from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
    def __init__(self, actions, learning_rate=0.03, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions)

    def choose_action(self, observation):
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]

            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))

            action = state_action.values.argmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.check_state_exist(s)

        q_predict = self.q_table.ix[s, a]
        q_target = r + self.gamma * self.q_table.ix[s_, :].max()

        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))

    def save_q_table(self):
        # Save the q_table for later loading.
        pkl.dump(self.q_table, open("q_table.p", "wb"))


class RLAgent(base_agent.BaseAgent):
    def __init__(self):
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.obs_spec = None
        self.action_spec = None
        self.base_top_left = False
        self.qlearn = QLearningTable(actions=list(range(len(ACTIONS.smart_actions))))
        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0
        self.previous_mineral_rate = None
        self.previous_worker_count = None
        self.previous_army_count = 0
        self.previous_barracks_count = 0
        self.previous_score = 0
        self.scores = [0]

        self.previous_action = None
        self.previous_state = None

    def reset(self):
        self.qlearn.save_q_table()
        self.episodes += 1
        print("Running episode: ", self.episodes)

    def step(self, obs):
        super(RLAgent, self).step(obs)

        if self.previous_worker_count is None and self.previous_mineral_rate is None:
            self.previous_worker_count = obs.observation['player'][6]
            self.previous_mineral_rate = obs.observation['score_cumulative'][9]

        player_y, player_x = (obs.observation['minimap'][MISC.PLAYER_RELATIVE] == MISC.PLAYER_SELF).nonzero()
        self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

        unit_type = obs.observation['screen'][MISC.UNIT_TYPE]

        depot_y, depot_x = (unit_type == UNITS.TERRAN_SUPPLY_DEPOT).nonzero()
        supply_depot_count = supply_depot_count = 1 if depot_y.any() else 0

        barracks_y, barracks_x = (unit_type == UNITS.TERRAN_BARRACKS).nonzero()
        barracks_count = 1 if barracks_y.any() else 0

        supply_limit = obs.observation['player'][4]
        army_supply = obs.observation['player'][5]

        killed_unit_score = obs.observation['score_cumulative'][5]
        killed_building_score = obs.observation['score_cumulative'][6]
        worker_count = obs.observation['player'][6]
        score = obs.observation['score_cumulative'][0]
        self.scores.append(score)
        sys.stdout.flush()
        current_state = [
            supply_depot_count,
            barracks_count,
            supply_limit,
            army_supply,
            worker_count
        ]

        if self.previous_action is not None:
            reward = 0

            if killed_unit_score > self.previous_killed_unit_score:
                reward += KILL_UNIT_REWARD

            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))

        rl_action = self.qlearn.choose_action(str(current_state))
        smart_action = ACTIONS.smart_actions[rl_action]

        self.previous_killed_unit_score = killed_unit_score
        self.previous_killed_building_score = killed_building_score
        self.previous_state = current_state
        self.previous_action = rl_action

        return get_action(obs, self.base_top_left, smart_action)
