import random
import math

import numpy as np
import pandas as pd
import pickle as pkl

import sys
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
import matplotlib.pyplot as plt

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21

_SCREEN = [0]

# Always square for now
_MAP_SIZE = [63, 63]

ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_SCV = 'selectscv'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_SELECT_BARRACKS = 'selectbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_SELECT_ARMY = 'selectarmy'
ACTION_ATTACK = 'attack'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_SELECT_SCV,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_SELECT_BARRACKS,
    ACTION_BUILD_MARINE,
    ACTION_SELECT_ARMY,
    ACTION_ATTACK,
]

KILL_UNIT_REWARD = 0.5
KILL_BUILDING_REWARD = 0.7
MORE_UNIT_REWARD = 0.5
LOST_SCV_PENATLY = 1
MINERAL_RATE_PENALTY = 0.1


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

            action = state_action.argmax()
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


def in_range(target):
    return _MAP_SIZE[0] > target[0] > 0 and _MAP_SIZE[1] > target[1] > 0


class RLAgent(base_agent.BaseAgent):
    def __init__(self):
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.obs_spec = None
        self.action_spec = None
        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))
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


    def transform_location(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]

        return [x + x_distance, y + y_distance]

    def reset(self):
        self.qlearn.save_q_table()
        self.scores = []

    def step(self, obs):
        super(RLAgent, self).step(obs)

        if self.previous_worker_count is None and self.previous_mineral_rate is None:
            self.previous_worker_count = obs.observation['player'][6]
            self.previous_mineral_rate = obs.observation['score_cumulative'][9]

        player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
        self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

        unit_type = obs.observation['screen'][_UNIT_TYPE]

        depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
        supply_depot_count = supply_depot_count = 1 if depot_y.any() else 0

        barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
        barracks_count = 1 if barracks_y.any() else 0

        supply_limit = obs.observation['player'][4]
        army_supply = obs.observation['player'][5]

        killed_unit_score = obs.observation['score_cumulative'][5]
        killed_building_score = obs.observation['score_cumulative'][6]
        mineral_rate = obs.observation['score_cumulative'][9]
        worker_count = obs.observation['player'][6]
        score = obs.observation['score_cumulative'][0]
        self.scores.append(score)
        print('Score: [%d%%]\r' % score, end="")
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

            if barracks_count > self.previous_barracks_count:
                reward += 5

            if army_supply > self.previous_army_count:
                reward += 0.3

            if killed_building_score > self.previous_killed_building_score:
                reward += KILL_BUILDING_REWARD

            if mineral_rate < self.previous_mineral_rate:
                reward -= MINERAL_RATE_PENALTY

            if worker_count < self.previous_worker_count:
                reward -= LOST_SCV_PENATLY

            reward += (score - self.previous_score) / 1000

            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))

        rl_action = self.qlearn.choose_action(str(current_state))
        smart_action = smart_actions[rl_action]

        self.previous_killed_unit_score = killed_unit_score
        self.previous_killed_building_score = killed_building_score
        self.previous_barracks_count = barracks_count
        self.previous_mineral_rate = mineral_rate
        self.previous_worker_count = worker_count
        self.previous_army_count = army_supply
        self.previous_score = score
        self.previous_state = current_state
        self.previous_action = rl_action

        if smart_action == ACTION_DO_NOTHING:
            return actions.FunctionCall(_NO_OP, [])

        elif smart_action == ACTION_SELECT_SCV:
            unit_type = obs.observation['screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()

            if unit_y.any():
                i = random.randint(0, len(unit_y) - 1)
                target = [unit_x[i], unit_y[i]]
                if in_range(target):
                    return actions.FunctionCall(_SELECT_POINT, [_SCREEN, target])

        elif smart_action == ACTION_BUILD_SUPPLY_DEPOT:
            if _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
                unit_type = obs.observation['screen'][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

                if unit_y.any():
                    target = self.transform_location(int(unit_x.mean()), 0, int(unit_y.mean()), 20)
                    if in_range(target):
                        return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_SCREEN, target])

        elif smart_action == ACTION_BUILD_BARRACKS:
            if _BUILD_BARRACKS in obs.observation['available_actions']:
                unit_type = obs.observation['screen'][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

                if unit_y.any():
                    target = self.transform_location(int(unit_x.mean()), 20, int(unit_y.mean()), 0)

                    if in_range(target):
                        return actions.FunctionCall(_BUILD_BARRACKS, [_SCREEN, target])

        elif smart_action == ACTION_SELECT_BARRACKS:
            unit_type = obs.observation['screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()

            if unit_y.any():
                target = [int(unit_x.mean()), int(unit_y.mean())]

                if in_range(target):
                    return actions.FunctionCall(_SELECT_POINT, [_SCREEN, target])

        elif smart_action == ACTION_BUILD_MARINE:
            if _TRAIN_MARINE in obs.observation['available_actions']:
                return actions.FunctionCall(_TRAIN_MARINE, [[1]])

        elif smart_action == ACTION_SELECT_ARMY:
            if _SELECT_ARMY in obs.observation['available_actions']:
                return actions.FunctionCall(_SELECT_ARMY, [[0]])

        elif smart_action == ACTION_ATTACK:
            if _ATTACK_MINIMAP in obs.observation["available_actions"]:
                if self.base_top_left:
                    return actions.FunctionCall(_ATTACK_MINIMAP, [[1], [39, 45]])

                return actions.FunctionCall(_ATTACK_MINIMAP, [[1], [21, 24]])

        return actions.FunctionCall(_NO_OP, [])