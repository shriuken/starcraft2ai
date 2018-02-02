import random
import math
import sys

import numpy as np
import pandas as pd
import pickle as pkl

from pysc2.agents import base_agent
from actions.actions import *
from pysc2.lib import actions

KILL_UNIT_REWARD = 1
DATA_FILE = 'sparse_agent_data'

# Stolen from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]

            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))

            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.check_state_exist(s)

        q_predict = self.q_table.ix[s, a]

        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()
        else:
            q_target = r  # next state is terminal

        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


class RLAgent(base_agent.BaseAgent):
    def __init__(self):
        super(RLAgent, self).__init__()

        self.qlearn = QLearningTable(actions=list(range(len(ACTIONS.smart_actions))))

        self.previous_action = None
        self.previous_state = None

        self.cc_y = None
        self.cc_x = None

        self.move_number = 0

    def reset(self):
        # self.qlearn.save_q_table()
        self.episodes += 1
        print("Running episode: ", self.episodes)

    def step(self, obs):
        super(RLAgent, self).step(obs)

        if obs.last():
            reward = obs.reward

            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, 'terminal')

            self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')

            self.previous_action = None
            self.previous_state = None

            self.move_number = 0

            return actions.FunctionCall(MISC.NO_OP, [])

        unit_type = obs.observation['screen'][MISC.UNIT_TYPE]

        if obs.first():
            player_y, player_x = (obs.observation['minimap'][MISC.PLAYER_RELATIVE] == MISC.PLAYER_SELF).nonzero()
            self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

            self.cc_y, self.cc_x = (unit_type == UNITS.TERRAN_COMMANDCENTER).nonzero()

        cc_y, cc_x = (unit_type == UNITS.TERRAN_COMMANDCENTER).nonzero()
        cc_count = 1 if cc_y.any() else 0

        depot_y, depot_x = (unit_type == UNITS.TERRAN_SUPPLY_DEPOT).nonzero()
        supply_depot_count = int(round(len(depot_y) / 69))

        barracks_y, barracks_x = (unit_type == UNITS.TERRAN_BARRACKS).nonzero()
        barracks_count = int(round(len(barracks_y) / 137))
        # action = get_action(obs, self.base_top_left, self.move_number, cc_count,
        #                   supply_depot_count, barracks_count, self.previous_state,
        #                   self.previous_action, self.qlearn, unit_type,
        #                   barracks_x, barracks_y, cc_x, cc_y)

        if self.move_number == 0:
            self.move_number += 1

            current_state = np.zeros(8)
            current_state[0] = cc_count
            current_state[1] = supply_depot_count
            current_state[2] = barracks_count
            current_state[3] = obs.observation['player'][MISC.ARMY_SUPPLY]

            hot_squares = np.zeros(4)
            enemy_y, enemy_x = (obs.observation['minimap'][MISC.PLAYER_RELATIVE] == MISC.PLAYER_HOSTILE).nonzero()
            for i in range(0, len(enemy_y)):
                y = int(math.ceil((enemy_y[i] + 1) / 32))
                x = int(math.ceil((enemy_x[i] + 1) / 32))

                hot_squares[((y - 1) * 2) + (x - 1)] = 1

            if not self.base_top_left:
                hot_squares = hot_squares[::-1]

            for i in range(0, 4):
                current_state[i + 4] = hot_squares[i]

            if self.previous_action is not None:
                self.qlearn.learn(str(self.previous_state), self.previous_action, 0, str(current_state))

            rl_action = self.qlearn.choose_action(str(current_state))

            previous_state = current_state
            previous_action = rl_action

            smart_action, x, y = split_action(previous_action)

            if smart_action == ACTIONS.ACTION_BUILD_BARRACKS or smart_action == ACTIONS.ACTION_BUILD_SUPPLY_DEPOT:
                unit_y, unit_x = (unit_type == UNITS.TERRAN_SCV).nonzero()

                if unit_y.any():
                    i = random.randint(0, len(unit_y) - 1)
                    target = [unit_x[i], unit_y[i]]

                    return actions.FunctionCall(ACTIONS.SELECT_POINT, [MISC.NOT_QUEUED, target])

            elif smart_action == ACTIONS.ACTION_BUILD_MARINE:
                if barracks_y.any():
                    i = random.randint(0, len(barracks_y) - 1)
                    target = [barracks_x[i], barracks_y[i]]

                    return actions.FunctionCall(ACTIONS.SELECT_POINT, [ACTIONS.SELECT_ALL, target])

            elif smart_action == ACTIONS.ACTION_ATTACK:
                if ACTIONS.SELECT_ARMY in obs.observation['available_actions']:
                    return actions.FunctionCall(ACTIONS.SELECT_ARMY, [MISC.NOT_QUEUED])

        elif self.move_number == 1:
            self.move_number += 1

            smart_action, x, y = split_action(self.previous_action)

            if smart_action == ACTIONS.ACTION_BUILD_SUPPLY_DEPOT:
                if supply_depot_count < 2 and ACTIONS.BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
                    if cc_y.any():
                        if supply_depot_count == 0:
                            target = transform_distance(round(cc_x.mean()), -35, round(cc_y.mean()), 0)
                        elif supply_depot_count == 1:
                            target = transform_distance(round(cc_x.mean()), -25, round(cc_y.mean()), -25)

                        return actions.FunctionCall(ACTIONS.BUILD_SUPPLY_DEPOT, [MISC.NOT_QUEUED, target])

            elif smart_action == ACTIONS.ACTION_BUILD_BARRACKS:
                if barracks_count < 2 and ACTIONS.BUILD_BARRACKS in obs.observation['available_actions']:
                    if cc_y.any():
                        if barracks_count == 0:
                            target = transform_distance(round(cc_x.mean()), 15, round(cc_y.mean()), -9)
                        elif barracks_count == 1:
                            target = transform_distance(round(cc_x.mean()), 15, round(cc_y.mean()), 12)

                        return actions.FunctionCall(ACTIONS.BUILD_BARRACKS, [MISC.NOT_QUEUED, target])

            elif smart_action == ACTIONS.ACTION_BUILD_MARINE:
                if ACTIONS.TRAIN_MARINE in obs.observation['available_actions']:
                    return actions.FunctionCall(ACTIONS.TRAIN_MARINE, [MISC.QUEUED])

            elif smart_action == ACTIONS.ACTION_ATTACK:
                do_it = True

                if len(obs.observation['single_select']) > 0 and obs.observation['single_select'][0][
                    0] == UNITS.TERRAN_SCV:
                    do_it = False

                if len(obs.observation['multi_select']) > 0 and obs.observation['multi_select'][0][
                    0] == UNITS.TERRAN_SCV:
                    do_it = False

                if do_it and ACTIONS.ATTACK_MINIMAP in obs.observation["available_actions"]:
                    x_offset = random.randint(-1, 1)
                    y_offset = random.randint(-1, 1)

                    return actions.FunctionCall(ACTIONS.ATTACK_MINIMAP, [MISC.NOT_QUEUED,
                                                                         transform_location(int(x) + (x_offset * 8),
                                                                                            int(y) + (y_offset * 8))])

        elif self.move_number == 2:
            move_number = 0

            smart_action, x, y = split_action(self.previous_action)

            if smart_action == ACTIONS.ACTION_BUILD_BARRACKS or smart_action == ACTIONS.ACTION_BUILD_SUPPLY_DEPOT:
                if UNITS.HARVEST_GATHER in obs.observation['available_actions']:
                    unit_y, unit_x = (unit_type == UNITS.NEUTRAL_MINERAL_FIELD).nonzero()

                    if unit_y.any():
                        i = random.randint(0, len(unit_y) - 1)

                        m_x = unit_x[i]
                        m_y = unit_y[i]

                        target = [int(m_x), int(m_y)]

                        return actions.FunctionCall(UNITS.HARVEST_GATHER, [MISC.QUEUED, target])

        return actions.FunctionCall(ACTIONS.NO_OP, [])
