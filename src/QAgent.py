import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from random import randrange

import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.distributions import Categorical
from collections import namedtuple
from pysc2.agents import base_agent
from actions.actions import *
from pysc2.lib import actions

DATA_FILE = 'sparse_agent_data'
USE_CUDA = True
IS_RANDOM = False
SavedAction = namedtuple('SavedAction', ['log_prob', 'value', 'x', 'y'])
layer = 2
num_layers = 1


def convolution_layer(x, input_layers):
    conv1 = nn.Conv2d(input_layers, 256, kernel_size=3, stride=1).cuda()
    bn1 = nn.BatchNorm2d(256).cuda()
    return F.relu(bn1(conv1(x)))


def residual_layer(x, input_layers, layer_size):
    conv1 = nn.Conv2d(input_layers, 256, kernel_size=3, stride=1).cuda()
    bn1 = nn.BatchNorm2d(256).cuda()
    conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1).cuda()
    bn2 = nn.BatchNorm2d(256).cuda()
    return F.relu(bn2(conv2(F.relu(bn1(conv1(x))))))


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        residual_out_size = 20736
        self.mp = nn.MaxPool2d(8)
        self.lin1 = nn.Linear(residual_out_size, 2048).cuda()
        self.action_head = nn.Linear(2048, len(ACTIONS.smart_actions)).cuda()
        self.value_head = nn.Linear(2048, 1).cuda()
        self.x = nn.Linear(2048, 64).cuda()
        self.y = nn.Linear(2048, 64).cuda()
        self.saved_log_probs = []
        self.rewards = []
        self.saved_actions = []
        self.updated = 0

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        if USE_CUDA and torch.cuda.is_available():
            probs, state_value, x, y = self.forward(Variable(state, ).cuda())
        else:
            probs, state_value, x, y = self.forward(Variable(state))
        m = Categorical(probs)
        x_probs = Categorical(x)
        y_probs = Categorical(y)
        x_coord = x_probs.sample()
        y_coord = y_probs.sample()
        action = m.sample()
        self.saved_actions.append([m.log_prob(action).data, state_value.data, x_probs.log_prob(x_coord).data, y_probs.log_prob(y_coord).data])
        return action.data[0], x_coord, y_coord

    def forward(self, x):
        x = convolution_layer(x, layer * num_layers)
        x = self.mp(residual_layer(x, 256, 78))
        x = F.relu(self.lin1(x.view(x.size(0), -1)))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        x_coord = self.x(x)
        y_coord = self.y(x)
        return F.softmax(action_scores, dim=-1), state_values, F.softmax(x_coord, dim=-1), F.softmax(y_coord, dim=-1)


def finish_episode(model, optimizer):
    model.updated += 1
    print("Updated Model: ", model.updated, " times")
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in model.rewards[::-1]:
        R = r + 0.99 * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    for (log_prob, value, x_prob, y_prob), r in zip(saved_actions, rewards):
        reward = r - value.data[0]
        policy_losses.append(-log_prob * Variable(reward) + -x_prob * Variable(reward) + -y_prob * Variable(reward))
        value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([r])).cuda()))
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    print(loss)
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]


policy = Policy()


class RLAgent(base_agent.BaseAgent):
    def __init__(self):
        super(RLAgent, self).__init__()
        self.policy = policy
        if USE_CUDA and torch.cuda.is_available():
            self.policy = self.policy.cuda()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.01)

        self.previous_action = None
        self.previous_state = None
        self.previous_cumulative_score_total = 0
        self.previous_killed_building_score = 0
        self.previous_killed_units_score = 0
        self.won = 0
        self.lost = 0
        self.tied = 0
        self.step_num = 0
        self.actions = []
        self.cc_y = None
        self.cc_x = None
        self.rl_input = None

        self.move_number = 0

    def reset(self):
        # self.qlearn.save_q_table()
        self.episodes += 1
        print("Running episode: ", self.episodes)
        print("Won: ", self.won, " Lost: ", self.lost, "Tied: ", self.tied)

    def step(self, obs):
        super(RLAgent, self).step(obs)
        self.step_num += 1
        if obs.last():
            reward = obs.reward * 15

            self.policy.rewards.append(reward)
            # plt.plot(self.policy.rewards)
            # plt.show()
            self.previous_action = None
            self.previous_state = None

            self.move_number = 0
            if IS_RANDOM is False:
                # Train our network.
                print("Episode Reward: ", sum(self.policy.rewards))
                plt.figure(figsize=(18.0, 12.0))
                plt.plot(self.actions, '.')
                script_dir = os.path.dirname(__file__)
                results_dir = os.path.join(script_dir, 'A2CResults/')
                sample_file_name = str(self.episodes)

                if not os.path.isdir(results_dir):
                    os.makedirs(results_dir)

                plt.savefig(results_dir + sample_file_name)
                finish_episode(self.policy, self.optimizer)
            if obs.reward > 0:
                self.won += 1
            elif obs.reward == 0:
                self.tied += 1
            else:
                self.lost += 1

            return actions.FunctionCall(ACTIONS.NO_OP, [])

        unit_type = obs.observation['screen'][MISC.UNIT_TYPE]

        if obs.first():
            self.previous_cumulative_score_total = obs.observation['score_cumulative'][0]
            player_y, player_x = (obs.observation['minimap'][MISC.PLAYER_RELATIVE] == MISC.PLAYER_SELF).nonzero()
            self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

            self.cc_y, self.cc_x = (unit_type == UNITS.TERRAN_COMMANDCENTER).nonzero()

        cc_y, cc_x = (unit_type == UNITS.TERRAN_COMMANDCENTER).nonzero()
        cc_count = 1 if cc_y.any() else 0

        depot_y, depot_x = (unit_type == UNITS.TERRAN_SUPPLY_DEPOT).nonzero()
        supply_depot_count = int(round(len(depot_y) / 69))

        barracks_y, barracks_x = (unit_type == UNITS.TERRAN_BARRACKS).nonzero()
        barracks_count = int(round(len(barracks_y) / 137))

        if self.move_number == 0:
            self.move_number += 1

            current_state = np.zeros(8)
            current_state[0] = cc_count
            current_state[1] = supply_depot_count
            current_state[2] = barracks_count
            current_state[3] = obs.observation['player'][MISC.ARMY_SUPPLY]

            # hot_squares = np.zeros(4)
            # enemy_y, enemy_x = (obs.observation['minimap'][MISC.PLAYER_RELATIVE] == MISC.PLAYER_HOSTILE).nonzero()

            if self.previous_action is not None:
                # reward = obs.observation['score_cumulative'][0] - self.previous_cumulative_score_total
                reward = 0
                killed_unit_score = obs.observation['score_cumulative'][5]
                killed_building_score = obs.observation['score_cumulative'][6]

                if killed_building_score > self.previous_killed_building_score:
                    reward += 1
                if killed_unit_score > self.previous_killed_units_score:
                    reward += 0.05

                self.policy.rewards.append(reward)
                # finish_episode(self.policy, self.optimizer)
                self.previous_cumulative_score_total = obs.observation['score_cumulative'][0]
                self.previous_killed_building_score = killed_building_score
                self.previous_killed_units_score = killed_unit_score

            if IS_RANDOM is False:
                if self.rl_input is None:
                    self.rl_input = np.array([np.pad(obs.observation['minimap'][5], [(0, 20), (0, 20)], mode='constant'),
                                     obs.observation['screen'][6]])
                    for _ in range(num_layers - 1):
                        self.rl_input = np.concatenate((self.rl_input, np.array(
                            [np.pad(obs.observation['minimap'][5], [(0, 20), (0, 20)], mode='constant'),
                             obs.observation['screen'][6]])))
                else:
                    self.rl_input = np.concatenate((self.rl_input, np.array([np.pad(obs.observation['minimap'][5], [(0, 20), (0, 20)], mode='constant'),
                                     obs.observation['screen'][6]])))

            if IS_RANDOM or self.rl_input.shape == (layer * num_layers, 84, 84):
                if IS_RANDOM:
                    rl_action = randrange(0, len(ACTIONS.smart_actions))
                else:
                    rl_action, x, y = self.policy.select_action(self.rl_input)
                    self.rl_input = np.delete(self.rl_input, np.s_[0:2], axis=0)

                self.previous_state = current_state
                self.previous_action = rl_action
                self.prev_x = x
                self.prev_y = y

                smart_action = ACTIONS.smart_actions[rl_action]
                self.actions.append(smart_action)

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

                        return actions.FunctionCall(ACTIONS.SELECT_POINT, [MISC.SELECT_ALL, target])

                elif smart_action == ACTIONS.ACTION_ATTACK:
                    if ACTIONS.SELECT_ARMY in obs.observation['available_actions']:
                        return actions.FunctionCall(ACTIONS.SELECT_ARMY, [MISC.NOT_QUEUED])

        elif self.move_number == 1:
            self.move_number += 1

            smart_action = ACTIONS.smart_actions[self.previous_action]
            x = self.prev_x
            y = self.prev_y

            if smart_action == ACTIONS.ACTION_BUILD_SUPPLY_DEPOT:
                if ACTIONS.BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
                    if cc_y.any():
                        target = [int(x), int(y)]

                        return actions.FunctionCall(ACTIONS.BUILD_SUPPLY_DEPOT, [MISC.NOT_QUEUED, target])

            elif smart_action == ACTIONS.ACTION_BUILD_BARRACKS:
                if ACTIONS.BUILD_BARRACKS in obs.observation['available_actions']:
                    if cc_y.any():
                        target = [int(x), int(y)]

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
                    target = [int(x), int(y)]

                    return actions.FunctionCall(ACTIONS.ATTACK_MINIMAP, [MISC.NOT_QUEUED, target])

        elif self.move_number == 2:
            self.move_number = 0

            smart_action = ACTIONS.smart_actions[self.previous_action]
            x = self.prev_x
            y = self.prev_y

            if smart_action == ACTIONS.ACTION_BUILD_BARRACKS or smart_action == ACTIONS.ACTION_BUILD_SUPPLY_DEPOT:
                if ACTIONS.HARVEST_GATHER in obs.observation['available_actions']:
                    unit_y, unit_x = (unit_type == UNITS.NEUTRAL_MINERAL_FIELD).nonzero()

                    if unit_y.any():
                        i = random.randint(0, len(unit_y) - 1)

                        m_x = unit_x[i]
                        m_y = unit_y[i]

                        target = [int(m_x), int(m_y)]

                        return actions.FunctionCall(ACTIONS.HARVEST_GATHER, [MISC.QUEUED, target])

        return actions.FunctionCall(ACTIONS.NO_OP, [])
