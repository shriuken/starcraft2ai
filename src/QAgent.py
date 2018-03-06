import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.distributions import Categorical
from collections import namedtuple
from pysc2.agents import base_agent
from actions.actions import *
from pysc2.lib import actions

DATA_FILE = 'sparse_agent_data'
USE_CUDA = False
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(4, 84, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(84)
        self.conv2 = nn.Conv2d(84, 256, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 256, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.mp = nn.MaxPool2d(5)
        self.lin1 = nn.Linear(1024, 4096)
        self.lin2 = nn.Linear(4096, 1024)
        self.action_head = nn.Linear(1024, len(ACTIONS.smart_actions))
        self.value_head = nn.Linear(1024, 1)
        self.saved_log_probs = []
        self.rewards = []
        self.saved_actions = []

    def forward(self, x):
        x = self.mp(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp(F.relu(self.bn4(self.conv4(x))))
        x = F.relu(self.lin1(x.view(x.size(0), -1)))
        x = F.relu(self.lin2(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values


def select_action(model, state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    if USE_CUDA and torch.cuda.is_available():
        probs, state_value = model(Variable(state).cuda())
    else:
        probs, state_value = model(Variable(state))
    m = Categorical(probs)
    action = m.sample()
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.data[0]


def finish_episode(model, optimizer):
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
    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.data[0]
        policy_losses.append(-log_prob * Variable(reward))
        value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([r]))))
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    print(loss)
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]


class RLAgent(base_agent.BaseAgent):
    def __init__(self):
        super(RLAgent, self).__init__()

        self.policy = Policy()
        if USE_CUDA and torch.cuda.is_available():
            self.policy = self.policy.cuda()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-5)

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

        self.move_number = 0

    def reset(self):
        # self.qlearn.save_q_table()
        self.episodes += 1
        print("Running episode: ", self.episodes)
        print("Won: ", self.won, " Lost: ", self.lost)

    def step(self, obs):
        super(RLAgent, self).step(obs)
        self.step_num += 1
        if obs.last():
            reward = -500 if obs.reward == 0 else obs.reward * 500

            self.policy.rewards.append(reward)
            # plt.plot(self.policy.rewards)
            # plt.show()
            self.previous_action = None
            self.previous_state = None

            self.move_number = 0
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

            rl_input = np.array([np.pad(obs.observation['minimap'][0], [(0, 20), (0, 20)], mode='constant'),
                                 np.pad(obs.observation['minimap'][1], [(0, 20), (0, 20)], mode='constant'),
                                 np.pad(obs.observation['minimap'][5], [(0, 20), (0, 20)], mode='constant'),
                                 obs.observation['screen'][6]])
            rl_action = select_action(self.policy, rl_input)

            self.previous_state = current_state
            self.previous_action = rl_action

            smart_action, x, y = split_action(self.previous_action)
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

            smart_action, x, y = split_action(self.previous_action)

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

            smart_action, x, y = split_action(self.previous_action)

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
