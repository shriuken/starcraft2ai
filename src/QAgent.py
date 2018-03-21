import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import shutil

import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.distributions import Categorical
from collections import namedtuple
from pysc2.agents import base_agent
from actions.actions import *
from pysc2.lib import actions
from random import randrange
from math import log

DATA_FILE = 'sparse_agent_data'
USE_CUDA = True
IS_RANDOM = False
resume = True
resume_best = False
evaluate = False

SavedAction = namedtuple('SavedAction', ['log_prob', 'value', 'x', 'y'])
layer = 5
num_layers = 2


def convolution_layer(x, input_layers):
    conv1 = nn.Conv2d(input_layers, 256, kernel_size=3, stride=1).cuda()
    bn1 = nn.BatchNorm2d(256).cuda()
    return F.relu(bn1(conv1(x)))


def residual_layer(x, input_layers):
    conv1 = nn.Conv2d(input_layers, 256, kernel_size=3, stride=1).cuda()
    bn1 = nn.BatchNorm2d(256).cuda()
    conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1).cuda()
    bn2 = nn.BatchNorm2d(256).cuda()
    return F.relu(bn2(conv2(F.relu(bn1(conv1(x))))))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.net_size = 32
        self.hidden = self.init_hidden()
        self.hidden2 = self.init_hidden()
        self.embedding_dim = self.net_size
        self.in_conv_mini = nn.Conv2d(5, 64, kernel_size=3, stride=2).cuda()
        self.in_conv_screen = nn.Conv2d(4, 64, kernel_size=3, stride=2).cuda()
        self.in_conv_other = nn.Conv2d(3, 64, kernel_size=3, stride=2).cuda()
        self.in_bn = nn.BatchNorm2d(64).cuda()
        self.in_large_conv = nn.Conv2d(64, self.net_size, kernel_size=5, stride=1).cuda()
        self.large_conv = nn.Conv2d(self.net_size, self.net_size, kernel_size=5, stride=1).cuda()
        self.bn = nn.BatchNorm2d(self.net_size).cuda()
        self.bn_conv = nn.BatchNorm2d(19).cuda()
        self.cat_conv = nn.Conv2d(96, 19, kernel_size=3, stride=1).cuda()
        self.small_conv_cat = nn.Conv2d(19, 19, kernel_size=3, stride=1).cuda()
        self.small_conv = nn.Conv2d(self.net_size, self.net_size, kernel_size=3, stride=1).cuda()
        self.mp = nn.MaxPool2d(3)
        self.lstm = nn.LSTM(475, 512).cuda()
        self.lin1 = nn.Linear(512, 64).cuda()
        self.action_head = nn.Linear(64, len(ACTIONS.smart_actions)).cuda()
        self.value_head = nn.Linear(64, 1).cuda()
        self.x = nn.Linear(64, 64).cuda()
        self.y = nn.Linear(64, 64).cuda()
        self.rewards = []
        self.saved_actions = []
        self.updated = 0

    def init_hidden(self):
        return (torch.autograd.Variable(torch.randn(1, 1, 512)).cuda(),
                torch.autograd.Variable(torch.randn((1, 1, 512))).cuda())

    def select_action(self, minimap, screen, other_features):
        minimap = torch.from_numpy(minimap).float().unsqueeze(0)
        screen = torch.from_numpy(screen).float().unsqueeze(0)
        other_features = torch.from_numpy(other_features).float().unsqueeze(0)
        if USE_CUDA and torch.cuda.is_available():
            probs, state_value, x, y = self.forward(Variable(minimap).cuda(), Variable(screen).cuda(), Variable(other_features).cuda())
        else:
            probs, state_value, x, y = self.forward(Variable(minimap), Variable(screen), Variable(other_features))
        m = Categorical(probs)
        x_probs = Categorical(x)
        y_probs = Categorical(y)
        x_coord = x_probs.sample()
        y_coord = y_probs.sample()
        action = m.sample()
        if not evaluate:
            self.saved_actions.append(
                [m.log_prob(action), state_value, x_probs.log_prob(x_coord), y_probs.log_prob(y_coord)])
        return action.data[0], x_coord, y_coord

    def forward(self, minimap, screen, other_features):
        minimap = F.relu(self.in_bn(self.in_conv_mini(minimap)))
        minimap = F.relu(self.bn(self.in_large_conv(minimap)))
        minimap = F.relu((self.bn(self.large_conv(minimap))))
        minimap = F.relu((self.bn(self.small_conv(minimap))))
        minimap = F.relu((self.bn(self.small_conv(minimap))))

        screen = F.relu(self.in_bn(self.in_conv_screen(screen)))
        screen = F.relu(self.bn(self.in_large_conv(screen)))
        screen = F.relu((self.bn(self.large_conv(screen))))
        screen = F.relu((self.bn(self.small_conv(screen))))
        screen = F.relu((self.bn(self.small_conv(screen))))

        other_features = F.relu(self.in_bn(self.in_conv_other(other_features)))
        other_features = F.relu(self.bn(self.in_large_conv(other_features)))
        other_features = F.relu((self.bn(self.large_conv(other_features))))
        other_features = F.relu((self.bn(self.small_conv(other_features))))
        other_features = F.relu((self.bn(self.small_conv(other_features))))

        x = torch.cat((minimap, screen, other_features), dim=1)

        x = F.relu(self.bn_conv(self.cat_conv(x)))
        x = F.relu(self.bn_conv(self.small_conv_cat(x)))
        x = F.relu(self.bn_conv(self.small_conv_cat(x)))
        x = F.relu(self.bn_conv(self.small_conv_cat(x)))
        x = F.relu(self.bn_conv(self.small_conv_cat(x)))
        x = F.relu(self.bn_conv(self.small_conv_cat(x)))
        x = F.relu(self.bn_conv(self.small_conv_cat(x)))
        x, self.hidden = self.lstm(x.view(x.size(0), -1).unsqueeze(0), self.hidden)
        x = x.squeeze(0)
        x = F.relu(self.lin1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        x_coord = self.x(x)
        y_coord = self.y(x)
        return F.softmax(action_scores, dim=-1), state_values, F.softmax(x_coord, dim=-1), F.softmax(y_coord, dim=-1)


def finish_episode(model, optimizer, is_best, episode, won, lost, tied):
    model.updated += 1
    print("Updated Model: ", model.updated, " times")
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    movement_losses = []
    rewards = []
    for r in model.rewards[::-1]:
        R = r + 0.99 * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    for (log_prob, value, x_prob, y_prob), r in zip(saved_actions, rewards):
        reward = r - value.data[0]
        policy_losses.append(-log_prob * Variable(reward))
        movement_losses.append(-x_prob * Variable(reward) + -y_prob * Variable(reward))
        value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([r])).cuda()))
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum() + torch.stack(movement_losses).sum()
    print(loss)
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]
    model.hidden = model.init_hidden()
    model.hidden2 = model.init_hidden()
    save_checkpoint({
        'epoch': episode + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'won': won,
        'lost': lost,
        'tied': tied
    }, is_best)


policy = Policy()


class RLAgent(base_agent.BaseAgent):
    def __init__(self):
        super(RLAgent, self).__init__()
        self.policy = policy
        if USE_CUDA and torch.cuda.is_available():
            self.policy = self.policy.cuda()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.0005)

        self.previous_action = None
        self.previous_state = None
        self.previous_cumulative_score_total = 0
        self.previous_killed_building_score = 0
        self.previous_killed_units_score = 0
        self.previous_build_unit_score = 0
        self.won = 0
        self.lost = 0
        self.tied = 0
        self.won_arr = []
        self.lost_arr = []
        self.tied_arr = []
        self.step_num = 0
        self.time_scalar = 0
        self.actions = []
        self.episode_rewards = []
        self.cc_y = None
        self.cc_x = None
        self.rl_input = None
        # This is to amp up rewards earlier and taper them off later
        # self.time_scalar = (0.0001 * self.step_num)

        self.move_number = 0

        if resume:
            file_to_load = "model_best.pth.tar" if resume_best else "checkpoint.pth.tar"
            if os.path.isfile(file_to_load):
                print("=> loading checkpoint '{}'".format(file_to_load))
                checkpoint = torch.load(file_to_load)
                self.episodes = checkpoint['epoch']
                self.policy.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                # self.won = checkpoint['won'] if 'won' in checkpoint.keys() else 0
                # self.lost = checkpoint['lost'] if 'lost' in checkpoint.keys() else 0
                # self.tied = checkpoint['tied'] if 'tied' in checkpoint.keys() else 0
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(file_to_load, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(file_to_load))

    def reset(self):
        # self.qlearn.save_q_table()
        self.episodes += 1
        self.step_num = 0
        print("Running episode: ", self.episodes)
        print("Won: ", self.won, " Lost: ", self.lost, "Tied: ", self.tied)

    def step(self, obs):
        super(RLAgent, self).step(obs)
        self.step_num += 1
        if obs.last():
            reward = -30 if obs.reward is 0 else obs.reward * 50
            # reward = (obs.reward / (3 * (log((self.step_num / 2501) * 3, 5)))) / 3 if obs.reward <= 0 else obs.reward / (3 * (log((self.step_num / 2501) * 3, 5)))
            # self.policy.rewards = [x + reward for x in self.policy.rewards]
            self.policy.rewards.append(reward)

            # plt.plot(self.policy.rewards)
            # plt.show()
            self.previous_action = None
            self.previous_state = None

            self.move_number = 0
            if IS_RANDOM is False:
                # Train our network.
                print("Episode Reward: ", sum(self.policy.rewards), self.policy.rewards[0])
                print("Time Scalar: ", self.time_scalar)
                print("Steps: ", self.step_num)
                if obs.reward > 0:
                    self.won += 1
                elif obs.reward == 0:
                    self.tied += 1
                else:
                    self.lost += 1

                self.won_arr.append(self.won / (self.won + self.lost + self.tied))
                self.tied_arr.append(self.tied / (self.won + self.lost + self.tied))
                self.lost_arr.append(self.lost / (self.won + self.lost + self.tied))
                # Keep track of the episode rewards.
                self.episode_rewards.append(sum(self.policy.rewards) / len(self.policy.rewards))

                plt.close()
                plt.title("Win, Loss, and Tie Percentage")
                plt.ylabel("Percent")
                plt.xlabel("Episode")
                plt.plot(self.won_arr)
                plt.plot(self.lost_arr)
                plt.plot(self.tied_arr)
                plt.legend(["Won", "Lost", "Tied"], loc='upper left')
                sample_file_name = "win_loss_tie_graph"
                plt.savefig(sample_file_name, overwrite=True)

                plt.close()

                plt.title("Average Reward over time")
                plt.ylabel("Average Reward")
                plt.xlabel("Episode")
                plt.plot(self.episode_rewards)
                sample_file_name = "reward_graph"
                plt.savefig(sample_file_name, overwrite=True)

                if len(self.episode_rewards) > 1:
                    is_best = sum(self.episode_rewards) / len(self.episode_rewards) > sum(
                        self.episode_rewards[:(len(self.episode_rewards)) - 1]) / (len(self.episode_rewards) - 1)
                else:
                    is_best = False
                if not evaluate:
                    finish_episode(self.policy, self.optimizer, is_best, self.episodes, self.won, self.lost, self.tied)

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
        supply_depot_count = int(len(depot_y) / 37)

        barracks_y, barracks_x = (unit_type == UNITS.TERRAN_BARRACKS).nonzero()
        barracks_count = int(len(barracks_y) / 80)

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
                reward = -0.05
                killed_unit_score = obs.observation['score_cumulative'][5]
                killed_building_score = obs.observation['score_cumulative'][6]
                built_unit_score = obs.observation['score_cumulative'][8]
                # #
                if killed_building_score > self.previous_killed_building_score:
                    reward += 1
                if killed_unit_score > self.previous_killed_units_score:
                    reward += .1

                self.policy.rewards.append(reward)
                self.previous_cumulative_score_total = obs.observation['score_cumulative'][0]
                self.previous_killed_building_score = killed_building_score
                self.previous_killed_units_score = killed_unit_score
                self.previous_build_unit_score = built_unit_score

            if IS_RANDOM is False:
                screen_state = np.array([
                    obs.observation['screen'][2], # creep
                    obs.observation['screen'][5], # player_relative [0-4][background, self, ally, neutral, enemy] 
                    obs.observation['screen'][7], # selected
                    obs.observation['screen'][8] # hit points
                ])
                minimap_state = np.array([
                    obs.observation['minimap'][0], # height map
                    obs.observation['minimap'][1], # visibility
                    obs.observation['minimap'][2], # creep
                    obs.observation['minimap'][5], # player_relative [0-4][background, self, ally, neutral, enemy] 
                    obs.observation['minimap'][6] # selected
                ])
                other_state = np.array([
                    np.resize(np.array(obs.observation['player'][1]), (64, 64)),
                    np.resize(np.array(obs.observation['player'][2]), (64, 64)),
                    np.resize(np.array(obs.observation['player'][8]), (64, 64))
                ])

            if IS_RANDOM:
                rl_action = randrange(0, len(ACTIONS.smart_actions))
                x = randrange(0, 64)
                y = randrange(0, 64)
            else:
                rl_action, x, y = self.policy.select_action(minimap_state, screen_state, other_state)
                # self.rl_input = np.delete(self.rl_input, np.s_[0:layer], axis=0)

            self.previous_state = current_state
            self.previous_action = rl_action
            self.prev_target = [0, 0]
            self.prev_x = x
            self.prev_y = y

            smart_action = ACTIONS.smart_actions[rl_action]
            self.actions.append(smart_action)

            # line_new = '{:>20}  {:>4}  {:>4}'.format(smart_action, int(x), int(y))
            # print(line_new)

            if smart_action == ACTIONS.ACTION_BUILD_BARRACKS or smart_action == ACTIONS.ACTION_BUILD_SUPPLY_DEPOT or smart_action == ACTIONS.ACTION_BUILD_REFINERY or smart_action == ACTIONS.ACTION_MINE_VESPENE:
                unit_y, unit_x = (unit_type == UNITS.TERRAN_SCV).nonzero()

                if unit_y.any():
                    i = random.randint(0, len(unit_y) - 1)
                    target = [unit_x[i], unit_y[i]]

                    return actions.FunctionCall(ACTIONS.SELECT_POINT, [MISC.NOT_QUEUED, target])

            elif smart_action == ACTIONS.ACTION_BUILD_TECH_LAB:
                unit_y, unit_x = (unit_type == UNITS.TERRAN_BARRACKS).nonzero()
                unit_y_fly, unit_x_fly = (unit_type == UNITS.TERRAN_BARRACKS_FLYING).nonzero()
                if unit_y_fly.any():
                    i = random.randint(0, len(unit_y_fly) - 1)

                    m_x = unit_x_fly[i]
                    m_y = unit_y_fly[i]

                    target = [m_x, m_y]

                    return actions.FunctionCall(ACTIONS.SELECT_POINT, [MISC.NOT_QUEUED, target])
                if unit_y.any():
                    i = random.randint(0, len(unit_y) - 1)

                    m_x = unit_x[i]
                    m_y = unit_y[i]

                    target = [m_x, m_y]

                    return actions.FunctionCall(ACTIONS.SELECT_POINT, [MISC.NOT_QUEUED, target])

            elif smart_action == ACTIONS.ACTION_BUILD_SCV:
                if cc_y.any() and obs.observation['player'][6] < 30:
                    cc_x[int(len(cc_x) / 2)]
                    target = [cc_x[int(len(cc_x) / 2)], cc_y[int(len(cc_y) / 2)]]

                    return actions.FunctionCall(ACTIONS.SELECT_POINT, [MISC.NOT_QUEUED, target])

            elif smart_action == ACTIONS.ACTION_BUILD_MARINE or smart_action == ACTIONS.ACTION_BUILD_MAURADER:
                if barracks_y.any():
                    i = random.randint(0, len(barracks_y) - 1)
                    target = [barracks_x[i], barracks_y[i]]
                    self.prev_target = target

                    return actions.FunctionCall(ACTIONS.SELECT_POINT, [MISC.SELECT_ALL, target])

            elif smart_action == ACTIONS.ACTION_ATTACK:
                if ACTIONS.SELECT_ARMY in obs.observation['available_actions']:
                    return actions.FunctionCall(ACTIONS.SELECT_ARMY, [MISC.NOT_QUEUED])

        elif self.move_number == 1:
            self.move_number += 1

            smart_action = ACTIONS.smart_actions[self.previous_action]
            x = self.prev_x
            y = self.prev_y

            if smart_action == ACTIONS.ACTION_MINE_VESPENE:
                unit_y, unit_x = (unit_type == UNITS.TERRAN_REFINERY).nonzero()

                if unit_y.any() and ACTIONS.SMART_SCREEN in obs.observation['available_actions']:
                    i = random.randint(0, len(unit_y) - 1)

                    m_x = unit_x[i]
                    m_y = unit_y[i]

                    target = [int(m_x), int(m_y)]

                    return actions.FunctionCall(ACTIONS.SMART_SCREEN, [MISC.QUEUED, target])

            if smart_action == ACTIONS.ACTION_BUILD_SUPPLY_DEPOT:
                if supply_depot_count < 8 and ACTIONS.BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
                    if cc_y.any():
                        target = [int(x), int(y)]

                        return actions.FunctionCall(ACTIONS.BUILD_SUPPLY_DEPOT, [MISC.NOT_QUEUED, target])

            elif smart_action == ACTIONS.ACTION_BUILD_BARRACKS:
                if ACTIONS.BUILD_BARRACKS in obs.observation['available_actions']:
                    if cc_y.any():
                        target = [int(x), int(y)]

                        return actions.FunctionCall(ACTIONS.BUILD_BARRACKS, [MISC.NOT_QUEUED, target])

            elif smart_action == ACTIONS.ACTION_BUILD_REFINERY:
                if ACTIONS.BUILD_REFINERY in obs.observation['available_actions']:
                    unit_y, unit_x = (unit_type == UNITS.NEUTRAL_VESPENE_MINE).nonzero()
                    if unit_y.any():
                        i = random.randint(0, len(unit_y) - 1)

                        m_x = unit_x[i]
                        m_y = unit_y[i]

                        target = [m_x, m_y]

                        return actions.FunctionCall(ACTIONS.BUILD_REFINERY, [MISC.NOT_QUEUED, target])

            elif smart_action == ACTIONS.ACTION_BUILD_TECH_LAB:
                if ACTIONS.BUILD_TECH_LAB in obs.observation['available_actions']:
                    target = [int(x), int(y)]
                    return actions.FunctionCall(ACTIONS.BUILD_TECH_LAB, [MISC.NOT_QUEUED, target])

            elif smart_action == ACTIONS.ACTION_BUILD_SCV:
                if ACTIONS.TRAIN_SCV in obs.observation['available_actions']:
                    return actions.FunctionCall(ACTIONS.TRAIN_SCV, [MISC.QUEUED])

            elif smart_action == ACTIONS.ACTION_BUILD_MARINE or smart_action == ACTIONS.ACTION_BUILD_MAURADER:
                if barracks_y.any():
                    return actions.FunctionCall(ACTIONS.SELECT_POINT, [MISC.SELECT_ALL, self.prev_target])

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

            if smart_action == ACTIONS.ACTION_BUILD_MARINE:
                if ACTIONS.TRAIN_MARINE in obs.observation['available_actions']:
                    return actions.FunctionCall(ACTIONS.TRAIN_MARINE, [MISC.QUEUED])

                # It isn't available because we aren't building a tech lab on a barracks... lol
                # this is annoying QQ.
            elif smart_action == ACTIONS.ACTION_BUILD_MAURADER:
                if ACTIONS.TRAIN_MAURAUDER in obs.observation['available_actions']:
                    return actions.FunctionCall(ACTIONS.TRAIN_MAURAUDER, [MISC.QUEUED])

            elif smart_action == ACTIONS.ACTION_BUILD_BARRACKS or smart_action == ACTIONS.ACTION_BUILD_SUPPLY_DEPOT or smart_action == ACTIONS.ACTION_BUILD_REFINERY:
                if ACTIONS.HARVEST_GATHER in obs.observation['available_actions']:
                    unit_y, unit_x = (unit_type == UNITS.NEUTRAL_MINERAL_FIELD).nonzero()

                    if unit_y.any():
                        i = random.randint(0, len(unit_y) - 1)

                        m_x = unit_x[i]
                        m_y = unit_y[i]

                        target = [int(m_x), int(m_y)]

                        return actions.FunctionCall(ACTIONS.HARVEST_GATHER, [MISC.QUEUED, target])

        return actions.FunctionCall(ACTIONS.NO_OP, [])
