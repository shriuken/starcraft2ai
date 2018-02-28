import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.distributions import Categorical

from pysc2.agents import base_agent
from actions.actions import *
from pysc2.lib import actions

DATA_FILE = 'sparse_agent_data'


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(7, 64, kernel_size=6, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=4, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        # Merging in current state
        self.lin1 = nn.Linear(8, 128)

        self.affine1 = nn.Linear(3200, 128)
        self.affine2 = nn.Linear(128, len(ACTIONS.smart_actions))

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x, y):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.affine1(x.view(x.size(0), -1)))
        x = torch.cat((F.relu(self.lin1(y)), x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


def select_action(policy, state, state2):
    state = torch.from_numpy(state).float().unsqueeze(0)
    state2 = torch.from_numpy(state2).float().unsqueeze(0)
    probs = policy(Variable(state), Variable(state2))
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.data[0]


def finish_episode(policy, optimizer):
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + 0.99 * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]
    torch.save(policy, 'policy.pt')
    torch.save(optimizer, 'optimizer.pt')


class RLAgent(base_agent.BaseAgent):
    def __init__(self):
        super(RLAgent, self).__init__()

        self.policy = Policy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)

        self.previous_action = None
        self.previous_state = None
        self.previous_cumulative_score_total = 0
        self.previous_killed_building_score = 0
        self.previous_killed_units_score = 0
        self.won = 0
        self.lost = 0


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

        if obs.last():
            reward = (obs.reward * 25)

            self.policy.rewards.append(reward)
            # plt.plot(self.policy.rewards)
            # plt.show()
            self.previous_action = None
            self.previous_state = None

            self.move_number = 0
            # Train our network.
            finish_episode(self.policy, self.optimizer)
            if obs.reward > 0:
                self.won += 1
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

            hot_squares = np.zeros(4)
            enemy_y, enemy_x = (obs.observation['minimap'][MISC.PLAYER_RELATIVE] == MISC.PLAYER_HOSTILE).nonzero()
            for i in range(0, len(enemy_y)):
                y = int(math.ceil((enemy_y[i] + 1) / 32)) - 1
                x = int(math.ceil((enemy_x[i] + 1) / 32)) - 1

                hot_squares[((y - 1) * 2) + (x - 1)] = 1

            if not self.base_top_left:
                hot_squares = hot_squares[::-1]

            for i in range(0, 4):
                current_state[i + 4] = hot_squares[i]

            if self.previous_action is not None:
                reward = obs.observation['score_cumulative'][0] - self.previous_cumulative_score_total
                killed_unit_score = obs.observation['score_cumulative'][5]
                killed_building_score = obs.observation['score_cumulative'][6]

                # if killed_building_score > self.previous_killed_building_score:
                #     reward += 1
                # if killed_unit_score > self.previous_killed_units_score:
                #     reward += 0.1

                self.policy.rewards.append(reward)
                # finish_episode(self.policy, self.optimizer)
                self.previous_cumulative_score_total = obs.observation['score_cumulative'][0]
                self.previous_killed_building_score = killed_building_score
                self.previous_killed_units_score = killed_unit_score

            rl_action = select_action(self.policy, obs.observation['minimap'], current_state)

            self.previous_state = current_state
            self.previous_action = rl_action

            smart_action, x, y = split_action(self.previous_action)

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
                if supply_depot_count < 8 and ACTIONS.BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
                    if cc_y.any():
                        target = [int(x), int(y)]

                        return actions.FunctionCall(ACTIONS.BUILD_SUPPLY_DEPOT, [MISC.NOT_QUEUED, target])

            elif smart_action == ACTIONS.ACTION_BUILD_BARRACKS:
                if barracks_count < 5 and ACTIONS.BUILD_BARRACKS in obs.observation['available_actions']:
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
                    x_offset = random.randint(-1, 1)
                    y_offset = random.randint(-1, 1)

                    return actions.FunctionCall(ACTIONS.ATTACK_MINIMAP, [MISC.NOT_QUEUED,
                                                                         transform_location(self.base_top_left,
                                                                                            int(x) + (x_offset * 8),
                                                                                            int(y) + (y_offset * 8))
                                                                         ])

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
