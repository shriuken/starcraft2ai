import random
import math
import numpy as np

import constants.actions as ACTIONS
import constants.units as UNITS
import constants.misc as MISC

from pysc2.lib import actions


def in_range(target):
    return MISC.MAP_SIZE[0] > target[0] > 0 and MISC.MAP_SIZE[1] > target[1] > 0


def transform_distance(x, x_distance, y, y_distance, base_top_left):
    if not base_top_left:
        return [x - x_distance, y - y_distance]

    return [x + x_distance, y + y_distance]


def transform_location(base_top_left, x, y):
    if x > MISC.MAP_SIZE[0]:
        x = MISC.MAP_SIZE[0]
    if y > MISC.MAP_SIZE[1]:
        y = MISC.MAP_SIZE[1]

    if not base_top_left:
        return [MISC.MAP_SIZE[0] - abs(x), MISC.MAP_SIZE[1] - abs(y)]

    return [abs(x), abs(y)]


def split_action(action_id):
    if action_id is None:
        return ACTIONS.smart_actions[0], 0, 0
    smart_action = ACTIONS.smart_actions[action_id]

    x = 0
    y = 0
    if '_' in smart_action:
        smart_action, x, y = smart_action.split('_')

    return smart_action, x, y


def get_action(obs, base_top_left, move_number, cc_count, supply_depot_count,
               barracks_count, previous_state, previous_action, qlearn,
               unit_type, barracks_x, barracks_y, cc_x, cc_y):
    if move_number == 0:
        move_number += 1

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

        if not base_top_left:
            hot_squares = hot_squares[::-1]

        for i in range(0, 4):
            current_state[i + 4] = hot_squares[i]

        if previous_action is not None:
            qlearn.learn(str(previous_state), previous_action, 0, str(current_state))

        rl_action = qlearn.choose_action(str(current_state))

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

    elif move_number == 1:
        move_number += 1

        smart_action, x, y = split_action(previous_action)

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

            if len(obs.observation['single_select']) > 0 and obs.observation['single_select'][0][0] == UNITS.TERRAN_SCV:
                do_it = False

            if len(obs.observation['multi_select']) > 0 and obs.observation['multi_select'][0][0] == UNITS.TERRAN_SCV:
                do_it = False

            if do_it and ACTIONS.ATTACK_MINIMAP in obs.observation["available_actions"]:
                x_offset = random.randint(-1, 1)
                y_offset = random.randint(-1, 1)

                return actions.FunctionCall(ACTIONS.ATTACK_MINIMAP, [MISC.NOT_QUEUED,
                                                                     transform_location(int(x) + (x_offset * 8),
                                                                                        int(y) + (y_offset * 8))])

    elif move_number == 2:
        move_number = 0

        smart_action, x, y = split_action(previous_action)

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
