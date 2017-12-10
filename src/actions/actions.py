import random

import constants.actions as ACTIONS
import constants.units as UNITS
import constants.misc as MISC

from pysc2.lib import actions


def in_range(target):
    return MISC.MAP_SIZE[0] > target[0] > 0 and MISC.MAP_SIZE[1] > target[1] > 0


def transform_location(base_top_left, x, x_distance, y, y_distance):
    if not base_top_left:
        return [x - x_distance, y - y_distance]

    return [x + x_distance, y + y_distance]


def get_action(obs, base_top_left, smart_action):
    if smart_action == ACTIONS.ACTION_DO_NOTHING:
        return actions.FunctionCall(ACTIONS.NO_OP, [])

    elif smart_action == ACTIONS.ACTION_SELECT_SCV:
        unit_type = obs.observation['screen'][MISC.UNIT_TYPE]
        unit_y, unit_x = (unit_type == UNITS.TERRAN_SCV).nonzero()

        if unit_y.any():
            i = random.randint(0, len(unit_y) - 1)
            target = [unit_x[i], unit_y[i]]
            if in_range(target):
                return actions.FunctionCall(ACTIONS.SELECT_POINT, [MISC.SCREEN, target])

    elif smart_action == ACTIONS.ACTION_BUILD_SUPPLY_DEPOT:
        if ACTIONS.BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
            unit_type = obs.observation['screen'][MISC.UNIT_TYPE]
            unit_y, unit_x = (unit_type == UNITS.TERRAN_COMMANDCENTER).nonzero()

            if unit_y.any():
                target = transform_location(base_top_left, int(unit_x.mean()), 0, int(unit_y.mean()), 20)
                if in_range(target):
                    return actions.FunctionCall(ACTIONS.BUILD_SUPPLY_DEPOT, [MISC.SCREEN, target])

    elif smart_action == ACTIONS.ACTION_BUILD_BARRACKS:
        if ACTIONS.BUILD_BARRACKS in obs.observation['available_actions']:
            unit_type = obs.observation['screen'][MISC.UNIT_TYPE]
            unit_y, unit_x = (unit_type == UNITS.TERRAN_COMMANDCENTER).nonzero()

            if unit_y.any():
                target = transform_location(base_top_left, int(unit_x.mean()), 20, int(unit_y.mean()), 0)

                if in_range(target):
                    return actions.FunctionCall(ACTIONS.BUILD_BARRACKS, [MISC.SCREEN, target])

    elif smart_action == ACTIONS.ACTION_SELECT_BARRACKS:
        unit_type = obs.observation['screen'][MISC.UNIT_TYPE]
        unit_y, unit_x = (unit_type == UNITS.TERRAN_BARRACKS).nonzero()

        if unit_y.any():
            target = [int(unit_x.mean()), int(unit_y.mean())]

            if in_range(target):
                return actions.FunctionCall(ACTIONS.SELECT_POINT, [MISC.SCREEN, target])

    elif smart_action == ACTIONS.ACTION_BUILD_MARINE:
        if ACTIONS.TRAIN_MARINE in obs.observation['available_actions']:
            return actions.FunctionCall(ACTIONS.TRAIN_MARINE, [[1]])

    elif smart_action == ACTIONS.ACTION_SELECT_ARMY:
        if ACTIONS.SELECT_ARMY in obs.observation['available_actions']:
            return actions.FunctionCall(ACTIONS.SELECT_ARMY, [[0]])

    elif smart_action == ACTIONS.ACTION_ATTACK:
        if ACTIONS.ATTACK_MINIMAP in obs.observation["available_actions"]:
            if base_top_left:
                return actions.FunctionCall(ACTIONS.ATTACK_MINIMAP, [[1], [39, 45]])

            return actions.FunctionCall(ACTIONS.ATTACK_MINIMAP, [[1], [21, 24]])

    return actions.FunctionCall(ACTIONS.NO_OP, [])
