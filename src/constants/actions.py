from pysc2.lib import actions

NO_OP = actions.FUNCTIONS.no_op.id
SELECT_POINT = actions.FUNCTIONS.select_point.id
BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
SELECT_ARMY = actions.FUNCTIONS.select_army.id
HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id
ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

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

for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if (mm_x + 1) % 32 == 0 and (mm_y + 1) % 32 == 0:
            smart_actions.append(ACTION_ATTACK + '_' + str(mm_x - 16) + '_' + str(mm_y - 16))