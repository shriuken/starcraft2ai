from pysc2.lib import actions

NO_OP = actions.FUNCTIONS.no_op.id
SELECT_POINT = actions.FUNCTIONS.select_point.id
BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
BUILD_REFINERY = actions.FUNCTIONS.Build_Refinery_screen.id
BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
BUILD_TECH_LAB = actions.FUNCTIONS.Build_TechLab_screen.id
TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id
TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
TRAIN_MAURAUDER = actions.FUNCTIONS.Train_Marauder_quick.id
SELECT_ARMY = actions.FUNCTIONS.select_army.id
HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id
SMART_SCREEN = actions.FUNCTIONS.Smart_screen.id
ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_SCV = 'selectscv'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_BUILD_REFINERY = 'buildrefinery'
ACTION_BUILD_TECH_LAB = 'buildtechlab'
ACTION_SELECT_BARRACKS = 'selectbarracks'
ACTION_BUILD_SCV = 'buildscv'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_BUILD_MAURADER = 'buildmaurader'
ACTION_SELECT_ARMY = 'selectarmy'
ACTION_ATTACK = 'attack'
ACTION_MINE_VESPENE = 'minevespene'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_BUILD_MARINE,
    ACTION_MINE_VESPENE,
    ACTION_BUILD_REFINERY,
    ACTION_BUILD_MAURADER,
    ACTION_BUILD_TECH_LAB,
    ACTION_BUILD_SCV,
    ACTION_ATTACK,
]

# for mm_x in range(0, 64):
#     for mm_y in range(0, 64):
#         if mm_x % 3 == 0 and mm_y % 3 == 0:
#             smart_actions.append(ACTION_DO_NOTHING)
#             smart_actions.append(ACTION_BUILD_MARINE)
#             smart_actions.append(ACTION_BUILD_BARRACKS + '_' + str(mm_x) + '_' + str(mm_y))
#             smart_actions.append(ACTION_BUILD_SUPPLY_DEPOT + '_' + str(mm_x) + '_' + str(mm_y))
#             smart_actions.append(ACTION_ATTACK + '_' + str(mm_x) + '_' + str(mm_y))
