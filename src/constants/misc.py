from pysc2.lib import features

PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
PLAYER_ID = features.SCREEN_FEATURES.player_id.index

PLAYER_SELF = 1
PLAYER_HOSTILE = 4
ARMY_SUPPLY = 5

NOT_QUEUED = [0]
QUEUED = [1]
SELECT_ALL = [2]

SCREEN = [0]

# Always square for now
MAP_SIZE = [63, 63]