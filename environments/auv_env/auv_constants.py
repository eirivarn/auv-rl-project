from enum import IntEnum
import numpy as np

########## AUV Environment Constants ##########

DEFAULT_AUV_GRID_SIZE = (200, 200)
DEFAULT_RESOLUTION = 0.1
DEFAULT_DOCKS = 1
DEFAULT_DOCK_RADIUS = 1.0
DEFAULT_USE_HISTORY = False
DEFAULT_HISTORY_LENGTH = 6
DEFAULT_WINDOW_SIZE = (800, 600)
DEFAULT_START_MODE = 'random'
DEFAULT_SPAWN_CLEARANCE = 1.5
DEFAULT_RANDOM_MAP = False
DEFAULT_MAP_FILL_PROB = 0.32
DEFAULT_SMOOTH_STEPS = 9
DEFAULT_BIRTH_LIMIT = 4
DEFAULT_DEATH_LIMIT = 4
DEFAULT_USE_DISCRETE_ACTIONS = True
DEFAULT_USE_DISCRETE_ACTIONS = True

# ---- Rewards, Penalties and Shaping ----
DEFAULT_DOCK_REWARD = 500.0
DEFAULT_WALL_THRESH = 1.0
DEFAULT_WALL_PENALTY_COEFF = 20.0
DEFAULT_COLLISION_PENALTY = 5.0
DEFAULT_PROGRESS_COEFF = 3.0
DEFAULT_TURN_PENALTY_COEFF = 30.0

# ---- Action Space ----
DEFAULT_DISCRETE_ACTIONS = [
    (0.3,  0.0),  # forward
    (0.3,  0.3),  # forward + left
    (0.3, -0.3),  # forward + right
    (0.0,  0.3),  # in-place left turn
    (0.0, -0.3),  # in-place right turn
]

# ---- Sonar Parameters ----
DEFAULT_N_BEAMS = 20
DEFAULT_SONAR_PARAMS = {
    "fov":                np.deg2rad(360),
    "n_beams":            DEFAULT_N_BEAMS,
    "max_range":          10.0,
    "resolution":         DEFAULT_RESOLUTION,
    "noise_std":          0.0,
    "compute_intensity":  False,
    "spreading_loss":     False,
    "debris_rate":        0,
    "ghost_prob":         0.0,
    "ghost_decay":        0.0,
}

