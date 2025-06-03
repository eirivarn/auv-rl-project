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
DEFAULT_DT = 0.1

# ---- Rewards, Penalties and Shaping ----
# Sparse dock reward (event-based, same for both)
DEFAULT_DISCRETE_DOCK_REWARD    = 500.0
DEFAULT_CONTINUOUS_DOCK_REWARD  = 10000.0

# Step cost (per action vs per second)
DEFAULT_DISCRETE_STEP_COST      = -1.0
DEFAULT_CONTINUOUS_STEP_COST    = -0.2

# Wall proximity penalty coefficient
DEFAULT_WALL_THRESH                = 1.0
DEFAULT_DISCRETE_WALL_PENALTY_COEFF   = 20.0
DEFAULT_CONTINUOUS_WALL_PENALTY_COEFF = 10.0

# Progress reward coefficient
DEFAULT_DISCRETE_PROGRESS_COEFF      = 3.0
DEFAULT_CONTINUOUS_PROGRESS_COEFF    = 1.0

# Turn penalty coefficient
DEFAULT_DISCRETE_TURN_PENALTY_COEFF     = 30.0
DEFAULT_CONTINUOUS_TURN_PENALTY_COEFF   = 1.0

# ---- Physics ----
DEFAULT_USE_PHYSICS = False
DEFAULT_MASS = 1.0
DEFAULT_DRAG_COEF = 0.1
DEFAULT_DT = 0.1
DEFAULT_CURRENT_PARAMS = {
    "current_speed": 0.0,
    "current_direction": 0.0,
    "current_noise_std": 0.0,
}

# ---- Action Space ----
DEFAULT_DISCRETE_ACTIONS = [
    (0.3,  0.0),  # forward
    (0.3,  0.3),  # forward + left
    (0.3, -0.3),  # forward + right
    (0.0,  0.3),  # in-place left turn
    (0.0, -0.3),  # in-place right turn
]

DEFAULT_MAX_THRUST = 0.4
DEFAULT_MAX_TORQUE = 0.4

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
