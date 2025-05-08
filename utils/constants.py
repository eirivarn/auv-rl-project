from enum import IntEnum
import pygame

########## Grid Environment Constants ##########

class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

NUM_ACTIONS = 4
LIDAR_DIMENSION  = 4

# ---- Colors ----
WHITE = (255, 255, 255)
GRID_COLOR = (200, 200, 200)
GOAL_COLOR = (0, 255, 0)
AGENT_COLOR = (255, 0, 0)
OBSTACLE_COLOR = (0, 0, 255)

# ---- Default Hyperparameters ----
DEFAULT_GRID_SIZE = (5, 5)
DEFAULT_CELL_SIZE = 64
DEFAULT_FPS = 10
DEFAULT_LIDAR_RANGE = 10
DEFAULT_HISTORY_LENGTH = 3
DEFAULT_SPAWN_MODE = "static"
DEFAULT_USE_LIDAR = False
DEFAULT_USE_HISTORY = False
DEFAULT_OBSTACLE_COUNT = 0
DEFAULT_OBSTACLE_POSITIONS = None

# ---- Directions ----
DIRECTIONS = [
    (0, -1),  # UP
    (0, 1),   # DOWN
    (-1, 0),  # LEFT
    (1, 0),   # RIGHT
]

#---------------------------------------------#


########## Q-Learning Constants ##########

# ---- Default Hyperparameters ----
DEFAULT_ALPHA           = 0.1
DEFAULT_GAMMA           = 0.99
DEFAULT_EPSILON_START   = 1.0
DEFAULT_EPSILON_END     = 0.01
DEFAULT_EPSILON_DECAY   = 0.995   

# ---- Q-Learning Training ----
DEFAULT_TRAINING_EPISODES = 10000
DEFAULT_TRAINING_MAX_STEPS = 100
DEFAULT_EVALUATION_EPISODES = 100
DEFAULT_EVALUATION_MAX_STEPS = 100

#---------------------------------------------#

