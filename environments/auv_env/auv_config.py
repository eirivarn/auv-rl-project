# config.py

from dataclasses import dataclass, field
from typing import List, Optional, Union

from .auv_constants import (
    DEFAULT_AUV_GRID_SIZE,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_RESOLUTION,
    DEFAULT_RANDOM_MAP,
    DEFAULT_MAP_FILL_PROB,
    DEFAULT_SMOOTH_STEPS,
    DEFAULT_BIRTH_LIMIT,
    DEFAULT_DEATH_LIMIT,

    DEFAULT_N_BEAMS,
    DEFAULT_SONAR_PARAMS,

    DEFAULT_DOCKS,
    DEFAULT_DOCK_RADIUS,
    DEFAULT_DOCK_REWARD,

    DEFAULT_USE_HISTORY,
    DEFAULT_HISTORY_LENGTH,

    DEFAULT_START_MODE,
    DEFAULT_SPAWN_CLEARANCE,

    DEFAULT_WALL_THRESH,
    DEFAULT_WALL_PENALTY_COEFF,
    DEFAULT_COLLISION_PENALTY,
    DEFAULT_PROGRESS_COEFF,
    DEFAULT_TURN_PENALTY_COEFF,

    DEFAULT_USE_DISCRETE_ACTIONS,
    DEFAULT_DISCRETE_ACTIONS,
)

@dataclass
class AUVEnvConfig:
    # grid and viewport
    grid_size: tuple = DEFAULT_AUV_GRID_SIZE
    window_size: tuple = DEFAULT_WINDOW_SIZE
    resolution: float = DEFAULT_RESOLUTION

    # map generation
    random_map: bool = DEFAULT_RANDOM_MAP
    map_fill_prob: float = DEFAULT_MAP_FILL_PROB
    smooth_steps: int = DEFAULT_SMOOTH_STEPS
    birth_limit: int = DEFAULT_BIRTH_LIMIT
    death_limit: int = DEFAULT_DEATH_LIMIT

    # sonar
    n_beams: int = DEFAULT_N_BEAMS
    sonar_params: dict = field(default_factory=lambda: DEFAULT_SONAR_PARAMS.copy())

    # docks
    docks: Union[int, List] = DEFAULT_DOCKS
    dock_radius: float = DEFAULT_DOCK_RADIUS
    dock_reward: float = DEFAULT_DOCK_REWARD

    # history buffer
    use_history: bool = DEFAULT_USE_HISTORY
    history_length: int = DEFAULT_HISTORY_LENGTH

    # start/spawn
    start_mode: str = DEFAULT_START_MODE
    spawn_clearance: float = DEFAULT_SPAWN_CLEARANCE

    # reward shaping
    wall_thresh: float = DEFAULT_WALL_THRESH
    wall_penalty_coeff: float = DEFAULT_WALL_PENALTY_COEFF
    collision_penalty: float = DEFAULT_COLLISION_PENALTY
    progress_coeff: float = DEFAULT_PROGRESS_COEFF
    turn_penalty_coeff: float = DEFAULT_TURN_PENALTY_COEFF

    # action space
    use_discrete_actions: bool = DEFAULT_USE_DISCRETE_ACTIONS
    discrete_actions: Optional[List] = field(init=False)

    def __post_init__(self):
        # initialize discrete actions if needed
        if self.use_discrete_actions:
            self.discrete_actions = DEFAULT_DISCRETE_ACTIONS.copy()
        else:
            self.discrete_actions = None
