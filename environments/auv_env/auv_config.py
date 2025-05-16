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

    # shared thresholds
    DEFAULT_WALL_THRESH,

    # mode-specific rewards
    DEFAULT_DISCRETE_DOCK_REWARD,
    DEFAULT_CONTINUOUS_DOCK_REWARD,
    DEFAULT_DISCRETE_STEP_COST,
    DEFAULT_CONTINUOUS_STEP_COST,
    DEFAULT_DISCRETE_COLLISION_PENALTY,
    DEFAULT_CONTINUOUS_COLLISION_PENALTY,
    DEFAULT_DISCRETE_WALL_PENALTY_COEFF,
    DEFAULT_CONTINUOUS_WALL_PENALTY_COEFF,
    DEFAULT_DISCRETE_PROGRESS_COEFF,
    DEFAULT_CONTINUOUS_PROGRESS_COEFF,
    DEFAULT_DISCRETE_TURN_PENALTY_COEFF,
    DEFAULT_CONTINUOUS_TURN_PENALTY_COEFF,

    DEFAULT_USE_HISTORY,
    DEFAULT_HISTORY_LENGTH,

    DEFAULT_START_MODE,
    DEFAULT_SPAWN_CLEARANCE,

    DEFAULT_USE_PHYSICS,
    DEFAULT_MASS,
    DEFAULT_DRAG_COEF,
    DEFAULT_DT,
    DEFAULT_CURRENT_PARAMS,

    # actions
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
    death_limit: float = DEFAULT_DEATH_LIMIT

    # sonar
    n_beams: int = DEFAULT_N_BEAMS
    sonar_params: dict = field(default_factory=lambda: DEFAULT_SONAR_PARAMS.copy())

    # docks
    docks: Union[int, List] = DEFAULT_DOCKS
    dock_radius: float = DEFAULT_DOCK_RADIUS

    # reward structuring
    discrete_dock_reward: float = DEFAULT_DISCRETE_DOCK_REWARD
    continuous_dock_reward: float = DEFAULT_CONTINUOUS_DOCK_REWARD

    discrete_step_cost: float = DEFAULT_DISCRETE_STEP_COST
    continuous_step_cost: float = DEFAULT_CONTINUOUS_STEP_COST

    discrete_collision_penalty: float = DEFAULT_DISCRETE_COLLISION_PENALTY
    continuous_collision_penalty: float = DEFAULT_CONTINUOUS_COLLISION_PENALTY

    wall_thresh: float = DEFAULT_WALL_THRESH
    discrete_wall_penalty_coeff: float = DEFAULT_DISCRETE_WALL_PENALTY_COEFF
    continuous_wall_penalty_coeff: float = DEFAULT_CONTINUOUS_WALL_PENALTY_COEFF

    discrete_progress_coeff: float = DEFAULT_DISCRETE_PROGRESS_COEFF
    continuous_progress_coeff: float = DEFAULT_CONTINUOUS_PROGRESS_COEFF

    discrete_turn_penalty_coeff: float = DEFAULT_DISCRETE_TURN_PENALTY_COEFF
    continuous_turn_penalty_coeff: float = DEFAULT_CONTINUOUS_TURN_PENALTY_COEFF

    # history buffer
    use_history: bool = DEFAULT_USE_HISTORY
    history_length: int = DEFAULT_HISTORY_LENGTH

    # start/spawn
    start_mode: str = DEFAULT_START_MODE
    spawn_clearance: float = DEFAULT_SPAWN_CLEARANCE

    # action space
    use_discrete_actions: bool = DEFAULT_USE_DISCRETE_ACTIONS
    discrete_actions: Optional[List] = field(init=False)

    # physics
    use_physics: bool = DEFAULT_USE_PHYSICS
    mass: float = DEFAULT_MASS
    drag_coef: float = DEFAULT_DRAG_COEF
    dt: float = DEFAULT_DT
    current_params: dict = field(default_factory=lambda: DEFAULT_CURRENT_PARAMS.copy())

    def __post_init__(self):
        if self.use_discrete_actions:
            self.discrete_actions = DEFAULT_DISCRETE_ACTIONS.copy()
        else:
            self.discrete_actions = None
