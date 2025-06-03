# RL for Autonomous Navigation

This project uses reinforcement learning (RL) to train agents for navigation in simulated grid-world and AUV (underwater vehicle) environments. We explore different RL methods (Q-Learning, DQN, TD3) and how they handle obstacles and different ways of representing the environment.

## Project Setup

Here's how to get this project running:

### Requirements

* **Python 3.11**

### Installation

1.  **Get the code:**
    ```bash
    git clone [https://github.com/eirivarn/auv-rl-project.git](https://github.com/eirivarn/auv-rl-project.git)
    cd auv-rl-project
    ```

2.  **Set up your environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install necessary libraries:**
    ```bash
    pip install -r requirements.txt
    ```

---

## How to Use It

You can run agents directly from Python scripts or use the Jupyter notebooks for interactive work.

The `notebooks/` folder contains `auv_notebook.ipynb` and `grid_notebook.ipynb`. These notebooks show you how to:
* Set up environments.
* Train agents.
* See how agents perform.
* Visualize learning.

---

## Project Folders

Here's a breakdown of the key directories and their contents:

* `agents/`:
    * Contains the core implementations of our reinforcement learning agents: `dqn_agent.py` (Deep Q-Network), `q_learning_agent.py` (Q-Learning), and `td3_agent.py` (Twin Delayed DDPG).

* `environments/`:
    * Defines the simulation environments where the agents learn and operate.
    * `auv_env/`: Dedicated files for the Autonomous Underwater Vehicle (AUV) environment, including `auv_env.py` (the environment itself), `auv_config.py` (configuration settings), `auv_constants.py` (physical constants), and `auv_utils.py` (helper functions for the AUV environment).
    * `grid_env/`: Files for the grid-world environment, featuring `grid_env.py` (the grid environment), `grid_constants.py`, and `grid_utils.py` for grid-specific utilities.

* `gifs/`:
    * Stores various animated GIFs generated from agent training and evaluation runs.
    * `demo_gifs/`: A subfolder containing example GIFs showcasing different agent behaviors, successful navigation, or specific challenges encountered.

* `models/`:
    * Contains pre-trained agent models, primarily in PyTorch's `.pth` format for neural networks (e.g., `dqn_gridenv.pth`, `td3_auv.pth`) and NumPy's `.npy` format for Q-tables (e.g., `q_table.npy`). These allow for quick evaluation or continuation of training.

* `notebooks/`:
    * Jupyter notebooks for interactive development, experimentation, and analysis.
    * `auv_notebook.ipynb`: Demonstrates how to set up, train, and evaluate agents in the AUV environment.
    * `grid_notebook.ipynb`: Provides similar demonstrations for the grid-world environment.

* `report/`:
    * Holds the project's official documentation in PDF format.
    * `Autonomous_and_Adaptive_Systems_Project__short.pdf`: A concise summary of the project.
    * `Autonomous_and_Adaptive_Systems_Project_long.pdf`: A detailed report covering methodology, results, and conclusions.

* `requirements.txt`:
    * Lists all the Python libraries and their versions required to run this project, ensuring consistent environments.

* `utils/`:
    * Contains general utility scripts shared across the project.
    * `graphics_utils.py`: Functions for rendering and visualizing the environments and agent actions.
    * `shared_utils.py`: Other common helper functions used by agents or environments.

---

## Reports and Visuals

* Check out the `report/` folder for our detailed project reports.
* See the `gifs/` folder for animations of agents in action.

