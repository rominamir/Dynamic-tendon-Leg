import os
import numpy as np
from datetime import datetime
import gym
from gym import utils
import mujoco
import mujoco_env
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo

class LegEnvBase(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    A custom leg environment based on MujocoEnv.
    This environment updates tendon stiffness over time according to different growth strategies (e.g. exponential).
    """

    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 40}

    def __init__(
        self,
        xml_file='leg.xml',
        render_mode='none',
        seed=None,
        stiffness_start=1000,
        stiffness_end=20000,
        num_timesteps=100_000,
        max_episode_steps=500,
        growth_factor=5,
        growth_type='exponential'
    ):
        # Define action and observation spaces
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float32)

        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(
            self,
            xml_file,
            5,
            render_mode=render_mode,
            observation_space=self.observation_space
        )

        # If a seed is provided, set it for reproducibility
        if seed is not None:
            self.seed(seed)

        # Key parameters for stiffness progression
        self.stiffness_start = stiffness_start
        self.stiffness_end = stiffness_end
        self.num_timesteps = num_timesteps
        self.max_episode_steps = max_episode_steps
        self.growth_factor = growth_factor
        self.growth_type = growth_type

        # Counters for steps in current episode and global steps
        self.steps_from_reset = 0
        self.global_step = 0  # Accumulates steps across all episodes

        # Variables to track stiffness values
        self.stiffness_scaling = self.stiffness_start
        self.stiffness_history = []
        self.distances = []

    def update_stiffness(self, global_step):
        """
        Updates the tendon stiffness based on the global_step.

        :param global_step: The total number of steps taken across all training episodes.
        """
        progress = global_step / max(1, self.num_timesteps)

        if self.growth_type == 'exponential':
            # Exponential growth
            exponent = self.growth_factor * progress
            self.stiffness_scaling = self.stiffness_start + (
                (self.stiffness_end - self.stiffness_start)
                * (np.exp(exponent) - 1)
                / (np.exp(self.growth_factor) - 1)
            )
        elif self.growth_type == 'logarithmic':
            # Logarithmic growth
            adjusted_progress = 1 - np.exp(-self.growth_factor * progress)
            self.stiffness_scaling = self.stiffness_start + (
                (self.stiffness_end - self.stiffness_start) * adjusted_progress
            )
        elif self.growth_type == 'linear':
            # Linear growth
            self.stiffness_scaling = self.stiffness_start + (
                progress * (self.stiffness_end - self.stiffness_start)
            )
        elif self.growth_type == 'constant':
            # No change in stiffness
            self.stiffness_scaling = self.stiffness_start

        # Ensure it does not exceed the final stiffness
        self.stiffness_scaling = min(self.stiffness_scaling, self.stiffness_end)

        # Apply the updated stiffness and record its value
        self.apply_stiffness(self.stiffness_scaling)
        self.stiffness_history.append(self.stiffness_scaling)

    def step(self, action):
        """
        Executes a simulation step given the agent's action.

        :param action: Agent's action (continuous).
        :return: Observation, reward, done, truncated, info (standard gym format).
        """
        # Obtain the position before simulation
        x_position_before = self.data.qpos[0]
        # Run simulation for frame_skip steps
        self.do_simulation(action, self.frame_skip)
        # Position after simulation
        x_position_after = self.data.qpos[0]

        # Simple reward = forward velocity - backward velocity
        x_velocity = (x_position_after - x_position_before) / self.dt
        reward = max(x_velocity, 0) - max(-x_velocity, 0)

        # Increment episode and global step counters
        self.steps_from_reset += 1
        self.global_step += 1

        # Update stiffness based on the global step
        self.update_stiffness(self.global_step)

        # Check if the episode has ended
        done = (self.steps_from_reset >= self.max_episode_steps)

        return self.get_obs(), reward, done, False, {}

    def reset_model(self):
        """
        Resets the environment state at the beginning of each episode.
        """
        self.steps_from_reset = 0
        # Optionally update stiffness here if desired, e.g.:
        # self.update_stiffness(self.global_step)

        return self.get_obs()

    def apply_stiffness(self, stiffness_value):
        """
        Apply the current stiffness to all tendons in the Mujoco model.
        """
        for i in range(len(self.model.tendon_stiffness)):
            self.model.tendon_stiffness[i] = stiffness_value

    def get_obs(self):
        """
        Constructs the observation by concatenating positions and velocities.
        """
        return np.concatenate(
            (self.data.qpos.flat.copy(), self.data.qvel.flat.copy())
        ).ravel()

    def seed(self, seed=None):
        """
        Sets the random seed for numpy and the gym spaces (if applicable).
        """
        np.random.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def save_stiffness_history(self, filename):
        """
        Saves the recorded stiffness history to a file as a numpy array.
        """
        os.makedirs(os.path.dirname(os.path.normpath(filename)), exist_ok=True)
        np.save(filename, np.array(self.stiffness_history))

    def save_distances(self, filename):
        """
        Saves distance data (if collected) to a file as a numpy array.
        """
        os.makedirs(os.path.dirname(os.path.normpath(filename)), exist_ok=True)
        np.save(filename, self.distances)

    def load_distances(self, filename):
        """
        Loads distance data from a file into the environment.
        """
        self.distances = np.load(filename).tolist()


def train_env(seed_value, train=True, growth_factor=0.05, growth_type='exponential'):
    """
    Demonstration of how to train the LegEnvBase environment with PPO.
    Creates directories for logs, models, etc., and then runs the training.

    :param seed_value: Random seed to ensure reproducibility.
    :param train: Whether to actually train the model (True) or just do a demonstration.
    :param growth_factor: Factor used in exponential/logarithmic growth of stiffness.
    :param growth_type: Type of stiffness progression ('exponential', 'logarithmic', 'linear', 'constant').
    """
    # Define folder name for logs and data
    folder = f'LegEnv_{datetime.now().strftime("%b%d")}_{growth_type}'
    os.makedirs(f"./tensorboard_log/{folder}/model/", exist_ok=True)
    os.makedirs(f"./data/{folder}/distance/", exist_ok=True)
    os.makedirs(f"./data/{folder}/stiffness/", exist_ok=True)
    os.makedirs(f"./videos/{folder}/", exist_ok=True)

    # Create environment
    env = LegEnvBase(
        render_mode=None,
        growth_factor=growth_factor,
        growth_type=growth_type
    )
    env.seed(seed_value)

    # Create and configure PPO model
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        seed=seed_value,
        tensorboard_log=f"./tensorboard_log/{folder}/ppo_seed{seed_value}",
        gamma=0.9,
        ent_coef=0.01,
        learning_rate=lambda p: p * 0.0003
    )

    # Train for 100k timesteps
    model.learn(total_timesteps=100_000)

    # Save the trained model
    model.save(f"./tensorboard_log/{folder}/model/PPO_seed_{seed_value}_{growth_type}.model")

    # Save distances and stiffness history
    env.save_distances(f'./data/{folder}/distance/distance_history_seed_{seed_value}_{growth_type}.npy')
    env.save_stiffness_history(f'./data/{folder}/stiffness/stiffness_history_seed_{seed_value}_{growth_type}.npy')

    env.close()