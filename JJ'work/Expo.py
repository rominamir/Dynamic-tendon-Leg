import os
#import gymnasium as gym

import mujoco
import numpy as np
from gym import utils, error
import gym
import mujoco_env
from math import exp

class ExpoLegEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 40}

    def __init__(self, xml_file='leg.xml', render_mode='none', seed=None,
                 stiffness_start=1000, stiffness_end=10000, num_timesteps=100_000,
                 max_episode_steps=500, growth_factor=0.01):
        """
        Exponential stiffness growth environment where the stiffness increases following an
        exponential function with a controllable growth factor.

        Parameters:
        - stiffness_start: Initial stiffness value.
        - stiffness_end: Maximum stiffness value.
        - num_timesteps: Total training timesteps.
        - max_episode_steps: Maximum steps per episode.
        - growth_factor: Controls the exponential growth trend (higher = faster increase).
        """
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        observation_space = gym.spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float32)

        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5, render_mode=render_mode, observation_space=observation_space)

        if seed is not None:
            self.seed(seed)

        self.stiffness_start = stiffness_start
        self.stiffness_end = stiffness_end
        self.num_timesteps = num_timesteps
        self.max_episode_steps = max_episode_steps
        self.growth_factor = growth_factor  # New parameter to control exponential growth
        self.episode_count = 0

        self.total_episodes = num_timesteps / max_episode_steps
        self.stiffness_scaling = self.stiffness_start
        self.stiffness_history = []
        self.distances = []

    def update_stiffness(self, episode_number):
        """
        Adjusts the stiffness value using an exponential function where the rate of increase
        is controlled by `growth_factor`.
        """
        # Compute progress ratio (normalized between 0 and 1)
        progress = episode_number / max(1, self.total_episodes)

        # Exponential growth equation with controllable growth factor
        exponent = self.growth_factor * progress * self.total_episodes
        self.stiffness_scaling = self.stiffness_start + (
            (self.stiffness_end - self.stiffness_start) *
            (np.exp(exponent) - 1) / (np.exp(self.growth_factor * self.total_episodes) - 1)
        )

        # Ensure the stiffness does not exceed the predefined maximum
        self.stiffness_scaling = min(self.stiffness_scaling, self.stiffness_end)
        self.apply_stiffness(self.stiffness_scaling)
        self.stiffness_history.append(self.stiffness_scaling)

        print(f"Episode: {episode_number}, Stiffness: {self.stiffness_scaling}")

    def step(self, action):
        """
        Simulates one step in the environment, computing reward based on movement speed.
        """
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        # Reward function: encourage forward movement and penalize backward movement
        reward = max(x_velocity, 0) - max(-x_velocity, 0)

        # Increment episode step count
        self.steps_from_reset += 1
        done = self.steps_from_reset >= self.max_episode_steps

        if done:
            self.update_stiffness(self.episode_count)
            self.episode_count += 1

        return self.get_obs(), reward, done, False, {}

    def reset_model(self):
        """
        Resets the environment state at the beginning of each episode and updates stiffness.
        """
        self.update_stiffness(self.episode_count)
        self.steps_from_reset = 0
        self.start_x_position = self.data.qpos[0]
        return self.get_obs()

    def apply_stiffness(self, stiffness_value):
        """
        Applies the current stiffness scaling value to all tendons in the model.
        """
        for i in range(len(self.model.tendon_stiffness)):
            self.model.tendon_stiffness[i] = stiffness_value
        print(f"Applied stiffness: {stiffness_value}")

    def get_obs(self):
        """
        Retrieves the current state observation (positions and velocities).
        """
        position_tmp = self.data.qpos.flat.copy()
        velocity_tmp = self.data.qvel.flat.copy()
        return np.concatenate((position_tmp, velocity_tmp)).ravel()

    def seed(self, seed=None):
        np.random.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def save_stiffness_history(self, filename):
        """
        Saves the history of stiffness values to a file.
        """
        np.save(filename, np.array(self.stiffness_history))