import os
import numpy as np
from datetime import datetime
import gym
from gym import utils
import mujoco
import mujoco_env
from stable_baselines3 import PPO, A2C
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
        stiffness_start=2000,
        stiffness_end=20000,
        num_timesteps=100_000,
        max_episode_steps=500,
        growth_factor=0.03,
        growth_type='exponential'
    ):
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

        if seed is not None:
            self.seed(seed)

        self.stiffness_start = stiffness_start
        self.stiffness_end = stiffness_end
        self.num_timesteps = num_timesteps
        self.max_episode_steps = max_episode_steps
        self.growth_factor = growth_factor
        self.growth_type = growth_type

        self.steps_from_reset = 0
        self.global_step = 0

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
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]

        x_velocity = (x_position_after - x_position_before) / self.dt
        reward = max(x_velocity, 0) - max(-x_velocity, 0)

        self.steps_from_reset += 1
        self.global_step += 1

        self.update_stiffness(self.global_step)

        done = (self.steps_from_reset >= self.max_episode_steps)

        return self.get_obs(), reward, done, False, {}

    def reset_model(self):
        self.steps_from_reset = 0
        return self.get_obs()

    def apply_stiffness(self, stiffness_value):
        for i in range(len(self.model.tendon_stiffness)):
            self.model.tendon_stiffness[i] = stiffness_value

    def get_obs(self):
        return np.concatenate(
            (self.data.qpos.flat.copy(), self.data.qvel.flat.copy())
        ).ravel()

    def seed(self, seed=None):
        np.random.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def save_stiffness_history(self, filename):
        os.makedirs(os.path.dirname(os.path.normpath(filename)), exist_ok=True)
        np.save(filename, np.array(self.stiffness_history))

    def save_distances(self, filename):
        os.makedirs(os.path.dirname(os.path.normpath(filename)), exist_ok=True)
        np.save(filename, self.distances)

    def load_distances(self, filename):
        self.distances = np.load(filename).tolist()


def train_env(seed_value, algorithm='PPO', growth_factor=0.03, growth_type='exponential'):
    folder = f'LegEnv_{datetime.now().strftime("%b%d")}_{growth_type}_{algorithm}'
    os.makedirs(f"./tensorboard_log/{folder}/model/", exist_ok=True)
    os.makedirs(f"./data/{folder}/distance/", exist_ok=True)
    os.makedirs(f"./data/{folder}/stiffness/", exist_ok=True)
    os.makedirs(f"./videos/{folder}/", exist_ok=True)

    env = LegEnvBase(
        render_mode=None,
        growth_factor=growth_factor,
        growth_type=growth_type
    )
    env.seed(seed_value)

    if algorithm == 'PPO':
        model = PPO('MlpPolicy', env, verbose=1, seed=seed_value, tensorboard_log=f"./tensorboard_log/{folder}/ppo")
    elif algorithm == 'A2C':
        model = A2C('MlpPolicy', env, verbose=1, seed=seed_value, tensorboard_log=f"./tensorboard_log/{folder}/a2c")
    else:
        raise ValueError("Unsupported algorithm! Choose between 'PPO' and 'A2C'")

    model.learn(total_timesteps=100_000)
    model.save(f"./tensorboard_log/{folder}/model/{algorithm}_seed_{seed_value}_{growth_type}.model")

    env.save_distances(f'./data/{folder}/distance/distance_history.npy')
    env.save_stiffness_history(f'./data/{folder}/stiffness/stiffness_history.npy')
    env.close()
