import os
import numpy as np
from datetime import datetime
import gym
from gym import utils
import mujoco
import mujoco_env
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import SubprocVecEnv

# Learning rate scheduler
class LearningRateSchedule:
    def __init__(self, schedule_type='constant', lr_start=3e-4, lr_end=1e-5, total_timesteps=1_000_000):
        self.schedule_type = schedule_type
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.total_timesteps = total_timesteps

    def __call__(self, progress_remaining):
        if self.schedule_type == 'linear':
            return self.lr_end + (self.lr_start - self.lr_end) * progress_remaining
        elif self.schedule_type == 'constant':
            return self.lr_start
        else:
            raise ValueError(f"Unsupported schedule type: {self.schedule_type}")

# Training configuration
class TrainingConfig:
    def __init__(self,
                 algorithm='PPO',
                 growth_type='linear',
                 growth_factor=3.0,
                 stiffness_start=5000,
                 stiffness_end=50000,
                 num_seeds=10,
                 total_timesteps=1_000_000,
                 lr_schedule_type='linear',
                 lr_start=5e-4,
                 lr_end=1e-5,
                 n_envs=1):
        self.algorithm = algorithm
        self.growth_type = growth_type
        self.growth_factor = growth_factor
        self.stiffness_start = stiffness_start
        self.stiffness_end = stiffness_end
        self.num_seeds = num_seeds
        self.total_timesteps = total_timesteps
        self.lr_schedule_type = lr_schedule_type
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.n_envs = n_envs
        self.run_date = datetime.now().strftime('%b%d')


    def format_k(self, x):
        return f"{int(x/1000)}k" if x % 1000 == 0 else str(x)

    def folder_name(self):
        date_tag = self.run_date 
        if self.lr_schedule_type == 'constant':
            lr_tag = f"{self.lr_schedule_type}_{self.lr_start:.0e}"
        else:
            lr_tag = f"{self.lr_schedule_type}_{self.lr_start:.0e}_to_{self.lr_end:.0e}"
        seeds_tag = f"seeds_{100}-{100 + self.num_seeds - 1}"
        safe_growth_type = self.growth_type.replace(':', '_')
        return f"LegEnv_{date_tag}_{safe_growth_type}_{lr_tag}_{self.algorithm}_{seeds_tag}"

    def get_lr_schedule(self):
        return LearningRateSchedule(
            schedule_type=self.lr_schedule_type,
            lr_start=self.lr_start,
            lr_end=self.lr_end,
            total_timesteps=self.total_timesteps
        )

# Custom Mujoco environment
class LegEnvBase(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 40}

    def __init__(self, xml_file='leg.xml', render_mode='none', seed=None,
                 stiffness_start=5000, stiffness_end=50000, num_epochs=1000,
                 max_episode_steps=1000, growth_factor=0.03, growth_type='exponential'):
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float32)
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5, render_mode=render_mode, observation_space=self.observation_space)

        if seed is not None:
            self.seed(seed)

        self.stiffness_start = stiffness_start
        self.stiffness_end = stiffness_end
        self.num_epochs = num_epochs
        self.max_episode_steps = max_episode_steps
        self.growth_factor = growth_factor
        self.growth_type = growth_type

        self.epoch_counter = 0
        self.steps_from_reset = 0
        self.global_step = 0

        self.stiffness_scaling = self.stiffness_start
        self.stiffness_history = []
        self.reward_episode = 0
        self.rewards = []
        self.displacements = []
        self.x_start = 0

        # Additional data: tendon forces & kinematics
        self.actuator_force_history = []
        self.actuator_forces_episode = []
        self.qpos_episode = []
        self.qvel_episode = []
        self.qpos_history = []
        self.qvel_history = []

        self.tendon_forces_history = [] 
        self.tendon_forces_episode = []

        self.tendon_length_history = [] 
        self.tendon_lengths_episode = []


        self.update_stiffness(self.epoch_counter)

    def update_stiffness(self, epoch):
        progress = epoch / max(1, self.num_epochs)
        if self.growth_type == 'exponential':
            exponent = self.growth_factor * progress
            self.stiffness_scaling = self.stiffness_start + (
                (self.stiffness_end - self.stiffness_start) * (np.exp(exponent) - 1) / (np.exp(self.growth_factor) - 1))
        elif self.growth_type == 'logarithmic':
            adjusted = 1 - np.exp(-self.growth_factor * progress)
            self.stiffness_scaling = self.stiffness_start + (self.stiffness_end - self.stiffness_start) * adjusted
        elif self.growth_type == 'linear':
            self.stiffness_scaling = self.stiffness_start + progress * (self.stiffness_end - self.stiffness_start)
        elif self.growth_type.startswith('constant_'):
            self.stiffness_scaling = float(self.growth_type.split('_')[1].replace('k', '000'))
            #self.stiffness_scaling = 500
        elif self.growth_type == 'curriculum_linear':
            if progress <= 0.25:
                self.stiffness_scaling = self.stiffness_start
            else:
                lin_progress = (progress - 0.25) / 0.75
                self.stiffness_scaling = self.stiffness_start + lin_progress * (self.stiffness_end - self.stiffness_start)
        self.stiffness_scaling = min(self.stiffness_scaling, self.stiffness_end)
        self.apply_stiffness(self.stiffness_scaling)
        self.stiffness_history.append(self.stiffness_scaling)

    def apply_stiffness(self, value):
        for i in range(len(self.model.tendon_stiffness)):
            self.model.tendon_stiffness[i] = value

    def step(self, action):
        x_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_after = self.data.qpos[0]
        velocity = (x_after - x_before) / self.dt
        reward = max(velocity, 0) - max(-velocity, 0)

        self.reward_episode += reward

        # Save tendon forces and kinematics
        
        # First-time setup: find tendonforce sensor indices
# First-time setup: get indices for all tendonforce sensors

        self.qpos_episode.append(self.data.qpos.copy())
        self.qvel_episode.append(self.data.qvel.copy())
        self.tendon_lengths_episode.append(self.data.ten_length.copy())

        self.steps_from_reset += 1
        self.global_step += 1
        done = self.steps_from_reset >= self.max_episode_steps    
        if done:
            self.rewards.append(self.reward_episode)
            self.displacements.append(x_after - self.x_start)
            self.reward_episode = 0
            # Save episode-level data
            self.actuator_force_history.append(self.actuator_forces_episode)
            self.actuator_forces_episode = []

            self.tendon_forces_history.append(self.tendon_forces_episode)
            self.tendon_forces_episode = []

            self.qpos_history.append(self.qpos_episode)
            self.qvel_history.append(self.qvel_episode)
            self.qpos_episode = []
            self.qvel_episode = []

            self.tendon_length_history.append(self.tendon_lengths_episode)
            self.tendon_lengths_episode = []
    
        return self.get_obs(), reward, done, False, {}

    def reset_model(self):
        self.steps_from_reset = 0
        self.epoch_counter += 1
        self.update_stiffness(self.epoch_counter)
        self.reward_episode = 0
        self.x_start = self.data.qpos[0]

        # Clear episode data
        self.actuator_forces_episode = []
        self.qpos_episode = []
        self.qvel_episode = []
        return self.get_obs()

    def get_obs(self):
        return np.concatenate((self.data.qpos.flat.copy(), self.data.qvel.flat.copy())).ravel()

    def get_position(self):
        return self.data.qpos[0]

    def seed(self, seed=None):
        np.random.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def save_reward_and_displacement(self, folder, seed):
        base = f'./data/{folder}/distance'
        os.makedirs(base, exist_ok=True)
        np.save(f'{base}/reward_history_seed_{seed}.npy', np.array(self.rewards))
        np.save(f'{base}/displacement_history_seed_{seed}.npy', np.array(self.displacements))

    def save_stiffness_history(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.save(filename, np.array(self.stiffness_history))

    # Save tendon forces & kinematics data
    def save_actuator_forces(self, folder, seed):
        base = f'./data/{folder}/actuator_forces'
        os.makedirs(base, exist_ok=True)
        np.save(f'{base}/actuator_forces_seed_{seed}.npy', np.array(self.actuator_force_history, dtype=object))

    def save_qpos_qvel(self, folder, seed):
        base = f'./data/{folder}/kinematics'
        os.makedirs(base, exist_ok=True)
        np.save(f'{base}/qpos_seed_{seed}.npy', np.array(self.qpos_history, dtype=object))
        np.save(f'{base}/qvel_seed_{seed}.npy', np.array(self.qvel_history, dtype=object))

    def save_tendon_lengths(self, folder, seed):
        base = f'./data/{folder}/tendon_lengths'
        os.makedirs(base, exist_ok=True)
        np.save(f'{base}/tendon_lengths_seed_{seed}.npy', np.array(self.tendon_length_history, dtype=object))
    
    def save_tendon_forces(self, folder, seed):
        base = f'./data/{folder}/tendon_forces'
        os.makedirs(base, exist_ok=True)
        np.save(f'{base}/tendon_forces_seed_{seed}.npy', np.array(self.tendon_forces_history, dtype=object))

# Model evaluation
def evaluate_model(model, env, num_episodes=10):
    distances = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        x_start = env.get_position()
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
        x_end = env.get_position()
        distances.append(x_end - x_start)
    return np.mean(distances)

# Aggregate results for multiple seeds
def aggregate_and_save_results(config: TrainingConfig):
    folder = config.folder_name()
    rewards, displacements = [], []
    for i in range(config.num_seeds):
        r_path = f'./data/{folder}/distance/reward_history_seed_{100 + i}.npy'
        d_path = f'./data/{folder}/distance/displacement_history_seed_{100 + i}.npy'
        if os.path.exists(r_path):
            rewards.append(np.load(r_path))
        if os.path.exists(d_path):
            displacements.append(np.load(d_path))

    save_dir = f'./data/aggregated_results/{folder}/'
    os.makedirs(save_dir, exist_ok=True)

    # Use growth_type in file names
    growth_tag = config.growth_type.replace(':', '_')

    if rewards:
        min_len = min(len(r) for r in rewards)
        rewards_trimmed = np.array([r[:min_len] for r in rewards])
        reward_mean = np.mean(rewards_trimmed, axis=0)
        reward_se = np.std(rewards_trimmed, axis=0, ddof=1) / np.sqrt(len(rewards))
        np.save(os.path.join(save_dir, f'reward_mean_{growth_tag}.npy'), reward_mean)
        np.save(os.path.join(save_dir, f'reward_se_{growth_tag}.npy'), reward_se)
        print(f"Saved aggregated reward mean and SE at {save_dir} with growth type '{growth_tag}'")

    if displacements:
        min_len = min(len(d) for d in displacements)
        displacements_trimmed = np.array([d[:min_len] for d in displacements])
        displacement_mean = np.mean(displacements_trimmed, axis=0)
        displacement_se = np.std(displacements_trimmed, axis=0, ddof=1) / np.sqrt(len(displacements))
        np.save(os.path.join(save_dir, f'displacement_mean_{growth_tag}.npy'), displacement_mean)
        np.save(os.path.join(save_dir, f'displacement_se_{growth_tag}.npy'), displacement_se)
        print(f"Saved aggregated displacement mean and SE at {save_dir} with growth type '{growth_tag}'")


# Factory function for parallel environment creation
def make_env(seed_offset, config, seed_value):
    def _init():
        env = LegEnvBase(render_mode=None,
                         growth_factor=config.growth_factor,
                         growth_type=config.growth_type,
                         stiffness_start=config.stiffness_start,
                         stiffness_end=config.stiffness_end)
        env.seed(seed_value + seed_offset)
        return env
    return _init

# Training with parallel environments
def train_env(seed_value, config: TrainingConfig):
    folder = config.folder_name()
    os.makedirs(f"./tensorboard_log/{folder}/model/", exist_ok=True)
    os.makedirs(f"./data/{folder}/distance/", exist_ok=True)
    os.makedirs(f"./data/{folder}/stiffness/", exist_ok=True)

    print("1")
    env = LegEnvBase(render_mode=None,
                     growth_factor=config.growth_factor,
                     growth_type=config.growth_type,
                     stiffness_start=config.stiffness_start,
                     stiffness_end=config.stiffness_end)
    env.seed(seed_value)

    lr_schedule = config.get_lr_schedule()

    if config.algorithm == 'PPO':
        model = PPO('MlpPolicy', env, verbose=1, seed=seed_value,
                    tensorboard_log=f"./tensorboard_log/{folder}/ppo",
<<<<<<< HEAD
                    learning_rate=lr_schedule)
=======
                    learning_rate=lr_schedule, n_epochs = 3,
                    n_steps=2048 // config.n_envs)  # Ensure n_steps is divisible by n_envs
>>>>>>> cfaefa7622221fc77afbf97fecb612bfcca14817
    elif config.algorithm == 'A2C':
        model = A2C('MlpPolicy', env, verbose=1, seed=seed_value,
                    tensorboard_log=f"./tensorboard_log/{folder}/a2c",
                    learning_rate=lr_schedule)
    else:
        raise ValueError("Unsupported algorithm! Choose between 'PPO' and 'A2C'")

    model.learn(total_timesteps=config.total_timesteps)
    final_model_path = f"./data/{folder}/final_model_seed_{seed_value}.zip"
    model.save(final_model_path)
    print(f"Model saved at: {final_model_path}")
    
    print("4")
    # Save logs and data
    env.save_reward_and_displacement(folder, seed_value)
    env.save_stiffness_history(f'./data/{folder}/stiffness/stiffness_history.npy')
    env.save_actuator_forces(folder, seed_value)
    env.save_qpos_qvel(folder, seed_value)
    env.save_tendon_lengths(folder, seed_value)
    env.save_tendon_forces( folder,seed_value )
    
    
    # Evaluate
    eval_env = make_env(0, config, seed_value)()
    avg_eval_distance = evaluate_model(model, eval_env, num_episodes=10)
    np.save(f'./data/{folder}/distance/eval_distance_seed_{seed_value}.npy', np.array(avg_eval_distance))
    eval_env.close()
    env.close()

    aggregate_and_save_results(config)