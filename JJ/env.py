import os
import numpy as np
from datetime import datetime
import gym
from gym import utils
import mujoco
import mujoco_env
from stable_baselines3 import PPO, A2C

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

def aggregate_eval_distances(folder, num_seeds=10):
    eval_distances = []
    for i in range(num_seeds):
        path = f'./data/{folder}/distance/eval_distance_seed_{100 + i}.npy'
        if os.path.exists(path):
            eval_distances.append(np.load(path))
    if len(eval_distances) >= 2:
        overall_avg = np.mean(eval_distances)
        np.save(f'./data/{folder}/distance/eval_distance_ALL.npy', overall_avg)
        print(f"✅ Updated overall average eval distance over {len(eval_distances)} seeds: {overall_avg:.4f}")
    else:
        print("⚠️ Not enough evaluation files to compute overall average.")

def aggregate_training_rewards(folder, num_seeds=10):
    rewards, displacements = [], []
    for i in range(num_seeds):
        r_path = f'./data/{folder}/distance/reward_history_seed_{100 + i}.npy'
        d_path = f'./data/{folder}/distance/displacement_history_seed_{100 + i}.npy'
        if os.path.exists(r_path):
            rewards.append(np.load(r_path))
        if os.path.exists(d_path):
            displacements.append(np.load(d_path))
    if rewards:
        min_len = min(len(r) for r in rewards)
        reward_mean = np.mean([r[:min_len] for r in rewards], axis=0)
        np.save(f'./data/{folder}/distance/reward_history_ALL.npy', reward_mean)
        print(f"✅ Saved avg reward curve over {len(rewards)} seeds.")
    if displacements:
        min_len = min(len(d) for d in displacements)
        disp_mean = np.mean([d[:min_len] for d in displacements], axis=0)
        np.save(f'./data/{folder}/distance/displacement_history_ALL.npy', disp_mean)
        print(f"✅ Saved avg displacement curve over {len(displacements)} seeds.")

class LegEnvBase(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 40}

    def __init__(self, xml_file='leg.xml', render_mode='none', seed=None,
                 stiffness_start=20000, stiffness_end=30000, num_epochs=1000,
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
        elif self.growth_type.startswith('constant:'):
            self.stiffness_scaling = float(self.growth_type.split(':')[1].replace('k', '000'))
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
        self.steps_from_reset += 1
        self.global_step += 1
        done = self.steps_from_reset >= self.max_episode_steps
        if done:
            self.rewards.append(self.reward_episode)
            self.displacements.append(x_after - self.x_start)
            self.reward_episode = 0
        return self.get_obs(), reward, done, False, {}

    def reset_model(self):
        self.steps_from_reset = 0
        self.epoch_counter += 1
        self.update_stiffness(self.epoch_counter)
        self.reward_episode = 0
        self.x_start = self.data.qpos[0]
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

def train_env(seed_value, algorithm='PPO', growth_factor=0.03, growth_type='exponential', num_seeds=10,
              stiffness_start=20000, stiffness_end=30000):
    # Format stiffness range for folder naming, e.g., 20k-30k
    def format_k(x):
        return f"{int(x/1000)}k" if x % 1000 == 0 else str(x)
    stiffness_tag = f"{format_k(stiffness_start)}-{format_k(stiffness_end)}"

    folder = f'LegEnv_{datetime.now().strftime("%b%d")}_{growth_type}_{stiffness_tag}_{algorithm}'
    os.makedirs(f"./tensorboard_log/{folder}/model/", exist_ok=True)
    os.makedirs(f"./data/{folder}/distance/", exist_ok=True)
    os.makedirs(f"./data/{folder}/stiffness/", exist_ok=True)

    env = LegEnvBase(render_mode=None, growth_factor=growth_factor, growth_type=growth_type,
                     stiffness_start=stiffness_start, stiffness_end=stiffness_end)
    env.seed(seed_value)

    if algorithm == 'PPO':
        model = PPO('MlpPolicy', env, verbose=1, seed=seed_value, tensorboard_log=f"./tensorboard_log/{folder}/ppo")
    elif algorithm == 'A2C':
        model = A2C('MlpPolicy', env, verbose=1, seed=seed_value, tensorboard_log=f"./tensorboard_log/{folder}/a2c")
    else:
        raise ValueError("Unsupported algorithm! Choose between 'PPO' and 'A2C'")

    model.learn(total_timesteps=1000000)
    model.save(f"./tensorboard_log/{folder}/model/{algorithm}_seed_{seed_value}_{growth_type}.model")

    env.save_reward_and_displacement(folder, seed_value)
    env.save_stiffness_history(f'./data/{folder}/stiffness/stiffness_history.npy')
    avg_eval_distance = evaluate_model(model, env, num_episodes=10)
    np.save(f'./data/{folder}/distance/eval_distance_seed_{seed_value}.npy', np.array(avg_eval_distance))
    env.close()

    aggregate_eval_distances(folder, num_seeds=num_seeds)
    aggregate_training_rewards(folder, num_seeds=num_seeds)

def run_training_with_multiple_seeds():
    growth_type = "constant:5000"
    growth_factor = 3.0
    algorithm = "PPO"
    num_seeds = 10
    stiffness_start = 20000
    stiffness_end = 30000

    folder = f'LegEnv_{datetime.now().strftime("%b%d")}_{growth_type}_{algorithm}"
    print(f"\n🚀 Starting {num_seeds} training runs with {growth_type} using {algorithm}")
    for i in range(num_seeds):
        train_env(100 + i, algorithm, growth_factor, growth_type, num_seeds,
                  stiffness_start=stiffness_start, stiffness_end=stiffness_end)

if __name__ == '__main__':
    run_training_with_multiple_seeds()
