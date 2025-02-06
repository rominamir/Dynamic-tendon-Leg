import numpy as np
from stable_baselines3 import PPO
from log import LogLegEnv  # Import logarithmic environment
import multiprocessing as mp
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import RecordVideo
from datetime import datetime
import os

# Get the current date
current_date = datetime.now().strftime("%b%d")  # Format as "Jan14"
folder = 'Logarithmic_10k_jan14'

# Ensure necessary directories exist
os.makedirs(f"./tensorboard_log/{folder}/model/", exist_ok=True)
os.makedirs(f"./data/{folder}/distance/", exist_ok=True)
os.makedirs(f"./data/{folder}/stiffness/", exist_ok=True)
os.makedirs(f"./videos/", exist_ok=True)


class CustomCallback(BaseCallback):
    def _on_step(self) -> bool:
        pass

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_rollout_start(self) -> None:
        self.model.save(f"PPO.model")


def linear_schedule(initial_value):
    """Dynamically adjusts learning rate during training."""
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func


def evaluate_model(seed_value, growth_factor=0.01):
    """Evaluates the trained model and records video."""
    print(f"Starting evaluation with seed {seed_value} and growth_factor {growth_factor}")

    env = LogLegEnv(render_mode="rgb_array", growth_factor=growth_factor)  # Adjustable growth factor

    env = RecordVideo(
        env,
        video_folder=f"./videos/seed_{seed_value}/",
        episode_trigger=lambda episode_id: True,  # Record every episode
    )

    env.seed(seed_value)

    model_path = f"./tensorboard_log/{folder}/model/PPO_seed_{seed_value}.model"
    model = PPO.load(model_path, env)

    obs, _ = env.reset()
    total_reward = 0
    num_episodes = 0

    for _ in range(10_000):  # Adjust as needed
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        if done:
            obs, _ = env.reset()
            num_episodes += 1

    avg_reward = total_reward / max(1, num_episodes)
    print(f"Average reward for seed {seed_value}: {avg_reward}")

    env.close()


def train_env(seed_value, train=True, growth_factor=0.01):
    """Trains the PPO model in LogLegEnv and records stiffness/distance history."""
    if train:
        print(f"Starting training with seed {seed_value} and growth_factor {growth_factor}")
        env = LogLegEnv(render_mode=None, growth_factor=growth_factor)  # Adjustable growth factor
        env.seed(seed_value)

        model = PPO(
            'MlpPolicy', env, verbose=1, seed=seed_value,
            tensorboard_log=f"./tensorboard_log/{folder}/ppo_seed{seed_value}",
            gamma=0.9,
            ent_coef=0.01,  # Match ent_coef to Exponential environment
            learning_rate=linear_schedule(0.0003)
        )

        model.learn(total_timesteps=100_000)

        # Save trained model
        model.save(f"./tensorboard_log/{folder}/model/PPO_seed_{seed_value}.model")

        # Save distance history
        env.save_distances(f'./data/{folder}/distance/distance_history_seed_{seed_value}.npy')

        # ✅ Save stiffness history for all seeds
        env.save_stiffness_history(f'./data/{folder}/stiffness/stiffness_history_seed_{seed_value}.npy')

        env.close()
    else:
        evaluate_model(seed_value, growth_factor)


def main_parallel(num_runs, train=True, growth_factor=0.01):
    """Runs multiple training/evaluation processes in parallel."""
    seeds = list(range(1, num_runs + 1))  # Generate seeds from 1 up to num_runs
    with mp.Pool(processes=num_runs) as pool:
        pool.starmap(train_env, [(seed, train, growth_factor) for seed in seeds])


if __name__ == '__main__':
    print("This is a Logarithmic function run")

    # Uncomment below lines to enable parallel execution
    # num_runs = int(input("Enter the number of parallel runs: "))
    # train = input("Enter 'train' or 'eval' for mode: ") == 'train'
    # growth_factor = float(input("Enter growth factor for stiffness adjustment: "))
    # main_parallel(num_runs, train=train, growth_factor=growth_factor)

    train_env(101, True, growth_factor=0.02)  # Run a single training instance with adjustable growth
