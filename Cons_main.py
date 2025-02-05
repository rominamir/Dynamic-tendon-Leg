import numpy as np
from stable_baselines3 import PPO
from Basde_Code_2 import ConLegEnv as LegEnv
import multiprocessing as mp
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import TimeLimit
from gymnasium.wrappers import RecordVideo

from datetime import datetime

# Get the current date
current_date = datetime.now()
# Format the date as "Dec9"
current_date = current_date.strftime("%b%d")
stiffness_value = 5000
file_name = 'Constant_test'
class CustomCallback(BaseCallback):
    def _on_step(self) -> bool:
        pass

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_rollout_start(self) -> None:
        self.model.save(f"PPO.model")

def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func

def evaluate_model(seed_value):
    print(f"Starting evaluation with seed {seed_value}")
    env = LegEnv(render_mode="rgb_array", stiffness_value=stiffness_value,)  # Use 'rgb_array' for video recording
    env = RecordVideo(
        env,
        video_folder=f"./videos/seed_{seed_value}/",  # Folder to save videos
        episode_trigger=lambda episode_id: True,  # Record every episode
    )
    env.seed(seed_value)
    
    model = PPO.load(f"./tensorboard_log/model/{file_name}/PPO_seed_{seed_value}.model", env)
    obs, _ = env.reset()  # Adjust for Gym API
    total_reward = 0
    num_episodes = 0
    for _ in range(10_000):  # Adjust as needed
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        if done:
            obs, _ = env.reset()  # Adjust for Gym API
            num_episodes += 1
    print(f"Average reward for seed {seed_value}: {total_reward / max(1, num_episodes)}")
    env.close()


def train_env(seed_value, train=True, fps=30):
    if train:
        print(f"Starting training with seed {seed_value}")
        env =  LegEnv(render_mode='rgb_array')
        video_folder = f"./videos/training_seed_{seed_value}"
        env = RecordVideo(
            env,
            video_folder=video_folder,            # Folder to save videos
            episode_trigger=lambda x: x % 50 == 0,  # Record every 50th episode
            name_prefix="training_video",
        )
        env.seed(seed_value)
        
        model = PPO(
            'MlpPolicy', env, verbose=1, seed=seed_value, 
            tensorboard_log=f"./tensorboard_log/{file_name}/ppo_seed{seed_value}",
            gamma=0.9, learning_rate=linear_schedule(0.0003)
        )
        
        model.learn(total_timesteps=100_000,)

        model.save(f"./tensorboard_log/model/{file_name}/PPO_seed_{seed_value}.model")
        env.save_distances(f'./data/{file_name}/distance/distance_history_seed_{seed_value}.npy')
        if seed_value == 100:
            env.save_stiffness_history(f'./data/{file_name}/stiffness/stiffness_history_seed_{seed_value}.npy')

        env.close()
    else:
        evaluate_model(seed_value)

def main_parallel(num_runs, train=True):
    seeds = list(range(1, num_runs + 1))  # Generate seeds from 1 up to num_runs
    with mp.Pool(processes=num_runs) as pool:
        pool.starmap(train_env, [(seed, train) for seed in seeds])


if __name__ == '__main__':
    #num_runs = int(input("Enter the number of parallel runs: "))
    #train = input("Enter 'train' or 'eval' for mode: ")
    #main_parallel(num_runs, train=train)
   train_env(100, True)
   #evaluate_model(100)