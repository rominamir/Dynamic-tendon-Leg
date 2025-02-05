import numpy as np
from stable_baselines3 import PPO
import imageio  # for writing frames to mp4
import os
from Basde_Code_2 import ConLegEnv_enhanced as ConLegEnv    #as DynamicLegEnv
#from Basde_Code import LegEnv as ConLegEnv
import multiprocessing as mp
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import RecordVideo
import numpy as np
from datetime import datetime

# Get the current date
current_date = datetime.now()
# Format the date as "Dec9"
current_date = current_date.strftime("%b%d")
folder = 'V2_Constant_5k_jan16_cons_enhanced' 

class VideoRecordingCallback(BaseCallback):
    """
    Saves a video of every Nth episode during training.
    """
    def __init__(self, record_freq=10, verbose=1, video_dir="./training_videos"):
        super().__init__(verbose)
        self.record_freq = record_freq
        self.video_dir = video_dir
        self.frames = []
        self.episode_count = 0
        os.makedirs(self.video_dir, exist_ok=True)

    def _on_training_start(self) -> None:
        if self.verbose > 0:
            print("Training started; video recording callback is active.")

    def _on_step(self) -> bool:
        # Render the current frame as an RGB array
        frame = self.training_env.render(mode="rgb_array")
        self.frames.append(frame)

        # If done=True for any environment in the vectorized env
        for i, done in enumerate(self.locals["dones"]):
            if done:
                self.episode_count += 1
                # If it's a 'record' episode
                if self.episode_count % self.record_freq == 0:
                    filename = os.path.join(self.video_dir, f"training_episode_{self.episode_count}.mp4")
                    # Save all frames from this episode
                    with imageio.get_writer(filename, fps=30) as writer:
                        for f in self.frames:
                            writer.append_data(f)
                    if self.verbose > 0:
                        print(f"[Callback] Saved video: {filename}")
                # Reset frames for the next episode
                self.frames = []

        # Return True so training continues
        return True

    def _on_training_end(self) -> None:
        if self.verbose > 0:
            print("Training ended; video recording callback is done.")

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
    env = ConLegEnv(render_mode="rgb_array")  # Use 'rgb_array' for video recording
    env = RecordVideo(
        env,
        video_folder=f"./videos/seed_{seed_value}/",  # Folder to save videos
        episode_trigger=lambda episode_id: True,  # Record every episode
    )
    env.seed(seed_value)

    model = PPO.load(f"./tensorboard_log/model/{folder}/PPO_seed_{seed_value}.model", env)
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
        env = ConLegEnv(render_mode=None)
        env.seed(seed_value)
        
        model = PPO(
            'MlpPolicy', env, verbose=1, seed=seed_value,
            tensorboard_log=f"./tensorboard_log/{folder}/ppo_seed{seed_value}",
            gamma=0.9, 
           
            learning_rate=0.0003) 
        
        model.learn(total_timesteps=300_000)

        model.save(f"./tensorboard_log/model//{folder}/PPO_seed_{seed_value}.model")
        env.save_distances(f'./data//{folder}/distance/distance_history_seed_{seed_value}.npy')
        #if seed_value == 1:
        #  env.save_stiffness_history(f'./data/{folder}/stiffness/stiffness_history_seed_{seed_value}.npy')
        #env.save_stiffness_history(f'./data//{folder}/stiffness/stiffness_history_seed_{seed_value}_test.npy')
        env.close()
    else:
        evaluate_model(seed_value)

def main_parallel(num_runs, train=True):
    seeds = list(range(1, num_runs + 1))  # Generate seeds from 1 up to num_runs
    with mp.Pool(processes=num_runs) as pool:
        pool.starmap(train_env, [(seed, train) for seed in seeds])

def train_env_(seed_value, train=True):
    if train:
        print(f"Starting training with seed {seed_value}")

        # 1) Create the environment with rgb_array rendering
        env = ConLegEnv(render_mode="rgb_array")
        env.seed(seed_value)

        # 2) Create the PPO model
        model = PPO(
            'MlpPolicy',
            env,
            verbose=1,
            seed=seed_value,
            tensorboard_log=f"./tensorboard_log/{folder}/ppo_seed{seed_value}",
            gamma=0.9,
            learning_rate=linear_schedule(0.0003)
        )

        # 3) Create our custom video-recording callback
        video_callback = VideoRecordingCallback(record_freq=10, verbose=1, video_dir=f"./training_videos/{folder}")

        # 4) Train the model with the callback
        model.learn(total_timesteps=100_000, callback=video_callback)

        # 5) Save the model and any other data
        model.save(f"./tensorboard_log/model/{folder}/PPO_seed_{seed_value}.model")
        env.save_distances(f'./data/{folder}/distance/distance_history_seed_{seed_value}.npy')
        #if seed_value == 101:
        #    env.save_stiffness_history(f'./data/{folder}/stiffness/stiffness_history_seed_{seed_value}.npy')

        env.close()

    else:
        evaluate_model(seed_value)
if __name__ == '__main__':   

    print(f"this is an {folder} functin run")
    #num_runs = int(input("Enter the number of parallel runs: "))
    #train = input("Enter 'train' or 'eval' for mode: ")
    #main_parallel(num_runs, train=train)
    train_env(1, True)
    #train_env_(101, True)