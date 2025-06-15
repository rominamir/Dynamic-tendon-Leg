import os
from datetime import datetime
from typing import Optional

import imageio
import gym
import mujoco
import mujoco_env
import numpy as np
from gym import utils
from stable_baselines3 import PPO

"""Constant‑stiffness PPO environment & utilities (Python 3.8+ compatible).
Folder names now embed the exact constant value, e.g. ``constant_10k``.
"""

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
FIXED_GLOBAL_SEED = 404

def set_global_seeds(seed: int) -> None:
    """Ensure deterministic behaviour across NumPy / Python / Torch / CUDA."""
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_global_seeds(FIXED_GLOBAL_SEED)

class ConstantLR:
    """Callable returning a fixed learning rate (for SB3 schedule API)."""

    def __init__(self, lr: float = 3e-4):
        self.lr = lr

    def __call__(self, *_):
        return self.lr


# -----------------------------------------------------------------------------
# Training configuration
# -----------------------------------------------------------------------------

class TrainingConfig:
    """Hyper‑parameter container for constant‑stiffness PPO experiments."""

    def __init__(
        self,
        stiffness_start: int = 5_000,
        stiffness_end: int = 50_000,
        num_seeds: int = 10,
        total_timesteps: int = 1_000_000,
        lr: float = 5e-4,
        max_episode_steps: int = 1_000,
        num_epochs: int = 1_000,
        seed_start: int = 100,
        seed_end: int = 124,
    ) -> None:
        self.algorithm = "PPO"
        self.stiffness_start = stiffness_start
        self.stiffness_end = stiffness_end
        self.num_seeds = num_seeds
        self.total_timesteps = total_timesteps
        self.max_episode_steps = max_episode_steps
        self.num_epochs = num_epochs
        self.lr = lr
        self.run_date = datetime.now().strftime("%b%d")
        self.seed_start = seed_start
        self.seed_end = seed_end
        self.lr_schedule = ConstantLR(lr)
        self.stiffness_tag = f"constant_{int(self.stiffness_start/1000)}k"

    # -------------------- helpers --------------------
	def folder_name(self) -> str:
	    seeds_tag = f"seeds_{self.seed_start}-{self.seed_end}"
	    return f"LegEnv_{self.run_date}_{self.stiffness_tag}_lr_{self.lr:.0e}_PPO_{seeds_tag}"


# -----------------------------------------------------------------------------
# Custom MuJoCo environment (constant stiffness only)
# -----------------------------------------------------------------------------

class LegEnvBase(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 40}

    def __init__(
        self,
        xml_file: str = "leg.xml",
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        stiffness_start: int = 5_000,
        stiffness_end: int = 50_000,
        num_epochs: int = 1_000,
        max_episode_steps: int = 1_000,
    ) -> None:
        # Action / observation spaces
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float32)

        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(
            self, xml_file, 5, render_mode=render_mode, observation_space=self.observation_space
        )

        if seed is not None:
            self.seed(seed)

        self.stiffness_start = stiffness_start
        self.stiffness_end = stiffness_end
        self.num_epochs = num_epochs
        self.max_episode_steps = max_episode_steps

        # Episode bookkeeping
        self.epoch_counter = 0
        self.steps_from_reset = 0
        self.global_step = 0

        # Data logging structures
        self.stiffness_scaling = stiffness_start
        self.stiffness_history: list[float] = []
        self.reward_episode = 0.0
        self.rewards: list[float] = []
        self.displacements: list[float] = []
        self.x_start = 0.0

        # Apply initial constant stiffness
        self._apply_stiffness(self.stiffness_scaling)

    # -------------------- MuJoCo helpers --------------------
    def _apply_stiffness(self, value: float) -> None:
        for i in range(len(self.model.tendon_stiffness)):
            self.model.tendon_stiffness[i] = value
        self.stiffness_history.append(value)

    # -------------------- RL loop --------------------
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
            self.reward_episode = 0.0
            self.steps_from_reset = 0

        return self.get_obs(), reward, done, False, {}

    def reset_model(self):
        self.epoch_counter += 1

        # Hard reset to zero state
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.x_start = self.data.qpos[0]
        return self.get_obs()

    def get_obs(self):
        return np.concatenate((self.data.qpos.flat.copy(), self.data.qvel.flat.copy())).ravel()

    def seed(self, seed: Optional[int] = None):
        np.random.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def viewer_setup(self):
        cam_id = mujoco.mj_name2id(self.model,
                                mujoco.mjtObj.mjOBJ_CAMERA,
                                "follow_leg")
        self.viewer.cam.fixedcamid = cam_id
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        pass 
    def render(self, mode=None, *args, **kwargs):
        if self.viewer is not None:
            chassis_pos = self.data.body('Chassis').xpos
            self.viewer.cam.lookat[:] = chassis_pos
        try:
            return super().render(*args, **kwargs)
        except TypeError as e:
            return super().render()


            


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

def train(config: TrainingConfig, seed: int) -> None:
    """
    Train a single PPO agent with the given random seed, save all artefacts, and
    (optionally) record a short demo video.

    Parameters
    ----------
    config : TrainingConfig
        Experiment-wide hyper-parameters and I/O conventions.
    seed : int
        Random seed used for the SB3 model, Gym spaces and NumPy RNGs.
    """
    # ---------------------------------------------------------------------- #
    # 0) I/O setup                                                           #
    # ---------------------------------------------------------------------- #
    folder    = config.folder_name()             # e.g. LegEnv_Jun14_constant_10k_lr_5e-04_PPO_seeds_100-124
    tag       = config.stiffness_tag            # e.g. constant_10k
    save_path = f"./data/{folder}"
    os.makedirs(save_path, exist_ok=True)

    # ---------------------------------------------------------------------- #
    # 1) Create training environment (headless for speed)                    #
    # ---------------------------------------------------------------------- #
    env = LegEnvBase(
        render_mode=None,                        # headless → no RGB overhead on cluster
        stiffness_start=config.stiffness_start,
        stiffness_end=config.stiffness_end,
        num_epochs=config.num_epochs,
        max_episode_steps=config.max_episode_steps,
    )
    env.seed(seed)

    # ---------------------------------------------------------------------- #
    # 2) Instantiate PPO and start learning                                  #
    # ---------------------------------------------------------------------- #
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.lr,                # constant LR
        seed=seed,
        verbose=1,
        tensorboard_log=f"./tensorboard_logs/{folder}/",
    )

    model.learn(total_timesteps=config.total_timesteps)

    # ---------------------------------------------------------------------- #
    # 3) Persist artefacts                                                   #
    # ---------------------------------------------------------------------- #
    model.save(f"{save_path}/model_{tag}_seed_{seed}")  # .zip added by SB3

    # episode-level histories
    np.save(f"{save_path}/rewards_{tag}_seed_{seed}.npy",       env.rewards)
    np.save(f"{save_path}/displacements_{tag}_seed_{seed}.npy", env.displacements)
    np.save(f"{save_path}/stiffness_{tag}_seed_{seed}.npy",     env.stiffness_history)

    # ---------------------------------------------------------------------- #
    # 4) Record a short demo video (optional)                                #
    #    Gracefully skip if GL/EGL unavailable on the node.                  #
    # ---------------------------------------------------------------------- #
    try:
        video_file = f"{save_path}/final_episode_{tag}_seed_{seed}.mp4"
        _record_final_episode_video(config, model, seed, video_file)
    except Exception as exc:
        print(f"[warn] video capture failed for seed {seed}: {exc}")

    env.close()



def _record_final_episode_video(
    config: TrainingConfig, model, seed: int, outfile: str, max_frames: int = 1_000
) -> None:
    """Run one episode in an RGB-array env and save it as an MP4."""
    env = LegEnvBase(
        render_mode="rgb_array",
        stiffness_start=config.stiffness_start,
        stiffness_end=config.stiffness_end,
        num_epochs=1,
        max_episode_steps=config.max_episode_steps,
    )
    env.seed(seed)
    _ = env.render()
    env.viewer_setup()

    obs, _ = env.reset()
    frames = []
    done = False
    steps = 0
    while not done and steps < max_frames:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        # cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "follow_leg")
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        steps += 1

    env.close()

    if not frames:
        raise RuntimeError("env.render() returned no frames — cannot save video")

    imageio.mimsave(outfile, frames, fps=60)
    print(f"[video] saved → {outfile}")


def aggregate_and_save_results(config: TrainingConfig) -> None:
    """Aggregate reward/displacement across seeds and save mean & SE."""
    folder = config.folder_name()
    tag = config.stiffness_tag
    save_dir = f'./data/aggregated_results/{folder}'
    os.makedirs(save_dir, exist_ok=True)

    reward_list, disp_list = [], []

    for seed in range(config.seed_start, config.seed_end + 1):
        r_path = f'./data/{folder}/rewards_{tag}_seed_{seed}.npy'
        d_path = f'./data/{folder}/displacements_{tag}_seed_{seed}.npy'
        if os.path.isfile(r_path):
            reward_list.append(np.load(r_path))
        if os.path.isfile(d_path):
            disp_list.append(np.load(d_path))

    # Reward
    if reward_list:
        min_len = min(len(r) for r in reward_list)
        rewards = np.stack([r[:min_len] for r in reward_list])
        np.save(f"{save_dir}/reward_mean_{tag}.npy", rewards.mean(axis=0))
        np.save(f"{save_dir}/reward_se_{tag}.npy",
                rewards.std(axis=0, ddof=1) / np.sqrt(rewards.shape[0]))
        print(f"✅ Aggregated rewards saved → {save_dir}")

    # Displacement
    if disp_list:
        min_len = min(len(d) for d in disp_list)
        disps = np.stack([d[:min_len] for d in disp_list])
        np.save(f"{save_dir}/displacement_mean_{tag}.npy", disps.mean(axis=0))
        np.save(f"{save_dir}/displacement_se_{tag}.npy",
                disps.std(axis=0, ddof=1) / np.sqrt(disps.shape[0]))
        print(f"✅ Aggregated displacements saved → {save_dir}")