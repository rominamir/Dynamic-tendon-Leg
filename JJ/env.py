import os
from datetime import datetime
from typing import Optional
from typing import List

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
    """
    Hyper-parameter container for constant-stiffness PPO experiments.
    Now supports fixed run_date for consistent folder naming.
    """
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
        run_date: str = None,   # New: allow passing a fixed run_date
        growth_type: str = "constant_unknown",  # ✅ Add this
        folder_seed_start: int = None,
        folder_seed_end: int = None
        ) -> None:
        self.growth_type = growth_type  # ✅ Save it for folder naming
        self.algorithm = "PPO"
        self.growth_type = growth_type

        self.stiffness_start = stiffness_start
        self.stiffness_end = stiffness_end
        self.num_seeds = num_seeds
        self.total_timesteps = total_timesteps
        self.max_episode_steps = max_episode_steps
        self.num_epochs = num_epochs
        self.lr = lr

        # Use passed run_date if provided, otherwise use current date
        if run_date is None:
            self.run_date = datetime.now().strftime("%b%d")
        else:
            self.run_date = run_date
        self.seed_start = seed_start
        self.seed_end = seed_end
        self.lr_schedule = ConstantLR(lr)
        self.stiffness_tag = f"constant_{int(self.stiffness_start/1000)}k"

        self.folder_seed_start = folder_seed_start if folder_seed_start is not None else seed_start
        self.folder_seed_end = folder_seed_end if folder_seed_end is not None else seed_end

    def folder_name(self) -> str:
        seeds_tag = f"seeds_{self.folder_seed_start}-{self.folder_seed_end}"

        lr_tag = f"{self.lr:.0e}".replace("e-0", "e-").replace("e-", "e-0")  # Formats like '5e-04'
        
        return f"LegEnv_{self.run_date}_{self.growth_type}_lr_{lr_tag}_PPO"


# -----------------------------------------------------------------------------
# Custom MuJoCo environment (constant stiffness only)
# -----------------------------------------------------------------------------

class LegEnvBase(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 40} #40

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

        # Additional episode-level data
        self.qpos_episode = []
        self.qvel_episode = []
        self.actuator_forces_episode = []
        self.tendon_lengths_episode = []

        self.qpos_history = []
        self.qvel_history = []
        
        self.tendon_force_history = []  # New: stores per-step tendon force info
        self.tendon_force_episode = []  # New: stores tendon forces for current episode

        self.actuator_force_history = []
        self.tendon_lengths_history = []

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
        # print(f"[Action] {action}")
        
        self.do_simulation(action, self.frame_skip)
        x_after = self.data.qpos[0]
        velocity = (x_after - x_before) / self.dt
        reward = max(velocity, 0) - max(-velocity, 0)

        self.reward_episode += reward

        # Log per-step data
        
        self.qpos_episode.append(self.data.qpos.copy())
        self.qvel_episode.append(self.data.qvel.copy())
        self.tendon_lengths_episode.append(self.data.ten_length.copy())
        self.actuator_forces_episode.append(self.data.actuator_force.copy())

        rest_lengths = [2.048585725311554, 1.2340794371759454, 1.7340794371759456]
        self._log_tendon_forces(rest_lengths)

        

        self.steps_from_reset += 1
        self.global_step += 1



        done = self.steps_from_reset >= self.max_episode_steps
        if done:
            
            # print(f"[DEBUG] Episode finished — reward = {self.reward_episode}, steps = {self.steps_from_reset}")

            self.rewards.append(self.reward_episode)

            # Save episode-level data
            self.qpos_history.append(self.qpos_episode)
            self.qvel_history.append(self.qvel_episode)
            self.tendon_lengths_history.append(self.tendon_lengths_episode)
            self.actuator_force_history.append(self.actuator_forces_episode)
            self.tendon_force_history.append(self.tendon_force_episode)
            

            # Reset per-episode storage
            self.qpos_episode = []
            self.qvel_episode = []
            self.tendon_lengths_episode = []
            self.actuator_forces_episode = []
            self.tendon_force_episode = []

            self.displacements.append(x_after - self.x_start)
            self.reward_episode = 0.0
            self.steps_from_reset = 0

        return self.get_obs(), reward, done, False, {}

    def reset_model(self):
        self.epoch_counter += 1

        # # Hard reset to zero state
        # self.data.qpos[:] = 0.0
        # self.data.qvel[:] = 0.0
        # mujoco.mj_forward(self.model, self.data)

        self.x_start = self.data.qpos[0]
        return self.get_obs()

    def get_obs(self):
        return np.concatenate((self.data.qpos.flat.copy(), self.data.qvel.flat.copy())).ravel()

    
    
    
    def _log_tendon_forces(self, rest_lengths: List[float]):
        """Compute and store active/passive forces dynamically from model values."""
        forces_this_step = []

        # for i in range(3):
        #     tendon_id = self.tendon_ids[i]
        #     actuator_id = self.actuator_ids[i]
        #     l = self.data.ten_length[tendon_id]
        #     v = self.data.ten_velocity[tendon_id]
        #     k = self.model.tendon_stiffness[tendon_id]
        #     d = self.model.tendon_damping[tendon_id]
        #     passive = k * (l - rest_lengths[i]) + d * v
        #     active = self.data.actuator_force[actuator_id] * self.model.actuator_gear[actuator_id][0]

        #     forces_this_step.append({
        #         "tendon_id": tendon_id,
        #         "active": active,
        #         "passive": passive
        #     })
            # Note: actuator_force is already affected by ctrl and sim physics
            # active = self.data.actuator_force[tendon_id] * self.model.actuator_gear[tendon_id][0]            

            # print(f"Timestep {self.steps_from_reset}, Tendon {tendon_id}: "
            # f"l={l:.3f}, v={v:.3f}, k={k:.2f}, d={d:.2f}, rest={rest_lengths[tendon_id]:.3f}, "
            # f"active={active:.2f}, passive={passive:.2f}")

            
        actuator_names = ["T_M0_motor", "T_M1_motor", "T_M2_motor"]
        tendon_names = ["T_M0", "T_M1", "T_M2"]

        for i, actuator_name in enumerate(actuator_names):
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            ctrl = self.data.ctrl[actuator_id]
            force = self.data.actuator_force[actuator_id]
            gear = self.model.actuator_gear[actuator_id][0] if self.model.actuator_gear.shape[1] > 0 else 1.0
            active_force = ctrl * gear
            # print(f"gear={gear}" )
            # print(f"[DEBUG] Actuator {actuator_id}:  ctrl={ctrl:.3f}, force={active_force:.3f}")

            l = self.data.ten_length[i]
            v = self.data.ten_velocity[i]

            k = self.model.tendon_stiffness[i]
            d = self.model.tendon_damping[i]
            passive = k * (l - rest_lengths[i]) + d * v
            forces_this_step.append({
                "tendon_id": actuator_id,
                "active": active_force,
                "passive": passive
            })
            # print(f"[DEBUG] {tendon_names[i]} -> actuator_id={actuator_id}, ctrl={ctrl:.3f}, "
            #     f"active_force={active_force:.3f}, passive_force={passive:.3f}"
            #     f"k={k:.3f}, v = {v:.3f}, l ={l:.3f},  rest_lengths = {rest_lengths[i]:.3f}")

        self.tendon_force_episode.append(forces_this_step)

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
    Train PPO agent with given seed, save model and all data in categorized subfolders.
    """
    # ---------- Setup save folders ----------
    folder    = config.folder_name()
    tag       = '' #config.stiffness_tag
    save_path = f"./data/{folder}"
    os.makedirs(save_path, exist_ok=True)

    # Create subfolders
    model_dir = os.path.join(save_path, "model")
    rewards_dir = os.path.join(save_path, "rewards")
    disp_dir = os.path.join(save_path, "displacements")
    stiffness_dir = os.path.join(save_path, "stiffness")
    kinematic_dir = os.path.join(save_path, "kinematics")
    actuator_force_dir = os.path.join(save_path, "actuator_forces")
    tendon_length_dir = os.path.join(save_path, "tendon_lengths")
    tendon_force_dir = os.path.join(save_path, "tendon_forces")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(rewards_dir, exist_ok=True)
    os.makedirs(disp_dir, exist_ok=True)
    os.makedirs(stiffness_dir, exist_ok=True)
    os.makedirs(kinematic_dir, exist_ok=True)
    os.makedirs(actuator_force_dir, exist_ok=True)
    os.makedirs(tendon_length_dir, exist_ok=True)
    os.makedirs(tendon_force_dir, exist_ok=True)


    # ---------- Create environment ----------
    env = LegEnvBase(
        render_mode=None,
        stiffness_start=config.stiffness_start,
        stiffness_end=config.stiffness_end,
        num_epochs=config.num_epochs,
        max_episode_steps=config.max_episode_steps,
    )
    env.seed(seed)

    # ---------- Setup PPO and train ----------
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.lr,
        # ent_coef=0.05,  # 🔧 Try 0.02 or 0.05

        seed=seed,
        verbose=1,
        tensorboard_log=f"./tensorboard_logs/{folder}/",
        device='cpu'
    )

    model.learn(total_timesteps=config.total_timesteps)

    # ---------- Save model and data ----------
    model.save(f"{model_dir}/model_seed_{seed}.zip")

    np.save(f"{rewards_dir}/rewards_seed_{seed}.npy", env.rewards)
    np.save(f"{disp_dir}/displacements_seed_{seed}.npy", env.displacements)
    np.save(f"{stiffness_dir}/stiffness_seed_{seed}.npy", env.stiffness_history)

    np.save(f"{kinematic_dir}/qpos_seed_{seed}.npy", np.array(env.qpos_history, dtype=object))
    np.save(f"{kinematic_dir}/qvel_seed_{seed}.npy", np.array(env.qvel_history, dtype=object))

    np.save(f'{actuator_force_dir}/actuator_forces_seed_{seed}.npy', np.array(env.actuator_force_history, dtype=object))
    np.save(f"{tendon_length_dir}/tendon_length_seed_{seed}.npy", np.array(env.tendon_lengths_history, dtype=object))
    np.save(f"{tendon_force_dir}/tendon_force_seed_{seed}.npy", np.array(env.tendon_force_history, dtype=object))

    # ---------- Record demo video (optional) ----------
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

    # Use new subfolders for each file type
    rewards_dir = f'./data/{folder}/rewards'
    disp_dir = f'./data/{folder}/displacements'

    for seed in range(config.seed_start, config.seed_end + 1):
        r_path = f'{rewards_dir}/rewards_{tag}_seed_{seed}.npy'
        d_path = f'{disp_dir}/displacements_{tag}_seed_{seed}.npy'
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
