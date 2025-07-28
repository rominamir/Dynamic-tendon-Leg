import os
import io
from datetime import datetime
from typing import Optional, List

import imageio
import gym
import torch
import mujoco
import mujoco_env
import numpy as np
from gym import utils
from stable_baselines3 import PPO

"""Constant-stiffness PPO environment & utilities (Python 3.8+ compatible).
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
# Stiffness Scheduler
# -----------------------------------------------------------------------------
class StiffnessScheduler:
    """Calculates the stiffness value based on the current epoch."""
    def __init__(self, schedule_type='constant', start=5000, end=50000, total_epochs=1000):
        self.schedule_type = schedule_type
        self.start = start
        self.end = end
        self.total_epochs = total_epochs

    def __call__(self, epoch: int) -> float:
        """Returns the stiffness for a given epoch."""
        if self.schedule_type == 'constant':
            return self.start
        elif self.schedule_type == 'linear':
            # Safety check to prevent division by zero if there's only one epoch.
            if self.total_epochs <= 1:
                return self.start
            # Calculate progress from 0.0 to 1.0 based on the current epoch.
            # (epoch - 1) is used because epochs are 1-indexed.
            progress = min((epoch - 1) / (self.total_epochs - 1), 1.0)
            return self.start + (self.end - self.start) * progress
        else:
            raise ValueError(f"Unsupported stiffness schedule: {self.schedule_type}")

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
        folder_seed_start: int = None,
        folder_seed_end: int = None,
        growth_type: str = "constant",
        ) -> None:
        def _fmt(v: int) -> str:
            """Formats an integer into a 'k' notation if large enough."""
            if v < 1000:
                return str(v)
            k = v / 1000
            return f"{int(k)}k" if v % 1000 == 0 else f"{k:.1f}k"
        self.growth_type = growth_type  # ✅ Save it for folder naming
        self.algorithm = "PPO"
        self.stiffness_start = stiffness_start
        self.stiffness_end = stiffness_end
        self.num_seeds = num_seeds
        self.total_timesteps = total_timesteps
        self.max_episode_steps = max_episode_steps
        self.num_epochs = num_epochs
        self.lr = lr
        self.prev_actuator_force = None
        self.prev_qpos = None

        # Use passed run_date if provided, otherwise use current date
        if run_date is None:
            self.run_date = datetime.now().strftime("%b%d")
        else:
            self.run_date = run_date
        self.seed_start = seed_start
        self.seed_end = seed_end
        self.lr_schedule = ConstantLR(lr)
        # Create a tag for stiffness range for folder naming.
        if self.growth_type.startswith("constant"):
            self.stiffness_tag = _fmt(self.stiffness_start)
        else:
            self.stiffness_tag = f"{_fmt(self.stiffness_start)}-{_fmt(self.stiffness_end)}"

        # Determine seed range for folder naming.
        self.folder_seed_start = folder_seed_start if folder_seed_start is not None else seed_start
        self.folder_seed_end = folder_seed_end if folder_seed_end is not None else seed_end

    def folder_name(self) -> str:
        """Generates a descriptive folder name based on training parameters."""
        def _fmt(v: int) -> str:
            """Inner helper to format numbers."""
            if v < 1000:
                return str(v)
            k = v / 1000
            return f"{int(k)}k" if v % 1000 == 0 else f"{k:.1f}k"
        seeds_tag = f"seeds_{self.folder_seed_start}-{self.folder_seed_end}"

        lr_tag = f"{self.lr:.0e}".replace("e-0", "e-").replace("e-", "e-0")  # Formats like '5e-04'
        start_tag = _fmt(self.stiffness_start)
        end_tag   = _fmt(self.stiffness_end)

        if self.growth_type.startswith("constant"):
            stiffness_part = start_tag
        else:
            stiffness_part = f"{start_tag}-{end_tag}"

        return f"LegEnv_{self.run_date}_{seeds_tag}_{self.growth_type}_{stiffness_part}_lr_{lr_tag}_PPO"



# -----------------------------------------------------------------------------
# Custom MuJoCo environment
# -----------------------------------------------------------------------------

class LegEnvBase(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 40} #40 with frame_skip 5

    def __init__(
        self,
        xml_file: str = "leg.xml",
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        stiffness_start: int = 5_000,
        stiffness_end: int = 50_000,
        num_epochs: int = 1_000,
        max_episode_steps: int = 1_000,
        stiffness_schedule_type: str = 'constant',
        total_training_steps: int = 1_000_000,
    ) -> None:
        # --- MujocoEnv setup ---
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float32)
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(
            self, xml_file, 25 , render_mode=render_mode, observation_space=self.observation_space
        )## frame skip was 5 now changing it to 10

        if seed is not None:
            self.seed(seed)

        # --- Cleaned and consolidated initialization logic ---
        self.max_episode_steps = max_episode_steps
        self.epoch_counter = 0
        self.steps_from_reset = 0
        self.global_step = 0
        self.step_counter = 0

        # --- Data logging structures (initialize all history lists first) ---
        self.stiffness_history: list[float] = []
        self.rewards: list[float] = []
        self.displacement_history = []
        self.qpos_history, self.qvel_history = [], []
        self.actuator_force_history, self.tendon_lengths_history, self.tendon_force_history = [], [], []

        # --- Per-episode data buffers ---
        self.qpos_episode, self.qvel_episode = [], []
        self.actuator_forces_episode, self.tendon_lengths_episode, self.tendon_force_episode = [], [], []

        self.reward_episode = 0.0
        self.x_start = 0.0

        # --- Scheduler setup ---
        # Calculate total epochs based on total training steps and episode length.
        self.total_epochs = total_training_steps // self.max_episode_steps
        self.stiffness_scheduler = StiffnessScheduler(
            schedule_type=stiffness_schedule_type,
            start=stiffness_start,
            end=stiffness_end,
            total_epochs=self.total_epochs
        )

        # Apply and record the initial stiffness after history lists are created.
        self.stiffness_scaling = stiffness_start
        self._apply_stiffness(self.stiffness_scaling)

        # --- Other attributes ---
        self.prev_actuator_force = None
        self.prev_qpos = None
        self.log_handle: Optional[io.TextIOBase] = None


    # -------------------- MuJoCo helpers --------------------
    def _apply_stiffness(self, value: float) -> None:
        """Applies a stiffness value to all tendons and records it."""
        for i in range(len(self.model.tendon_stiffness)):
            self.model.tendon_stiffness[i] = value
        # Append the new stiffness value to the history log.
        self.stiffness_history.append(value)

    # -------------------- RL loop --------------------
    def step(self, action):
        """Executes one timestep within the environment."""
        self.step_counter += 1 # A simple counter for steps.

        x_before = self.data.qpos[0]

        # Apply control using standard method
        self.do_simulation(action, self.frame_skip)

        # Stiffness update is now handled in `reset_model`, not here.
        # # new_stiffness = self.stiffness_scheduler(self.global_step)
        # # self._apply_stiffness(new_stiffness)

        x_after = self.data.qpos[0]
        # Calculate forward velocity as the primary reward signal.
        velocity = (x_after - x_before) / (self.dt * self.frame_skip)

        # Reward logic (example)
        reward = 1000 * velocity

        # Get current actuator forces and joint angles for penalty calculation.
        actuator_force = self.data.actuator_force.copy()
        qpos = self.data.qpos.copy()

        # Penalty coefficients (you can tune these)
        lambda_force = 10  # penalty weight for actuator force rate
        lambda_qpos = 0   # penalty weight for joint angle rate


        # Add penalties for rate of change to encourage smoother movements.
        if self.prev_actuator_force is not None and self.prev_qpos is not None:
            delta_force = actuator_force - self.prev_actuator_force
            delta_qpos = qpos - self.prev_qpos

            # Apply L2 penalty on rate of change
            force_penalty = lambda_force * np.sum(delta_force[1:])
            qpos_penalty = lambda_qpos * np.sum(delta_qpos**2)

            # Subtract penalties from reward
            reward -= force_penalty
            reward -= qpos_penalty

        self.reward_episode += reward

        # Update previous values for the next step's penalty calculation.
        self.prev_actuator_force = actuator_force
        self.prev_qpos = qpos

        # Log per-step data into episode buffers.
        self.qpos_episode.append(self.data.qpos.copy())
        self.qvel_episode.append(self.data.qvel.copy())
        self.tendon_lengths_episode.append(self.data.ten_length.copy())
        self.actuator_forces_episode.append(self.data.actuator_force.copy())

        rest_lengths = [2.048585725311554, 1.2340794371759454, 1.7340794371759456]
        self._log_tendon_forces(rest_lengths)

        self.steps_from_reset += 1
        self.global_step += 1

        # Check if the episode is done.
        done = self.steps_from_reset >= self.max_episode_steps
        if done:
            # --- Log episode reward ---
            self.rewards.append(self.reward_episode)

            # --- Append episode data to main history logs ---
            self.qpos_history.append(self.qpos_episode)
            self.qvel_history.append(self.qvel_episode)
            self.tendon_lengths_history.append(self.tendon_lengths_episode)
            self.actuator_force_history.append(self.actuator_forces_episode)
            self.tendon_force_history.append(self.tendon_force_episode)

            # --- Log final displacement for the episode ---
            displacement = x_after - self.x_start
            self.displacement_history.append(displacement)

            # --- Clear episode data buffers for the next episode ---
            self.qpos_episode = []
            self.qvel_episode = []
            self.tendon_lengths_episode = []
            self.actuator_forces_episode = []
            self.tendon_force_episode = []

            # --- Reset episode-specific counters ---
            self.reward_episode = 0.0
            self.steps_from_reset = 0

        return self.get_obs(), reward, done, False, {}

    def reset_model(self):
        """Resets the environment for a new episode."""
        self.epoch_counter += 1
        new_k = self.stiffness_scheduler(self.epoch_counter)
        self._apply_stiffness(new_k)
        
        # Calculate and apply the new stiffness for this new epoch.
        new_stiffness = self.stiffness_scheduler(self.epoch_counter)
        self._apply_stiffness(new_stiffness)

        # Optional: Print progress for debugging.
        if self.epoch_counter % 50 == 0 or self.epoch_counter == 1:
             print(f"Epoch: {self.epoch_counter}/{self.total_epochs}, New Stiffness: {new_stiffness:.2f}")

        # Reset state variables for penalty calculations.
        self.prev_actuator_force = None
        self.prev_qpos = None

        # Store the starting x-position to calculate displacement later.
        self.x_start = self.data.qpos[0]
        return self.get_obs()

    def get_obs(self):
        """Returns the current observation from the environment."""
        return np.concatenate((self.data.qpos.flat.copy(), self.data.qvel.flat.copy())).ravel()

    def _log_tendon_forces(self, rest_lengths: List[float]):
        """Compute and store active/passive forces dynamically from model values."""
        forces_this_step = []

        actuator_names = ["T_M0_motor", "T_M1_motor", "T_M2_motor"]

        for i, actuator_name in enumerate(actuator_names):
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            ctrl = self.data.ctrl[actuator_id]
            # Calculate active force from control signal and gear ratio.
            gear = self.model.actuator_gear[actuator_id][0] if self.model.actuator_gear.shape[1] > 0 else 1.0
            active_force = ctrl * gear

            # Calculate passive force from tendon properties (Hooke's law).
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

        self.tendon_force_episode.append(forces_this_step)

    def seed(self, seed: Optional[int] = None):
        """Sets the seed for this env's random number generators."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def viewer_setup(self):
        """Sets up the camera for rendering."""
        cam_id = mujoco.mj_name2id(self.model,
                                mujoco.mjtObj.mjOBJ_CAMERA,
                                "follow_leg")
        self.viewer.cam.fixedcamid = cam_id
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING

    def render(self, mode=None, *args, **kwargs):
        """Renders the environment with camera tracking."""
        if self.viewer is not None:
            try:
                chassis_pos = self.data.body('Chassis').xpos
                self.viewer.cam.lookat[:] = chassis_pos
            except Exception:
                pass

        if self.render_mode == "human" and self.viewer is not None:
            # Disable overlay creation by monkey patching the viewer
            self._get_viewer(self.render_mode)._create_overlay = lambda: None

        try:
            return super().render(*args, **kwargs)
        except TypeError:
            # Fallback for older gym versions
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
    save_path = f"./data/{folder}"
    os.makedirs(save_path, exist_ok=True)

    # Create subfolders for different data types.
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
    # Pass necessary parameters from the config to the environment.
    env = LegEnvBase(
        render_mode=None,
        stiffness_schedule_type=config.growth_type,
        stiffness_start=config.stiffness_start,
        stiffness_end=config.stiffness_end,
        num_epochs=config.num_epochs,
        max_episode_steps=config.max_episode_steps,
        total_training_steps=config.total_timesteps,
    )
    env.seed(seed)
    # ---------- log saves ----------
    logs_dir = os.path.join(save_path, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"log_seed_{seed}.txt")
    env.log_handle = open(log_path, "w", buffering=1)


    # ---------- Setup PPO and train ----------
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.lr_schedule,
        ent_coef=0.005,  # 🔧 Try 0.02 or 0.05
        clip_range=0.1,
        seed=seed,
        verbose=1,
        tensorboard_log=f"./tensorboard_logs/{folder}/",
        device='cpu'
    )
    try:
        model.learn(total_timesteps=config.total_timesteps)
    finally:
        env.log_handle.close()

    # ---------- Save model and data ----------
    model.save(f"{model_dir}/model_seed_{seed}.zip")

    # Save all collected data to .npy files.
    np.save(f"{rewards_dir}/rewards_{config.stiffness_tag}_seed_{seed}.npy", env.rewards)
    np.save(f"{disp_dir}/displacements_seed_{seed}.npy", np.array(env.displacement_history, dtype=float))
    np.save(f"{stiffness_dir}/stiffness_{config.stiffness_tag}_seed_{seed}.npy", env.stiffness_history)

    np.save(f"{kinematic_dir}/qpos_seed_{seed}.npy", np.array(env.qpos_history, dtype=object))
    np.save(f"{kinematic_dir}/qvel_seed_{seed}.npy", np.array(env.qvel_history, dtype=object))

    np.save(f'{actuator_force_dir}/actuator_forces_seed_{seed}.npy', np.array(env.actuator_force_history, dtype=object))
    np.save(f"{tendon_length_dir}/tendon_length_seed_{seed}.npy", np.array(env.tendon_lengths_history, dtype=object))
    np.save(f"{tendon_force_dir}/tendon_force_seed_{seed}.npy", np.array(env.tendon_force_history, dtype=object))

    # ---------- Record demo video (optional) ----------
    try:
        video_file = f"{save_path}/final_episode_seed_{seed}.mp4"
        _record_final_episode_video(config, model, seed, video_file)
    except Exception as exc:
        print(f"[warn] video capture failed for seed {seed}: {exc}")

    env.close()

def _record_final_episode_video(
    config: TrainingConfig, model, seed: int, outfile: str, max_frames: int = 1_000
) -> None:
    """Run one episode in an RGB-array env and save it as an MP4."""

    # Create an environment with a constant, final stiffness for video recording.
    env = LegEnvBase(
        render_mode="rgb_array",
        # Key change: Set schedule_type to 'constant' for the video.
        stiffness_schedule_type='constant',
        # Key change: Set the start stiffness to the final stiffness value.
        stiffness_start=config.stiffness_end,
        stiffness_end=config.stiffness_end,
        max_episode_steps=config.max_episode_steps,
        # total_training_steps is not used in constant mode but passed for consistency.
        total_training_steps=config.total_timesteps,
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
        r_path = f"{rewards_dir}/rewards_{config.stiffness_tag}_seed_{seed}.npy"
        d_path = f'{disp_dir}/displacements_seed_{seed}.npy'
        if os.path.isfile(r_path):
            reward_list.append(np.load(r_path))
        if os.path.isfile(d_path):
            disp_list.append(np.load(d_path))

    # Aggregate and save reward data.
    if reward_list:
        min_len = min(len(r) for r in reward_list)
        rewards = np.stack([r[:min_len] for r in reward_list])
        np.save(f"{save_dir}/reward_mean_{tag}.npy", rewards.mean(axis=0))
        np.save(f"{save_dir}/reward_se_{tag}.npy",
                rewards.std(axis=0, ddof=1) / np.sqrt(rewards.shape[0]))
        print(f"✅ Aggregated rewards saved → {save_dir}")

    # Aggregate and save displacement data.
    if disp_list:
        min_len = min(len(d) for d in disp_list)
        disps = np.stack([d[:min_len] for d in disp_list])
        np.save(f"{save_dir}/displacement_mean_{tag}.npy", disps.mean(axis=0))
        np.save(f"{save_dir}/displacement_se_{tag}.npy",
                disps.std(axis=0, ddof=1) / np.sqrt(disps.shape[0]))
        print(f"✅ Aggregated displacements saved → {save_dir}")