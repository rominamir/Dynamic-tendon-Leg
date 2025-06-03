# import os
# import imageio
# import numpy as np
# from stable_baselines3 import PPO
# from env import LegEnvBase  # replace with correct import
# import imageio



# # ---- Setup ----
# folder = r"C:\Users\User\Desktop\New folder\AimIV\Dynamic-tendon-Leg\Dynamic-tendon-Leg\data\LegEnv_May28_constant30k_constant30kOnly_constant_5e-04_PPO"
# model_path = fr"{folder}/final_model_seed_100.zip"
# video_path = fr"{folder}/final_run_100.gif"

# # Load trained model and environment
# env = LegEnvBase(render_mode="rgb_array")  # Must be in rgb_array mode
# model = PPO.load(model_path)

# # ---- Record ----
# frames = []
# obs, _ = env.reset()
# done = False

# while not done:
#     frame = env.render()
#     frames.append(frame)
#     action, _ = model.predict(obs, deterministic=True)
#     obs, _, done, _, _ = env.step(action)

# env.close()

# # ---- Save Video ----

# imageio.mimsave(video_path, frames, duration=33)

# print(f"🎥 Saved video to: {video_path}")

import imageio
from stable_baselines3 import PPO
from env import LegEnvBase  # Replace with your actual env file
import mujoco

folder = r"C:\Users\User\Desktop\New folder\AimIV\Dynamic-tendon-Leg\Dynamic-tendon-Leg\data\LegEnv_May28_constant30k_constant30kOnly_constant_5e-04_PPO"
model_path = fr"{folder}/final_model_seed_100.zip"
video_path = fr"{folder}/final_run_100_40k.gif"

# Load trained model and environment
env = LegEnvBase(render_mode="human")  # Must be in rgb_array mode
model = PPO.load(model_path)


obs, _ = env.reset()


cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "Chassis_camera")

env.viewer.cam.fixedcamid = cam_id
env.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED


env.viewer.cam.fixedcamid = cam_id
env.viewer.cam.type = 2  # MJCAM_FIXED

frames = []
done = False
while not done:
    frame = env.render()
    frames.append(frame)
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _, _ = env.step(action)

env.close()

# Save as gif (use duration, not fps)
imageio.mimsave(video_path, frames, duration=33)
