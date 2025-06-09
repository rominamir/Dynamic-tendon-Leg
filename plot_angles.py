import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm
import re

# Define folder and seeds
folder = r"C:\Users\User\Desktop\Dynamic-tendon-Leg\data\LegEnv_Jun07_constant_30k_constant_5e-04_PPO_seeds_100-100"
seeds = [113, 114, 115, 116, 117, 118]  # Example seeds



# Extract stiffness token (like 5k, 10k, 40k) using regex
match = re.search(r'(\d+k)', folder)
stiffness_label = match.group(1) if match else "Unknown"


# Set up figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, seed in enumerate(seeds):
    # Load data
    qpos_path = fr'{folder}/kinematics/qpos_seed_{seed}.npy'
    qpos_data = np.load(qpos_path, allow_pickle=True)
    last_qpos = np.array([np.array(x, dtype=np.float32) for x in qpos_data[-1]])
    last_qpos = np.stack(last_qpos)

    # Joint angles
    hip_angles = np.degrees(last_qpos[:, 1])
    knee_angles = np.degrees(last_qpos[:, 2])

    # Segments for color line
    points = np.array([hip_angles, knee_angles]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    time_steps = np.linspace(0, 1, len(segments))
    colors = cm.plasma(time_steps)
    lc = LineCollection(segments, colors=colors, linewidth=2)

    ax = axes[i]
    ax.add_collection(lc)
    ax.set_xlim(-20, 80)
    ax.set_ylim(-100, 20)

    # Only show labels/ticks on bottom-left subplot
    if i == 3:
        ax.set_xlabel("Hip Angle")
        ax.set_ylabel("Knee Angle")
        ax.set_xticks(np.linspace(-20, 80, 6))
        ax.set_yticks(np.linspace(-100, 20, 7))
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

    ax.set_title(f"Seed {seed}")

plt.suptitle(f"Angle-Angle Plots Across Seeds\nTendon Stiffness {stiffness_label.upper()}", fontsize=16)


fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
mappable = cm.ScalarMappable(cmap='plasma')
mappable.set_array(np.linspace(0, 1, 100))
fig.colorbar(mappable, cax=cbar_ax, label="Normalized Time")

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.show()
