import numpy as np
import matplotlib.pyplot as plt
import os

def moving_average(data, window_size=5):
    """Apply moving average smoothing."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Define your groups: folder name and seed range
groups = {
    "Constant 5K": r"C:\Users\User\Desktop\Dynamic-tendon-Leg\data\LegEnv_Jun06_constant_5k_constant_5e-04_PPO_seeds_100-100\distance",
    "Constant 10K": r"C:\Users\User\Desktop\Dynamic-tendon-Leg\data\LegEnv_Jun06_constant_10k_constant_5e-04_PPO_seeds_100-100\distance",
    "Constant 15K": r"C:\Users\User\Desktop\Dynamic-tendon-Leg\data\LegEnv_Jun06_constant_15k_constant_5e-04_PPO_seeds_100-100\distance",
    "Constant 20K": r"C:\Users\User\Desktop\Dynamic-tendon-Leg\data\LegEnv_Jun06_constant_20k_constant_5e-04_PPO_seeds_100-100\distance",
    "Constant 25K": r"C:\Users\User\Desktop\Dynamic-tendon-Leg\data\LegEnv_Jun06_constant_10k_constant_5e-04_PPO_seeds_100-100\distance",
    "Constant 30K": r"C:\Users\User\Desktop\Dynamic-tendon-Leg\data\LegEnv_Jun06_constant_30k_constant_5e-04_PPO_seeds_100-100_default\distance",
    "Constant 35K": r"C:\Users\User\Desktop\Dynamic-tendon-Leg\data\LegEnv_Jun06_constant_35k_constant_5e-04_PPO_seeds_100-100\distance",
    "Constant 40K": r"C:\Users\User\Desktop\Dynamic-tendon-Leg\data\LegEnv_Jun06_constant_40k_constant_5e-04_PPO_seeds_100-100\distance",
    "Constant 45K": r"C:\Users\User\Desktop\Dynamic-tendon-Leg\data\LegEnv_Jun06_constant_45k_constant_5e-04_PPO_seeds_100-100\distance",
    "Constant 50K": r"C:\Users\User\Desktop\Dynamic-tendon-Leg\data\LegEnv_Jun06_constant_50k_constant_5e-04_PPO_seeds_100-100\distance",
}

# Set corresponding seeds for each group
group_seeds = {
    "Constant 5K": range(100, 110),
    "Constant 10K": range(100, 110),
    "Constant 15K": range(100, 110),
    "Constant 20K": range(100, 110),
    "Constant 25K": range(100, 110),
    "Constant 30K": range(100, 110),# Change these based on your seed filenames
    "Constant 35K": range(100, 110),
    "Constant 40K": range(100, 110),
    "Constant 45K": range(100, 110),
    "Constant 50K": range(100,110),
}

#
# Colors for plotting
colors = [  'blue', 'green', 'red', 'purple', 'black', 'orange', 'cyan','magenta','brown','darkgrey']


# Smoothing window size
window_size = 5

plt.figure(figsize=(20, 12))

for idx, (label, folder) in enumerate(groups.items()):
    seeds = group_seeds[label]
    data_list = []
    for seed in seeds:
        file_path = os.path.join(folder, f"displacement_history_seed_{seed}.npy")
        if os.path.exists(file_path):
            data_list.append(np.load(file_path))
        else:
            print(f"Warning: {file_path} not found.")
    
    if data_list:
        group_data = np.array(data_list)
        group_mean = group_data.mean(axis=0)
        group_std = group_data.std(axis=0)

        # Apply smoothing
        smoothed_mean = moving_average(group_mean, window_size)
        smoothed_std = moving_average(group_std, window_size)

        # Add zero at the beginning
        smoothed_mean = np.insert(smoothed_mean, 0, 0)
        smoothed_std = np.insert(smoothed_std, 0, 0)

        # Update episodes to start at 0
        episodes = np.arange(len(smoothed_mean))

        plt.plot(episodes, smoothed_mean, label=label, color=colors[idx], linestyle='-')
        plt.fill_between(episodes, smoothed_mean - smoothed_std, smoothed_mean + smoothed_std, color=colors[idx], alpha=0.2)

    else:
        print(f"No data loaded for {label}.")

# Final plot formatting
plt.title("Displacement across different tendon stiffnesses(Mean ± Std)", fontsize = 18)
plt.xlabel("Episode")
plt.ylabel("Displacement (m)")
plt.legend()
plt.grid(True)
plt.show()