import numpy as np
import matplotlib.pyplot as plt
import os

folder_name = r"C:\Users\User\Desktop\Dynamic-tendon-Leg\data\df_0.001_fs_25"


# Define your groups: folder name and seed range
groups = {
     "Constant 1": fr"{folder_name}\LegEnv_Aug07_constant_1_lr_5e-04_PPO\displacements",
    # "Constant 5": fr"{folder_name}\LegEnv_Jul24_constant_5_lr_5e-04_PPO\displacements",

    "Constant 10": fr"{folder_name}\LegEnv_Jul20_constant_1k_lr_5e-04_PPO\displacements",
    "Constant 30": fr"{folder_name}\LegEnv_Jul20_constant_3k_lr_5e-04_PPO\displacements",
    "Constant 50": fr"{folder_name}\LegEnv_Jul20_constant_5k_lr_5e-04_PPO\displacements",

    "Constant 100": fr"{folder_name}\LegEnv_Jul20_constant_10k_lr_5e-04_PPO\displacements",
    "Constant 200": fr"{folder_name}\LegEnv_Jul20_constant_20k_lr_5e-04_PPO\displacements",
    "Constant 400": fr"{folder_name}\LegEnv_Jul20_constant_40k_lr_5e-04_PPO\displacements",
   
    "Dynamic 10-100": fr"{folder_name}\LegEnv_Aug02_seeds_100-120_linear_10-100_lr_5e-04_PPO\displacements",
    "Dynamic 50-100": fr"{folder_name}\LegEnv_Aug01_seeds_100-110_linear_50-100_lr_5e-04_PPO\displacements",
    "Dynamic 50-200": fr"{folder_name}\LegEnv_Aug01_seeds_100-110_linear_50-200_lr_5e-04_PPO\displacements",



}

# # Set corresponding seeds for each group
# group_seeds = {
#      "Constant 1": range(100, 125),
#     # "Constant 5": range(100, 120),
    
#     "Constant 10": range(100, 125),
#     "Constant 30": range(100, 125),
#     "Constant 50": range(100, 125),

#     "Constant 100": range(100, 125),
#     "Constant 200": range(100, 120),
#     "Constant 400": range(100, 125),
#     "Dynamic 10-100": range(100, 120),
#     "Dynamic 50-100": range(100, 120),
#     "Dynamic 50-200": range(100, 120),
#     # "Constant 40": range(100, 110),
#     # "Constant 50": range(100, 120),
#     # "Constant 60": range(100, 110),
#     # "Constant 70": range(100, 110),
 
   
# }

total_seed = 125       
def moving_average(data, window_size=5):
    """Apply moving average smoothing."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


group_seeds = {label: range(100, total_seed) for label in groups.keys()}
colors = ['blue', 'green', 'red', 'purple', 'black', 'orange', 'cyan','magenta','brown','darkgrey']
window_size = 5

def plot_combined(groups, group_seeds, colors, window_size):
    """Plot all groups in a single plot."""
    plt.figure(figsize=(20, 12))
    
    for idx, (label, folder) in enumerate(groups.items()):
        seeds = group_seeds[label]
        data_list = []
        for seed in seeds:
            file_path = os.path.join(folder, f"displacements_seed_{seed}.npy")
            if os.path.exists(file_path):
                data_list.append(np.load(file_path))
            else:
                print(f"Warning: {file_path} not found.")
        
        if data_list:
            group_data = np.array(data_list)
            group_mean = group_data.mean(axis=0)
            group_std = group_data.std(axis=0)

            smoothed_mean = moving_average(group_mean, window_size)
            smoothed_std = moving_average(group_std, window_size)

            smoothed_mean = np.insert(smoothed_mean, 0, 0)
            smoothed_std = np.insert(smoothed_std, 0, 0)

            episodes = np.arange(len(smoothed_mean))

            plt.plot(episodes, smoothed_mean, label=label, color=colors[idx], linestyle='-')
            plt.fill_between(episodes, smoothed_mean - smoothed_std, smoothed_mean + smoothed_std, color=colors[idx], alpha=0.2)
        else:
            print(f"No data loaded for {label}.")
    
    plt.title("Displacement across different tendon stiffnesses (Mean ± Std)", fontsize=28)
    plt.xlabel("Episode", fontsize=18)
    plt.ylabel("Displacement (m)", fontsize=18)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_subplots(groups, group_seeds, colors, window_size):
    """Plot each group in its own subplot with unified styling and y-axis limits."""


    num_groups = len(groups)
    rows, cols = 3, 2
    fig, axs = plt.subplots(rows, cols, figsize=(20, 20))
    axs = axs.flatten()

    # -------- Step 1: Compute global y-limits --------
    all_ymins = []
    all_ymaxs = []

    for label, folder in groups.items():
        seeds = group_seeds[label]
        data_list = []
        for seed in seeds:
            file_path = os.path.join(folder, f"displacements_seed_{seed}.npy")
            if os.path.exists(file_path):
                data_list.append(np.load(file_path))
        if data_list:
            group_data = np.array(data_list)
            group_mean = group_data.mean(axis=0)
            group_std = group_data.std(axis=0)

            smoothed_mean = moving_average(group_mean, window_size)
            smoothed_std = moving_average(group_std, window_size)

            lower = smoothed_mean - smoothed_std
            upper = smoothed_mean + smoothed_std

            all_ymins.append(lower.min())
            all_ymaxs.append(upper.max())

    global_ymin = min(all_ymins)
    global_ymax = max(all_ymaxs)

    # -------- Step 2: Plot each group in its subplot --------
    for idx, (label, folder) in enumerate(groups.items()):
        seeds = group_seeds[label]
        data_list = []
        for seed in seeds:
            file_path = os.path.join(folder, f"displacements_seed_{seed}.npy")
            if os.path.exists(file_path):
                data_list.append(np.load(file_path))
            else:
                print(f"Warning: {file_path} not found.")

        ax = axs[idx]
        if data_list:
            group_data = np.array(data_list)
            group_mean = group_data.mean(axis=0)
            group_std = group_data.std(axis=0)

            smoothed_mean = moving_average(group_mean, window_size)
            smoothed_std = moving_average(group_std, window_size)

            smoothed_mean = np.insert(smoothed_mean, 0, 0)
            smoothed_std = np.insert(smoothed_std, 0, 0)

            episodes = np.arange(len(smoothed_mean))

            ax.plot(episodes, smoothed_mean, color=colors[idx], linestyle='-')
            ax.fill_between(episodes, smoothed_mean - smoothed_std, smoothed_mean + smoothed_std, color=colors[idx], alpha=0.2)
            
            label = label.replace("Constant", "").strip()
            ax.set_title(label)

            # Show ticks/labels only on bottom-left subplot (index 8)
            if idx != 8:
   
                ax.tick_params(labelbottom=False, labelleft=False)
                ax.set_xlabel("")
                ax.set_ylabel("")

            ax.set_ylim(global_ymin, global_ymax)
            ax.grid(True)
        else:
            ax.set_title(f"{label} (No data)")
            ax.axis('off')
                 
    fig.supxlabel("Episode", fontsize=16)
    fig.supylabel("Displacement (m)", fontsize=16)

    # Hide any unused subplots (in case fewer than 10 groups)
    for extra_ax in axs[num_groups:]:
        extra_ax.axis('off')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # reserve space for suptitle
    fig.subplots_adjust(hspace=0.4, wspace=0.01)
    fig.suptitle("Displacement across different tendon stiffnesses (Mean ± Std)", fontsize=22)
    plt.show()




def plot_all_subplots(groups, group_seeds, colors, window_size):
    """Plot each group in its own subplot with all seeds shown individually."""

    num_groups = len(groups)
    rows, cols = 2, 5
    fig, axs = plt.subplots(rows, cols, figsize=(16, 6)) # 20,20
    axs = axs.flatten()

    # -------- Step 1: Compute global y-limits --------
    all_ymins = []
    all_ymaxs = []

    for label, folder in groups.items():
        seeds = group_seeds[label]
        for seed in seeds:
            file_path = os.path.join(folder, f"displacements_seed_{seed}.npy")
            if os.path.exists(file_path):
                data = np.load(file_path)
                smoothed = moving_average(data, window_size)
                all_ymins.append(np.min(smoothed))
                all_ymaxs.append(np.max(smoothed))

    global_ymin = min(all_ymins) - 0.1  # Add a small margin for better visibility
    global_ymax = max(all_ymaxs) + 0.5  # Add a small margin for better visibility

    # -------- Step 2: Plot each group in its subplot --------
    for idx, (label, folder) in enumerate(groups.items()):
        seeds = group_seeds[label]
        ax = axs[idx]

        # plotted = False
        for seed in seeds:
            file_path = os.path.join(folder, f"displacements_seed_{seed}.npy")
            if os.path.exists(file_path):
                data = np.load(file_path)
                smoothed = moving_average(data, window_size)
                smoothed = np.insert(smoothed, 0, 0)  # add 0 at beginning to match your earlier behavior
                episodes = np.arange(len(smoothed))

                ax.plot(episodes, smoothed, color=colors[idx], alpha=0.3, linewidth=1)
                plotted = True
            else:
                print(f"Warning: {file_path} not found.")

        if plotted:
            label = label.replace("Constant", "").strip()
            ax.set_title(label)
            ax.set_ylim(global_ymin, global_ymax)

            ax.grid(True)
            if idx != 5:
                #ax.set_xlabel("Episode")
                #ax.set_ylabel("Displacement (m)")
    
                ax.tick_params(labelbottom=False, labelleft=False)
                ax.set_xlabel("")
                ax.set_ylabel("")
        else:
            
            ax.set_title(f"{label} (No data)")
            ax.axis('off')
       
        fig.supxlabel("Episode", fontsize=14)
        fig.supylabel("Cumulitive Displacement (m)", fontsize=14)

    # Hide any unused subplots
    for extra_ax in axs[num_groups:]:
        extra_ax.axis('off')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.subplots_adjust(hspace=0.4, wspace=0.01)
    fig.suptitle("Cumulitive displacement across different tendon stiffnesses [N/m]", fontsize=20)
    plt.savefig(os.path.join("displacement_subplots_all.png"), dpi=700)
    plt.show()

# ======================
# MAIN EXECUTION SECTION
# ======================
if __name__ == "__main__":
    mode = "all"  # Change to "combined" or "subplots" or "all"

    if mode == "combined":
        plot_combined(groups, group_seeds, colors, window_size)
    elif mode == "subplots": #subplot of mean and sd of each stiffness
        plot_subplots(groups, group_seeds, colors, window_size)
    elif mode == "all":
        plot_all_subplots(groups, group_seeds, colors, window_size)
    else:
        print("Invalid mode. Use 'combined' or 'subplots'.")