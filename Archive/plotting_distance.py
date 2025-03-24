import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def distnace_boxplot():
    # Step 1: Load data from the .npy files
    npy_files = [f'./data/constant_10k_jan14_old/distance/distance_history_seed_{seed_value}.npy' for seed_value in range(1, 21)]  # Adjust filenames if necessary
    final_distances_cons= []

    for file in npy_files:
        try:
            data = np.load(file)  # Load the file
            #final_distance = np.sum(data)  # Assume distances are cumulative
            final_distance = data[-1]  # Assume distances are cumulative

            final_distances_cons.append(final_distance)
        except Exception as e:
            print(f"Error loading constant {file}: {e}")


    # Step 1: Load data from the .npy files
        npy_files_ = [f'./data/constant_10k_jan13_old/distance/distance_history_seed_{seed_value}.npy' for seed_value in range(1, 21)]  # Adjust filenames if necessary

    final_distances_dynamic = []

    for file in npy_files_:
        try:
            data = np.load(file)  # Load the file
            #final_distance = np.sum(data)  # Assume distances are cumulative
            final_distance = data[-1]  # Assume distances are cumulative

            final_distances_dynamic.append(final_distance)
        except Exception as e:
            print(f"Error loading dynamic {file}: {e}")


    f= [final_distances_cons, final_distances_dynamic]
    print (np.shape(f[0]), np.shape(f[1]))
    # Step 2: Create a boxplot
    plt.figure(figsize=(8, 6))
    #plt.boxplot(f[0], f[1], vert=False, patch_artist=True, boxprops=dict(facecolor="lightblue"), labels=['Cosntant', 'Monotonic'])
    df = pd.concat([ pd.Series(final_distances_cons),  pd.Series(final_distances_dynamic)], axis=1)
    sns.boxplot(data=df, )
    plt.legend(labels=['Monotonic_20k', 'Monotonic_10k'])
    plt.title("Final Distances Across 20 Runs")
    plt.xlabel("Distance")
    plt.yticks([1], ["Runs"])  # Single category for all runs
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Step 3: Show the plot
    plt.show()

#distnace_boxplot()




def plot_distance(filename):
    distance_history = np.load(filename)
    plt.figure(figsize=(10, 6))
    plt.plot(distance_history, marker='o', linestyle='-', color='b')
    plt.title("Distance Traveled per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Distance Traveled")
    plt.grid(True)
    plt.show()

# Example usage
#plot_distance('distance_history_seed_Dec09_7.npy')
#plot_distance('./data/Monotonic/distance/distance_history_seed_60.npy')
plot_distance('./data/Constant_10k_jan13/distance/distance_history_seed_1.npy')
plot_distance('./data/monotonic_20k/distance/distance_history_seed_1.npy')
plot_distance('./data/monotonic_10k_jan14/distance/distance_history_seed_1.npy')



def plot_mean_std(group1_files, group2_files, group3_files, group4_files ):
    """
    Plot mean and standard deviation of distances for two groups (constant and dynamic tendons).
    
    Parameters:
    - group1_files: List of file paths for group 1 (constant tendons).
    - group2_files: List of file paths for group 2 (dynamic tendons).
    """

    # Load all distance histories for both groups
    group1_data = [np.load(file) for file in group1_files]
    group2_data = [np.load(file) for file in group2_files]
    group3_data = [np.load(file) for file in group3_files]
    group4_data = [np.load(file) for file in group4_files]

    
    # Convert to arrays for easier computation
    group1_data = np.array(group1_data)
    group2_data = np.array(group2_data)
    group3_data = np.array(group3_data)
    group4_data = np.array(group4_data)


    
    # Compute mean and std for each episode
    group1_mean = group1_data.mean(axis=0)
    group1_std = group1_data.std(axis=0)
    
    group2_mean = group2_data.mean(axis=0)
    group2_std = group2_data.std(axis=0)

    group3_mean = group3_data.mean(axis=0)
    group3_std = group3_data.std(axis=0)

    group4_mean = group4_data.mean(axis=0)
    group4_std = group4_data.std(axis=0)
    
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Group 1: Constant Tendons _0k
    
    episodes = np.arange(1, group1_mean.shape[0] + 1)
    plt.plot(episodes, group1_mean, label="Constant stiffness_1k", color="blue", linestyle="-")
    plt.fill_between(episodes, group1_mean - group1_std, group1_mean + group1_std, color="blue", alpha=0.2) # label="Constant Tendons (Std)")
    
    # Group 2: Dynamic Tendons
    plt.plot(episodes, group2_mean, label="Constant stiffness_10k", color="green", linestyle="-")
    plt.fill_between(episodes, group2_mean - group2_std, group2_mean + group2_std, color="green", alpha=0.2)

    # Group 3: Dynamic Tendons
    plt.plot(episodes, group3_mean, label="Monotonic stiffness", color="purple", linestyle="-")
    plt.fill_between(episodes, group3_mean - group3_std, group3_mean + group3_std, color="purple", alpha=0.2)
    
        # Group 3: Dynamic Tendons
    plt.plot(episodes, group4_mean, label="Exponential stiffness", color="palevioletred", linestyle="-")
    plt.fill_between(episodes, group4_mean - group3_std, group3_mean + group3_std, color="palevioletred", alpha=0.2)
    
    # Formatting the plot
    plt.title("Comparison of Distance Traveled (Mean and Std) over 20 runs")
    plt.xlabel("Episode")
    plt.ylabel("Distance Traveled")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage


npy_files = [f'./data/monotonic_10k/distance/distance_history_seed_{seed_value}.npy' for seed_value in range(1, 21)]  # Adjust filenames if necessary
npy_files_ = [f'./data/monotonic_10k/distance/distance_history_seed_{seed_value}.npy' for seed_value in range(1, 21)]  # Adjust filenames if necessary
npy_files__ = [f'./data/monotonic_20k/distance/distance_history_seed_{seed_value}.npy' for seed_value in range(1, 21)]  # Adjust filenames if necessary
Exp_files = [f'./data/constant/distance/distance_history_seed_{seed_value}.npy' for seed_value in range(1, 21)]  # Adjust filenames if necessary



                
plot_mean_std(npy_files, npy_files_, npy_files__, Exp_files)








