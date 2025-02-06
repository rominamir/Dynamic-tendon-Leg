import numpy as np
import matplotlib.pyplot as plt

# Load stiffness history from provided files
files = {
    "Logarithmic": "stiffness_history_seed_101_logarithmic.npy",
    "Constant": "stiffness_history_seed_101_constant.npy",
    "Exponential": "stiffness_history_seed_101_exponential.npy",
    "Linear": "stiffness_history_seed_101_linear.npy",
}

# Dictionary to store loaded data
stiffness_data = {}

# Load the data
for key, filename in files.items():
    try:
        stiffness_data[key] = np.load(filename)
    except Exception as e:
        print(f"Error loading {filename}: {e}")

# Plot the stiffness history
plt.figure(figsize=(10, 6))

for label, data in stiffness_data.items():
    plt.plot(data, label=label)

plt.xlabel("Timesteps")
plt.ylabel("Stiffness Value")
plt.title("Stiffness History Comparison")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
