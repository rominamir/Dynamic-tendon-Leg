import numpy as np
import matplotlib.pyplot as plt

file_path = "stiffness_history_seed_101.npy"
data = np.load(file_path)

plt.figure(figsize=(8, 5))
plt.plot(data, marker='o', linestyle='-')
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Stiffness History - stiffness_history_seed_101.npy")
plt.grid(True)


plt.show()
