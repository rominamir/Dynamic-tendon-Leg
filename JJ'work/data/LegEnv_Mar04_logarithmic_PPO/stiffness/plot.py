import matplotlib.pyplot as plt
import numpy as np

stiffness_data = np.load("stiffness_history.npy")

plt.plot(stiffness_data)
plt.xlabel("Timestep")
plt.ylabel("Stiffness")
plt.title("Stiffness Over Time")
plt.show()
