import numpy as np
import matplotlib.pyplot as plt

def exponential_stiffness_growth(stiffness_start, stiffness_end, total_episodes, growth_factor):
    """
    Computes stiffness values based on an exponential growth function.

    Parameters:
    - stiffness_start: Initial stiffness value.
    - stiffness_end: Maximum stiffness value.
    - total_episodes: Number of episodes for training.
    - growth_factor: Controls the growth rate (higher = faster increase).

    Returns:
    - episodes: Array of episode numbers.
    - stiffness_values: Array of stiffness values for each episode.
    """
    episodes = np.arange(total_episodes)
    exponent = growth_factor * episodes
    stiffness_values = stiffness_start + (
        (stiffness_end - stiffness_start) * (np.exp(exponent) - 1) / (np.exp(growth_factor * total_episodes) - 1)
    )
    return episodes, stiffness_values

# Parameters
stiffness_start = 1000
stiffness_end = 10000
total_episodes = 100
growth_factors = [0.005, 0.01, 0.02, 0.05]  # Different growth factors for comparison

# Plot Exponential Growth
plt.figure(figsize=(10, 5))
for growth_factor in growth_factors:
    episodes, stiffness_values = exponential_stiffness_growth(stiffness_start, stiffness_end, total_episodes, growth_factor)
    plt.plot(episodes, stiffness_values, label=f'Growth Factor {growth_factor}')

plt.xlabel('Episode')
plt.ylabel('Stiffness Value')
plt.title('Exponential Growth of Stiffness')
plt.legend()
plt.grid(True)
plt.show()
