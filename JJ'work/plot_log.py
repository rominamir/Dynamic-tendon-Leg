import numpy as np
import matplotlib.pyplot as plt

def logarithmic_stiffness_growth(stiffness_start, stiffness_end, total_episodes, growth_factor):
    """
    Computes stiffness values based on a logarithmic growth function.

    Parameters:
    - stiffness_start: Initial stiffness value.
    - stiffness_end: Maximum stiffness value.
    - total_episodes: Number of episodes for training.
    - growth_factor: Controls the growth rate (higher = faster initial increase).

    Returns:
    - episodes: Array of episode numbers.
    - stiffness_values: Array of stiffness values for each episode.
    """
    episodes = np.arange(total_episodes)
    progress = episodes / total_episodes
    adjusted_progress = 1 - np.exp(-growth_factor * progress)
    stiffness_values = stiffness_start + (stiffness_end - stiffness_start) * adjusted_progress
    return episodes, stiffness_values

# Parameters
stiffness_start = 2000
stiffness_end = 20000
total_episodes = 100
growth_factors = [2, 5, 10, 15]  # Different growth factors for comparison

# Plot Logarithmic Growth
plt.figure(figsize=(10, 5))
for growth_factor in growth_factors:
    episodes, stiffness_values = logarithmic_stiffness_growth(stiffness_start, stiffness_end, total_episodes, growth_factor)
    plt.plot(episodes, stiffness_values, label=f'Growth Factor {growth_factor}')

plt.xlabel('Episode')
plt.ylabel('Stiffness Value')
plt.title('Logarithmic Growth of Stiffness')
plt.legend()
plt.grid(True)
plt.show()
