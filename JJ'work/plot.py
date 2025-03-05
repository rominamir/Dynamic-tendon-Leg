import numpy as np
import matplotlib.pyplot as plt


class StiffnessUpdater:
    def __init__(self, num_timesteps, stiffness_start, stiffness_end, growth_factor):
        self.num_timesteps = num_timesteps
        self.stiffness_start = stiffness_start
        self.stiffness_end = stiffness_end
        self.growth_factor = growth_factor
        self.stiffness_history = []

    def update_stiffness(self, global_step, growth_type):
        progress = global_step / max(1, self.num_timesteps)

        if growth_type == 'exponential':
            stiffness_scaling = self.stiffness_start + (
                    (self.stiffness_end - self.stiffness_start) *
                    (np.exp(self.growth_factor * progress) - 1) /
                    (np.exp(self.growth_factor) - 1)
            )
        elif growth_type == 'logarithmic':
            adjusted_progress = (1 - np.exp(-self.growth_factor * progress)) / (1 - np.exp(-self.growth_factor))
            stiffness_scaling = self.stiffness_start + (
                    (self.stiffness_end - self.stiffness_start) * adjusted_progress
            )


        elif growth_type == 'linear':
            stiffness_scaling = self.stiffness_start + (
                    progress * (self.stiffness_end - self.stiffness_start)
            )
        elif growth_type == 'constant':
            stiffness_scaling = self.stiffness_start
        else:
            raise ValueError("Invalid growth_type. Choose from ['exponential', 'logarithmic', 'linear', 'constant'].")

        stiffness_scaling = min(stiffness_scaling, self.stiffness_end)
        self.stiffness_history.append(stiffness_scaling)
        return stiffness_scaling

    def visualize_growth(self):
        timesteps = np.arange(self.num_timesteps)
        growth_types = ['exponential', 'logarithmic', 'linear', 'constant']

        plt.figure(figsize=(10, 6))

        for growth_type in growth_types:
            self.stiffness_history = []
            for t in timesteps:
                self.update_stiffness(t, growth_type)
            plt.plot(timesteps, self.stiffness_history, label=growth_type)

        plt.xlabel("Timesteps")
        plt.ylabel("Stiffness Scaling")
        plt.title("Stiffness Scaling Growth Trends")
        plt.legend()
        plt.grid(True)
        plt.show()


# 参数设置
num_timesteps = 200
stiffness_start = 2000
stiffness_end = 20000
growth_factor = 3  # 影响指数和对数增长的速率

# 创建对象并可视化
updater = StiffnessUpdater(num_timesteps, stiffness_start, stiffness_end, growth_factor)
updater.visualize_growth()
