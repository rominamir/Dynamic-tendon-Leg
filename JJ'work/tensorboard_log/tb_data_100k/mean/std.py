import pandas as pd
import matplotlib.pyplot as plt

# 文件路径（均值和标准差）
mean_files = {
    "ppo_log": "ppo_log.csv",
    "ppo_linear": "ppo_linear.csv",
    "ppo_exp": "ppo_exp.csv",
    "ppo_constant": "ppo_constant.csv",
    "a2c_log": "a2c_log.csv",
    "a2c_linear": "a2c_linear.csv",
    "a2c_exp": "a2c_exp.csv",
    "a2c_constant": "a2c_constant.csv",
}

std_files = {
    "ppo_log": "ppo_log_std.csv",
    "ppo_linear": "ppo_linear_std.csv",
    "ppo_exp": "ppo_exp_std.csv",
    "ppo_constant": "ppo_constant_std.csv",
    "a2c_log": "a2c_log_std.csv",
    "a2c_linear": "a2c_linear_std.csv",
    "a2c_exp": "a2c_exp_std.csv",
    "a2c_constant": "a2c_constant_std.csv",
}

# 读取数据
mean_data = {name: pd.read_csv(path) for name, path in mean_files.items()}
std_data = {name: pd.read_csv(path) for name, path in std_files.items()}

# 计算全局最小值和最大值，确保所有子图纵轴范围相同
all_values = []
for key in mean_data.keys():
    mean_rewards = mean_data[key]["Value"]
    std_rewards = std_data[key]["Value"]
    all_values.extend(mean_rewards - std_rewards)
    all_values.extend(mean_rewards + std_rewards)

y_min, y_max = min(all_values), max(all_values)  # 计算全局最小值和最大值

# 颜色映射
colors = {
    "ppo_log": "blue", "ppo_linear": "green", "ppo_exp": "red", "ppo_constant": "purple",
    "a2c_log": "blue", "a2c_linear": "green", "a2c_exp": "red", "a2c_constant": "purple"
}

# ------------- 2x4 子图矩阵绘制 -------------
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Mean and Standard Deviation of Rewards Over Training Steps")

for ax, (key, df) in zip(axes.flat, mean_data.items()):
    steps = df["Step"]
    mean_rewards = df["Value"]
    std_rewards = std_data[key]["Value"]  # 获取对应的标准差

    ax.plot(steps, mean_rewards, color=colors[key], label=key)  # 绘制均值曲线
    ax.fill_between(steps, mean_rewards - std_rewards, mean_rewards + std_rewards,
                    color=colors[key], alpha=0.2)  # 添加标准差阴影
    ax.set_title(key)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid()
    ax.set_ylim(y_min, y_max)  # 设置所有子图相同的纵轴范围

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
