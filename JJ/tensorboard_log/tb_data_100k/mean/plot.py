import pandas as pd
import matplotlib.pyplot as plt

# 文件路径
file_paths = {
    "ppo_log": "ppo_log.csv",
    "ppo_linear": "ppo_linear.csv",
    "ppo_exp": "ppo_exp.csv",
    "ppo_constant": "ppo_constant.csv",
    "a2c_log": "a2c_log.csv",
    "a2c_linear": "a2c_linear.csv",
    "a2c_exp": "a2c_exp.csv",
    "a2c_constant": "a2c_constant.csv",
}

# 读取数据
dataframes = {name: pd.read_csv(path) for name, path in file_paths.items()}

# 颜色映射
colors = {
    "log": "blue",
    "linear": "green",
    "exp": "red",
    "constant": "purple"
}

# ------------- 绘制 PPO 训练奖励曲线 -------------
plt.figure(figsize=(12, 6))
for key in ["ppo_log", "ppo_linear", "ppo_exp", "ppo_constant"]:
    strategy = key.split("_")[1]  # 提取策略名称
    plt.plot(dataframes[key]["Step"], dataframes[key]["Value"], label=f"PPO {strategy}", color=colors[strategy])

plt.xlabel("Training Steps")
plt.ylabel("Reward Value")
plt.title("PPO Training Reward Curve for Different Learning Rate Schedules")
plt.legend()
plt.grid()
plt.show()

# ------------- 绘制 A2C 训练奖励曲线 -------------
plt.figure(figsize=(12, 6))
for key in ["a2c_log", "a2c_linear", "a2c_exp", "a2c_constant"]:
    strategy = key.split("_")[1]  # 提取策略名称
    plt.plot(dataframes[key]["Step"], dataframes[key]["Value"], label=f"A2C {strategy}", color=colors[strategy])

plt.xlabel("Training Steps")
plt.ylabel("Reward Value")
plt.title("A2C Training Reward Curve for Different Learning Rate Schedules")
plt.legend()
plt.grid()
plt.show()
