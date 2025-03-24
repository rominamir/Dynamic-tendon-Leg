import os
import pandas as pd
import matplotlib.pyplot as plt

# 文件路径
file_paths = {
    "PPO_constant": "PPO_constant_mean.csv",
    "PPO_exp": "PPO_exp_mean.csv",
    "PPO_linear": "PPO_linear_mean.csv",
    "PPO_log": "PPO_log_mean.csv",
    "A2C_constant": "A2C_constant_mean.csv",
    "A2C_exp": "A2C_exp_mean.csv",
    "A2C_linear": "A2C_liner_mean.csv",
    "A2C_log": "A2C_log_mean.csv",
}

# 读取数据
data = {}
for key, path in file_paths.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"文件 {key} 的列名: {df.columns.tolist()}")  # 先打印列名检查
        data[key] = df
    else:
        print(f"警告: 文件 {path} 不存在，跳过此数据。")

# 可能的列名
step_column_options = ["Step", "step", "steps", "Timestep", "t"]
reward_column_options = ["Mean Reward", "mean_reward", "Reward", "reward", "returns"]

# 查找合适的列名
def find_column(df, options):
    for option in options:
        if option in df.columns:
            return option
    return None  # 没有找到

# 开始绘图
for plot_type, methods in [
    ("PPO", ["PPO_constant", "PPO_exp", "PPO_linear", "PPO_log"]),
    ("A2C", ["A2C_constant", "A2C_exp", "A2C_linear", "A2C_log"]),
]:
    plt.figure(figsize=(10, 5))
    for key in methods:
        if key in data:
            df = data[key]
            step_col = find_column(df, step_column_options)
            reward_col = find_column(df, reward_column_options)

            if step_col and reward_col:
                plt.plot(df[step_col], df[reward_col], label=key)
            else:
                print(f"⚠️ 无法在 {key} 中找到合适的 Step 或 Reward 列，跳过。")

    plt.xlabel("Training Steps")
    plt.ylabel("Mean Reward")
    plt.title(f"{plot_type}: Comparison of Different Learning Rate Strategies")
    plt.legend()
    plt.grid()
    plt.show()

# PPO vs A2C (相同学习率策略对比)
for strategy in ["constant", "exp", "linear", "log"]:
    plt.figure(figsize=(10, 5))
    for key in [f"PPO_{strategy}", f"A2C_{strategy}"]:
        if key in data:
            df = data[key]
            step_col = find_column(df, step_column_options)
            reward_col = find_column(df, reward_column_options)

            if step_col and reward_col:
                plt.plot(df[step_col], df[reward_col], label=key)
            else:
                print(f"⚠️ 无法在 {key} 中找到合适的 Step 或 Reward 列，跳过。")

    plt.xlabel("Training Steps")
    plt.ylabel("Mean Reward")
    plt.title(f"PPO vs A2C: {strategy.capitalize()} Learning Rate Strategy")
    plt.legend()
    plt.grid()
    plt.show()
