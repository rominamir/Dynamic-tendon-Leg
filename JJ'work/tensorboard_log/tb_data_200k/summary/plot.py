import os
import pandas as pd
import matplotlib.pyplot as plt

# 获取当前脚本所在的文件夹路径
folder_path = os.path.dirname(os.path.abspath(__file__))

# 获取所有CSV文件
files = sorted(os.listdir(folder_path))

# 筛选出 mean 和 std 文件
mean_files = [f for f in files if f.startswith("mean_ppo") and f.endswith(".csv")]
std_files = [f for f in files if f.startswith("std_ppo") and f.endswith(".csv")]

# 确保均值和标准差文件匹配
mean_files.sort()
std_files.sort()

# 读取数据
mean_dfs = {f: pd.read_csv(os.path.join(folder_path, f)) for f in mean_files}
std_dfs = {f.replace("mean", "std"): pd.read_csv(os.path.join(folder_path, f.replace("mean", "std"))) for f in
           mean_files}

# 确保数据格式正确
for key in mean_dfs:
    mean_dfs[key]["Step"] = mean_dfs[key]["Step"].astype(float)
    mean_dfs[key]["Value"] = mean_dfs[key]["Value"].astype(float)

for key in std_dfs:
    std_dfs[key]["Step"] = std_dfs[key]["Step"].astype(float)
    std_dfs[key]["Value"] = std_dfs[key]["Value"].astype(float)

# 获取所有均值数据的最大最小值
all_values = [df["Value"].values for df in mean_dfs.values()]
y_min = min(map(min, all_values))
y_max = max(map(max, all_values))

# 创建 2×4 子图布局
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

# 遍历 mean 和 std 文件，绘制图像
for i, key in enumerate(mean_files):
    mean_df = mean_dfs[key]
    std_df = std_dfs[key.replace("mean", "std")]

    x = mean_df["Step"]  # 取 Step 作为 x 轴
    mean_values = mean_df["Value"]  # 取 Value 作为均值
    std_values = std_df["Value"]  # 取 Value 作为标准差

    # 绘制曲线
    ax = axes[i]
    ax.plot(x, mean_values, label="Mean", color='b')
    ax.fill_between(x, mean_values - std_values, mean_values + std_values, color='b', alpha=0.2, label="Std Dev")

    # 设置统一纵坐标范围
    ax.set_ylim(y_min, y_max)

    # 设置标题
    title = key.replace("mean_ppo_", "").replace(".csv", "").replace("_", " ")
    ax.set_title(title)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Reward")
    ax.legend()

# 调整子图布局
plt.tight_layout()
plt.show()
