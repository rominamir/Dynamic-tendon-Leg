import os
import numpy as np
import matplotlib.pyplot as plt

# 获取当前目录
root_dir = os.getcwd()  # 获取当前 Python 文件所在目录

# 获取所有实验文件夹（排除 Python 文件）
experiment_folders = sorted([
    f for f in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, f)) and not f.endswith(".py")
])

# 创建存储统计值的列表
means = []
stds = []
labels = []

# 遍历每个实验文件夹
for folder in experiment_folders:
    distance_path = os.path.join(root_dir, folder, "distance")

    if os.path.exists(distance_path):
        values = []

        # 遍历所有 .npy 文件
        for file in os.listdir(distance_path):
            if file.endswith(".npy"):
                data = np.load(os.path.join(distance_path, file))
                values.append(data)

        if values:
            values = np.array(values)  # 形状 (num_seeds, timesteps)
            mean_val = np.mean(values, axis=0)  # 计算均值
            std_val = np.std(values, axis=0)  # 计算标准差

            means.append(mean_val)
            stds.append(std_val)
            labels.append(folder)

# 绘制 2×4 的子图
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()  # 将 2D 数组展平

for i in range(len(means)):
    ax = axes[i]
    ax.plot(means[i], label="Mean")
    ax.fill_between(range(len(means[i])), means[i] - stds[i], means[i] + stds[i], alpha=0.3, label="Std Dev")
    ax.set_title(labels[i])
    ax.legend()

plt.tight_layout()
plt.show()
