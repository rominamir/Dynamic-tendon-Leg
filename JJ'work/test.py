import gymnasium as gym

env = gym.make("Humanoid-v4", render_mode="human")
env.reset()
for _ in range(10000):
    env.step(env.action_space.sample())  # 采取随机动作
env.close()
