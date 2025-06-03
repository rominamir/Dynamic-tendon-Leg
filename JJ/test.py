import numpy as np
from env import LegEnvBase  # 假设你的 LegEnvBase 定义在 env.py 文件里

def test_env():
    # 创建环境实例
    env = LegEnvBase(xml_file='leg.xml', 
                      render_mode=None, 
                      stiffness_start=5000, 
                      stiffness_end=50000, 
                      growth_type='linear', 
                      growth_factor=3.0)
    
    # 重置环境，获取初始观察
    obs = env.reset()
    print("Initial observation:", obs)

    # 执行一个随机动作
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)

    print("Next observation:", obs)
    print("Reward:", reward)
    print("Done:", done)

    # 关闭环境
    env.close()
    print("Environment test complete.")

if __name__ == '__main__':
    test_env()
