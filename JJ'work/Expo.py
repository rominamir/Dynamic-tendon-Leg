class ExpoLegEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 40}

    def __init__(self, xml_file='leg.xml', render_mode='none', seed=None,
                 stiffness_start=1000, stiffness_end=10000, num_timesteps=100_000, max_episode_steps=500):
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        observation_space = gym.spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float32)

        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5, render_mode=render_mode, observation_space=observation_space)

        if seed is not None:
            self.seed(seed)

        self.stiffness_start = stiffness_start
        self.stiffness_end = stiffness_end
        self.num_timesteps = num_timesteps
        self.max_episode_steps = max_episode_steps
        self.episode_count = 0

        self.total_episodes = num_timesteps / max_episode_steps
        self.stiffness_scaling = self.stiffness_start
        self.stiffness_history = []
        self.distances = []

    def update_stiffness(self, episode_number):
        k = 0.01192
        self.stiffness_scaling = self.stiffness_start + (
            (self.stiffness_end - self.stiffness_start) *
            (np.exp(k * episode_number) - 1) / (np.exp(k * self.total_episodes) - 1)
        )
        self.stiffness_scaling = min(self.stiffness_scaling, self.stiffness_end)
        self.apply_stiffness(self.stiffness_scaling)
        self.stiffness_history.append(self.stiffness_scaling)

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        reward = max(x_velocity, 0) - max(-x_velocity, 0)
        observation = self.get_obs()
        self.steps_from_reset += 1
        done = self.steps_from_reset >= self.max_episode_steps

        if done:
            self.update_stiffness(self.episode_count)
            self.episode_count += 1

        return observation, reward, done, False, {}

    def reset_model(self):
        self.update_stiffness(self.episode_count)
        self.steps_from_reset = 0
        self.start_x_position = self.data.qpos[0]  # 初始化起始位置
        return self.get_obs()
