import os
#import gymnasium as gym

import mujoco
import numpy as np
from gym import utils, error
import gym
import mujoco_env
from math import exp


DEFAULT_CAMERA_CONFIG = {
    'distance': 5.0}


class LegEnv_arxiv(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 40,
    }

    def __init__(self,xml_file='leg.xml', render_mode='none', seed = None, 
                 stiffness_value = 1000,
                 num_timesteps=100000,
                max_episode_steps = 500
                 ):
                # Define action space: example with 3 continuous variables, with bounds from -1 to 1
        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)
        observation_space = gym.spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float32)

        # Initialize base classes
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5, render_mode=render_mode, observation_space=observation_space)

        if seed is not None:
            self.seed(seed)

        # Initialize stiffness parameters
        self._reset_noise_scale = 1e-8
        

        self.num_timesteps = num_timesteps
        self.max_episode_steps = max_episode_steps  # Maximum steps in each episode (e.g., 500)
        self.stiffness_history = []  # Keep track of stiffness over episodes

        # Progress tracking
        self.total_timesteps = 0  # Total number of timesteps elapsed
        self.episode_step = 0  # Steps in the current episode
        self.steps_from_reset = 0  
        self.episode_count = 0

        self.distances = []  # List to store distances for each episode
        self.current_distance = 0  # Distance in the current episode
        self.previous_position = None  # Track the previous position
        self.total_episodes=num_timesteps/max_episode_steps
        
        self.stiffness_value = stiffness_value  # Set your desired constant stiffness here
        self._set_constant_stiffness(self.stiffness_value)  # Apply the constant stiffness

        '''
        #self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32)
        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5, render_mode=render_mode, observation_space=observation_space)

        #names = [self.model.geom(i).name for i in range(self.model.ngeom)]
       # print(names)
        if seed is not None:
            self.seed(seed)
        #print (self.data.geom('Chassis_frame').xpos[0])
        mujoco.mj_kinematics(self.model, self.data)
        #print('raw access:\n', self.data.geom_xpos)

        self._reset_noise_scale = 1e-8
        self.max_episode_steps = max_episode_steps  # Maximum steps in each episode (e.g., 500)
        self.stiffness_history = []
        self.distances = []  # List to store distances for each episode
        self.current_distance = 0  # Distance in the current episode
        self.previous_position = None  # Track the previous position
       
        self.stiffness_value = stiffness_value  # Set your desired constant stiffness here
        self._set_constant_stiffness(self.stiffness_value)  # Apply the constant stiffness
        
        if seed is not None:
            self.seed(seed)
        '''
    def _set_constant_stiffness(self, stiffness_value):
        # Set constant stiffness for all tendons
        for i in range(self.model.ntendon):
            self.model.tendon_stiffness[i] = stiffness_value
        #print(f"Constant stiffness set to: {stiffness_value:.2f}")

    def step(self, action):
 
        x_position_before = self.data.qpos[0] #self.data.geom('Chassis_frame').xpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        #self.data.geom('Chassis_frame').xpos[0
        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)
        # z_position = self.sim.data.qpos[1]
        # planar_rotation = self.sim.data.qpos[2]
        ctrl_cost = 0 #self.control_cost(action)

        forward_reward =  x_velocity
        #standing_cost = self._forward_reward_weight * -10 *(abs(z_position-2)+abs(planar_rotation)) - abs(x_velocity)


        observation = self.get_obs()
          #Update distance traveled in the episode
        self.current_distance += abs(x_position_after - x_position_before)

        reward = forward_reward - ctrl_cost


        done = self.steps_from_reset >= self.max_episode_steps
        self.steps_from_reset += 1
        if done:
            self.distances.append(self.current_distance)  # Save total distance for the episode
            self.save_stiffness_history.append(self.self.model.tendon_stiffness[0])  # Save total distance for the episode

            self.current_distance = 0  # Reset for the next episode



        #######
        info = {
            #"values": state,
        }
        if self.render_mode == "human":
            self.render()



        return observation, reward, done, False, info

    def get_obs(self):
        position_tmp = self.data.qpos.flat.copy()
        velocity_tmp = self.data.qvel.flat.copy()

        """
        Here we put the cubes in the order we want. We save the pos and velocity of each cube (A,B,C) and for the observation, we put first cube in k = 11:16, and the second ....
        This way the cube can see the objects without respect to their color. The order should come from the input string, we can write the if here and put the string in self.operator.
        """
        # print(position.shape,velocity.shape)

        observation = np.concatenate((position_tmp, velocity_tmp)).ravel()

        return observation

    def reset_model(self):
    
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        self.steps_from_reset = 0

        qpos = self.init_qpos.copy()
        qvel = self.init_qvel + self._reset_noise_scale * np.random.randn(self.model.nv)

        # Add noise to other states but ensure x-position starts at zero
        qpos += np.random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qpos[0] = 0.0  # Ensure x-position starts at zero

        self.set_state(qpos, qvel)

        # Reset distance tracking
        self.current_distance = 0
        self.previous_position = self.data.qpos[0]  # Matches initial x-position

        self.stiffness_history.append(self.stiffness_value)

        observation = self.get_obs()
        return observation


    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def seed(self, seed=None):
        np.random.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def save_stiffness_history(self, filename):
        # Save the history of stiffness values to a file

        filename = os.path.normpath(filename)
        directory = os.path.dirname(filename)
        os.makedirs(directory, exist_ok=True)  # Ensure the directory exists

        np.save(filename, np.array(self.stiffness_history))

    def save_distances(self, filename):
          # Construct a clean path
        filename = os.path.normpath(filename)
        directory = os.path.dirname(filename)
        os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
        np.save(filename, self.distances)

    def load_distances(self, filename):
          # Construct a clean path
        filename = os.path.normpath(filename)
        directory = os.path.dirname(filename)
        os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
        self.distances = np.load(filename).tolist()

    def get_position(self):
        # Assuming the first element of qpos represents the x-position
        return self.data.qpos[0]

class ConLegEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 40,
    }
    
    def __init__(self, xml_file='leg.xml', render_mode='none', seed=None, 
             #stiffness_start=constant_stiff_value, stiffness_end=constant_stiff_value, 
             num_timesteps=100_000, max_episode_steps=500):
        # Define action space: example with 3 continuous variables, with bounds from -1 to 1
        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)
        observation_space = gym.spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float32)

        # Initialize base classes
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5, render_mode=render_mode, observation_space=observation_space)

        if seed is not None:
            self.seed(seed)

        # Initialize stiffness parameters
        self._reset_noise_scale = 1e-8
        self.constant_stiff_value = 5000
        self.stiffness_start = self.constant_stiff_value
        self.stiffness_end = self.constant_stiff_value
        self.num_timesteps = num_timesteps
        self.max_episode_steps = max_episode_steps  # Maximum steps in each episode (e.g., 500)
        self.stiffness_scaling = self.stiffness_start  # Current stiffness scaling factor
        self.stiffness_history = []  # Keep track of stiffness over episodes
        self.stiffness_increment = (self.stiffness_end - self.stiffness_start) / (self.num_timesteps / self.max_episode_steps)

        # Progress tracking
        self.total_timesteps = 0  # Total number of timesteps elapsed
        self.episode_step = 0  # Steps in the current episode
        self.steps_from_reset = 0  
        self.episode_count = 0

        self.distances = []  # List to store distances for each episode
        self.current_distance = 0  # Distance in the current episode
        self.previous_position = None  # Track the previous position
        self.total_episodes=num_timesteps/max_episode_steps
    
    def update_stiffness(self, episode_number):
            # Update stiffness using the natural exponential function
        self.stiffness_scaling = self.constant_stiff_value

        # Apply stiffness and ensure it doesn't exceed the maximum
        self.apply_stiffness(self.stiffness_scaling)

            # Append the stiffness value to the history
        #self.stiffness_history.append(self.stiffness_scaling)
        # Debug print
        print(f"Episode: {episode_number}, Stiffness: {self.stiffness_scaling}")

    def apply_stiffness(self, stiffness_value):
        for i in range(len(self.model.tendon_stiffness)):
            self.model.tendon_stiffness[i] = self.constant_stiff_value
        print(f"Applied stiffness: {stiffness_value}")


        

        print(f"Applied stiffness: {stiffness_value}")

    def step(self, action):
    # Track the x-position before the simulation step
        x_position_before = self.data.qpos[0]
        
        # Perform the simulation step
        self.do_simulation(action, self.frame_skip)
        
        # Track the x-position after the simulation step and calculate velocity
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        # Calculate reward
        ctrl_cost = 0  # Adjust this if you have control penalties
        #forward_reward = x_velocity
        #reward = forward_reward - ctrl_cost
        if x_velocity < 0:
            backward_penalty = abs(x_velocity)  
        else:
            backward_penalty = 0

        forward_reward = max(x_velocity, 0)
        reward = forward_reward - backward_penalty
        # Prepare observation
        observation = self.get_obs()

        #Update distance traveled in the episode
        self.current_distance += abs(x_position_after - x_position_before)
        
        # Check if the episode is done
        done = self.steps_from_reset >= self.max_episode_steps
        self.steps_from_reset += 1
        

        # Adjust stiffness at the end of the episode
        if done:

            net_displacement = x_position_after - self.start_x_position

            # 2) Append it to self.distances
            self.distances.append(net_displacement)
        
            #self.distances.append(self.current_distance)  # Save total distance for the episode
            self.current_distance = 0  # Reset for the next episode
            self.update_stiffness(self.episode_count)


        # Render if necessary
        if self.render_mode == "human":
            self.render()

        return observation, reward, done, False, {}


    def get_obs(self):
        position_tmp = self.data.qpos.flat.copy()
        velocity_tmp = self.data.qvel.flat.copy()
        return np.concatenate((position_tmp, velocity_tmp)).ravel()

 
    def reset_model(self, options=None):
        #super().reset_model(seed=seed)

        
        # Update stiffness at the start of each episode
        self.stiffness_scaling = self.stiffness_start + (self.episode_count *   self.stiffness_increment)
        self.stiffness_scaling = np.clip(self.stiffness_scaling, self.stiffness_start, self.stiffness_end)
        self.apply_stiffness(self.stiffness_scaling)  # Apply the updated stiffness

        self.stiffness_history.append(self.stiffness_scaling)
        print(f"Episode: {self.episode_count}, Stiffness: {self.stiffness_scaling}")

        # Increment episode count
        self.episode_count += 1


        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        self.steps_from_reset = 0

        qpos = self.init_qpos.copy() #+ np.random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * np.random.randn(self.model.nv)
        
        qpos[0] = 0.0  # Ensure x-position starts at zero

        
        self.start_x_position = self.data.qpos[0]

        self.set_state(qpos, qvel)
        
        self.current_distance = 0
        self.previous_position = self.get_position()  # Initial position



        return self.get_obs()


    def seed(self, seed=None):
        np.random.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def save_stiffness_history(self, filename):
        # Save the history of stiffness values to a file
        filename = os.path.normpath(filename)
        directory = os.path.dirname(filename)
        os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
        
        np.save(filename, np.array(self.stiffness_history))

    def save_distances(self, filename):
        filename = os.path.normpath(filename)
        directory = os.path.dirname(filename)
        os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
        np.save(filename, self.distances)

    def load_distances(self, filename):
        self.distances = np.load(filename).tolist()

    def get_position(self):
    # Assuming the first element of qpos represents the x-position
        return self.data.qpos[0]




class ConLegEnv_enhanced(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 40,
    }

    def __init__(
        self,
        xml_file="leg.xml",
        render_mode="none",
        seed=None,
        stiffness_value=5000,  # <--- Single constant stiffness value
        max_episode_steps= 500,
    ):
        """
        A simplified environment with constant stiffness throughout training.
        No dynamic stiffness updates.
        """
        # 1) Define action/observation spaces
        #    Example: 3 continuous actions in [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        observation_space = gym.spaces.Box(
            low=-10, high=10, shape=(6,), dtype=np.float32
        )

        # 2) Initialize parent classes (MujocoEnv + EzPickle)
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip=5,
            render_mode=render_mode,
            observation_space=observation_space,
        )

        # 3) Set the seed if provided
        if seed is not None:
            self.seed(seed)

        # 4) Store constant stiffness and apply it once
        self.constant_stiffness = stiffness_value
        self._apply_stiffness(self.constant_stiffness)

        # 5) Episode management
        self.max_episode_steps = max_episode_steps
        self.steps_from_reset = 0
        self._reset_noise_scale = 1e-8

        # 6) Optional: tracking total distance traveled
        self.distances = []
        self.current_distance = 0
        self.previous_position = None

        print(f"[Init] Constant stiffness set to {self.constant_stiffness}")

    def step(self, action):
        """
        Step the simulation forward with the given action.
        Compute reward, check done, and return observation.
        """
        # 1) Position before step
        x_pos_before = self.data.qpos[0]

        # 2) Simulate
        self.do_simulation(action, self.frame_skip)

        # 3) Position after step
        x_pos_after = self.data.qpos[0]
        x_velocity = (x_pos_after - x_pos_before) / self.dt

        # 4) Example reward structure
        #    Penalize backward movement, reward forward movement
        if x_velocity < 0:
            backward_penalty = abs(x_velocity)
        else:
            backward_penalty = 0
        forward_reward = max(x_velocity, 0)
        reward = forward_reward - backward_penalty

        # 5) Build observation
        obs = self._get_obs()

        # 6) Track distance
        self.current_distance += abs(x_pos_after - x_pos_before)

        # 7) Check termination
        self.steps_from_reset += 1
        done = self.steps_from_reset >= self.max_episode_steps

        # 8) If done, track total displacement or distance
        if done:
            net_displacement = x_pos_after - self.start_x_position
            self.distances.append(net_displacement)
            self.current_distance = 0  # Reset for next episode

        # 9) Optional: render if needed
        if self.render_mode == "human":
            self.render()

        return obs, reward, done, False, {}

    def reset_model(self, seed=None, options=None):
        """
        Resets the simulation state at the start of each episode.
        Applies the constant stiffness again to ensure consistency.
        """
        # 1) Reset counters
        self.steps_from_reset = 0

        # 2) Apply constant stiffness (no change over time)
        self._apply_stiffness(self.constant_stiffness)

        # 3) Initialize qpos/qvel with small noise
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        qvel += self._reset_noise_scale * np.random.randn(self.model.nv)
        qpos[0] = 0.0  # Make sure we start at x=0

        self.set_state(qpos, qvel)

        # 4) Track initial position
        self.start_x_position = self.data.qpos[0]
        self.previous_position = self.start_x_position

        return self._get_obs()

    def _get_obs(self):
        """
        Return current joint positions + velocities as observation.
        """
        position_tmp = self.data.qpos.flat.copy()
        velocity_tmp = self.data.qvel.flat.copy()
        return np.concatenate((position_tmp, velocity_tmp))

    # -----------------------------------------------------------------------
    # Helper Methods
    # -----------------------------------------------------------------------
    def _apply_stiffness(self, stiffness_value):
        """
        Assign a constant stiffness to all tendons in the model.
        """
        for i in range(self.model.ntendon):
            self.model.tendon_stiffness[i] = stiffness_value
        print(f"[Stiffness] Applied constant stiffness: {stiffness_value}")

    def get_position(self):
        return self.data.qpos[0]

    def viewer_setup(self):
        # Optional camera setup
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def seed(self, seed=None):
        np.random.seed(seed)
        self.action_space.seed(seed)
        if hasattr(self, "observation_space"):
            self.observation_space.seed(seed)

    def save_distances(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.save(filename, self.distances)

    def load_distances(self, filename):
        self.distances = np.load(filename).tolist()
