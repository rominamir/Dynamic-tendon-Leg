import os
#import gymnasium as gym

import mujoco
import numpy as np
from gym import utils, error
import gym
import mujoco_env
from math import exp


DEFAULT_CAMERA_CONFIG = {
    'distance': 15.0}


class LegEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 40,
    }

    def __init__(self,xml_file='leg.xml', render_mode='none', seed = None, 
                 stiffness_value = 10000,
                 num_timesteps=100_000, max_episode_steps = 500
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
        #self.current_distance = 0  # Distance in the current episode
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
        x_velocity = ((x_position_after - x_position_before)  / self.dt)
        # z_position = self.sim.data.qpos[1]
        # planar_rotation = self.sim.data.qpos[2]
        ctrl_cost = 0 #self.control_cost(action)
        #forward_reward =  x_velocity
        #standing_cost = self._forward_reward_weight * -10 *(abs(z_position-2)+abs(planar_rotation)) - abs(x_velocity)
        if x_velocity < 0:
            backward_penalty = abs(x_velocity)  
        else:
            backward_penalty = 0

        forward_reward = max(x_velocity, 0)
        reward = forward_reward - backward_penalty
        # Prepare observation
       

        observation = self.get_obs()
          #Update distance traveled in the episode
        #self.current_distance += x_position_after - x_position_before

        #reward = forward_reward - ctrl_cost


        done = self.steps_from_reset >= self.max_episode_steps
        self.steps_from_reset += 1
        if done:
                        # 1) Compute net displacement from the beginning of the episode
            net_displacement = x_position_after - self.start_x_position

            # 2) Append it to self.distances
            self.distances.append(net_displacement)            
            #self.distances.append(self.current_distance)  # Save total distance for the episode
            #self.save_stiffness_history.append(self.self.model.tendon_stiffness[0])  # Save total distance for the episode

            #self.current_distance = 0  # Reset for the next episode



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
        #qpos += np.random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qpos[0] = 0.0  # Ensure x-position starts at zero


        self.start_x_position = self.data.qpos[0]

        self.set_state(qpos, qvel)

        # Reset distance tracking
        self.current_distance = 0
        
        self.previous_position = self.get_position()  # same as self.data.qpos[0] 

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



class DynamicLegEnv_base(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 40,
    }

    def __init__(self, xml_file='leg.xml', render_mode='none', seed=None, 
                 stiffness_start=0, stiffness_end=10000, num_timesteps=2000, max_episode_steps = 100):
        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5, render_mode=render_mode, observation_space=observation_space)

        if seed is not None:
            self.seed(seed)

        # Initialize stiffness parameters
        self._reset_noise_scale = 1e-8
        self.total_episodes = num_timesteps/max_episode_steps

        self.stiffness_start = 0
        self.stiffness_end = 10_000
        self.num_timesteps = num_timesteps
        #self.stiffness_increment = (self.stiffness_end - self.stiffness_start) / (self.num_timesteps / self.frame_skip)
        self.stiffness_increment = (self.stiffness_end - self.stiffness_start) / (self.total_episodes)

        self.stiffness_scaling = self.stiffness_start
        self.stiffness_history = []

    def step(self, action):
        # Update stiffness incrementally
        self.stiffness_scaling += self.stiffness_increment
        for i in range(self.model.ntendon):
            self.model.tendon_stiffness[i] = self.stiffness_scaling

        self.stiffness_history.append(self.stiffness_scaling)
        
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        # Calculate reward
        ctrl_cost = 0
        forward_reward = x_velocity
        reward = forward_reward - ctrl_cost

        # Prepare observation and check if done
        observation = self.get_obs()
        done = self.steps_from_reset > 500
        self.steps_from_reset += 1

        if self.render_mode == "human":
            self.render()

        return observation, reward, done, False, {}

    def get_obs(self):
        position_tmp = self.data.qpos.flat.copy()
        velocity_tmp = self.data.qvel.flat.copy()
        return np.concatenate((position_tmp, velocity_tmp)).ravel()

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        self.steps_from_reset = 0

        qpos = self.init_qpos + np.random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * np.random.randn(self.model.nv)
        self.set_state(qpos, qvel)

        # Reset stiffness scaling to start value on reset
        self.stiffness_scaling = self.stiffness_start
        return self.get_obs()

    def seed(self, seed=None):
        np.random.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def save_stiffness_history(self, filename):
        np.save(filename, np.array(self.stiffness_history))

class DynamicLegEnv_base_(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 40,
    }

    def __init__(self, xml_file='leg.xml', render_mode='none', seed=None, 
                 stiffness_start=0, stiffness_end=10000, num_timesteps=2000, max_episode_steps = 100):
        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5, render_mode=render_mode, observation_space=observation_space)

        if seed is not None:
            self.seed(seed)

        # Initialize stiffness parameters
        self._reset_noise_scale = 1e-8
        self.total_episodes = num_timesteps/max_episode_steps

        self.stiffness_start = 0
        self.stiffness_end = 10_000
        self.num_timesteps = num_timesteps
        #self.stiffness_increment = (self.stiffness_end - self.stiffness_start) / (self.num_timesteps / self.frame_skip)
        self.stiffness_increment = (self.stiffness_end - self.stiffness_start) / (self.total_episodes)

        self.stiffness_scaling = self.stiffness_start
        self.stiffness_history = []
        self.distances = []  # List to store distances for each episode

    def step(self, action):
        # Update stiffness incrementally
        self.stiffness_scaling += self.stiffness_increment
        for i in range(self.model.ntendon):
            self.model.tendon_stiffness[i] = self.stiffness_scaling

        self.stiffness_history.append(self.stiffness_scaling)
        
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        # Calculate reward
        ctrl_cost = 0
        forward_reward = x_velocity
        reward = forward_reward - ctrl_cost

        # Prepare' observation and check if done
        observation = self.get_obs()
        done = self.steps_from_reset > 500
        self.steps_from_reset += 1
        if done:
            # 1) Compute' net displacement from the beginning of the episode

            net_displacement = x_position_after - self.start_x_position

            # 2) Append it to self.distances

            self.distances.append(net_displacement)
            #### self.update_stiffness(self.episode_count)
            #self.distances.append(self.current_distance)  # Save total distance forthe episode]
            #self.current_ distance = 0  # Reset for the next episode


        if self.render_mode == "human":
            self.render()

        return observation, reward, done, False, {}

    def get_obs(self):
        position_tmp = self.data.qpos.flat.copy()
        velocity_tmp = self.data.qvel.flat.copy()
        return np.concatenate((position_tmp, velocity_tmp)).ravel()

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        self.steps_from_reset = 0

        qpos = self.init_qpos + np.random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * np.random.randn(self.model.nv)
        self.set_state(qpos, qvel)

        # Reset stiffness scaling to start value on reset
        self.stiffness_scaling = self.stiffness_start

                # Ensure the x-position starts at zero
        qpos[0] = 0.0
        # Actually apply to simulator
        self.set_state(qpos, qvel)

        # 5) Record the new initial position as start_x_position
        self.start_x_position = self.data.qpos[0]
        self.previous_position = self.get_position()  # or just self.data.qpos[0]

        return self.get_obs()

    def seed(self, seed=None):
        np.random.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def save_stiffness_history(self, filename):
        np.save(filename, np.array(self.stiffness_history))
    def save_distances(self, filename):
        filename = os.path.normpath(filename)
        directory = os.path.dirname(filename)
        os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
        np.save(filename, self.distances)

 
class DynamicLegEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 40,
    }

    def __init__(self, xml_file='leg.xml', render_mode='none', seed=None, 
             stiffness_start=1000, stiffness_end=5000, 
             num_timesteps=500_000, max_episode_steps=500):
        # Define action space: example with 3 continuous variables, with bounds from -1 to 1
        self.action_space = gym.spaces.Box(low=-1, high=1.0, shape=(3,), dtype=np.float32)

        #self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)
        observation_space = gym.spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float32)

        # Initialize base classes
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5, render_mode=render_mode, observation_space=observation_space)

        if seed is not None:
            self.seed(seed)

        # Initialize stiffness parameters
        self._reset_noise_scale = 1e-8

        self.stiffness_start = stiffness_start
        self.stiffness_end = stiffness_end
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
    
    '''
    def update_stiffness(self, episode_number):
        # Compute stiffness based on the episode number 
        progress = episode_number / self.total_episodes  
        self.stiffness_scaling = self.stiffness_start * ((self.stiffness_end / self.stiffness_start) ** progress)
        self.stiffness_scaling = min(self.stiffness_scaling, self.stiffness_end)
        self.apply_stiffness(self.stiffness_scaling)
        print(f"Episode: {episode_number}, Stiffness: {self.stiffness_scaling}")
    '''
    def apply_stiffness(self, stiffness_value):
        for i in range(len(self.model.tendon_stiffness)):
            self.model.tendon_stiffness[i] = stiffness_value
        print(f"Applied stiffness: {self.model.tendon_stiffness[0] }")

    def step(self, action):
    # Track the x-position before the simulation step
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        
        x_velocity = (x_position_after - x_position_before) / self.dt

        # Calculate reward
        ctrl_cost = 0  # Adjust this if you have control penalties
        if x_velocity < 0:
            backward_penalty = abs(x_velocity)  
        else:
            backward_penalty = 0
        
        forward_reward = max(x_velocity, 0)
        reward = forward_reward - backward_penalty
        
        #forward_reward = x_velocity
        #reward = forward_reward - ctrl_cost

        # Prepare observation
        observation = self.get_obs()

        #Update distance traveled in the episode
        #self.current_distance += x_position_after - x_position_before
        
        # Check if the episode is done
        done = self.steps_from_reset >= self.max_episode_steps
        self.steps_from_reset += 1
        

        # Adjust stiffness at the end of the episode
        if done:
            # 1) Compute net displacement from the beginning of the episode
            net_displacement = x_position_after - self.start_x_position

            # 2) Append it to self.distances
            self.distances.append(net_displacement)
            #### self.update_stiffness(self.episode_count)
            #self.distances.append(self.current_distance)  # Save total distance for the episode
            self.current_distance = 0  # Reset for the next episode


        # Render if necessary
        if self.render_mode == "human":
            self.render()

        return observation, reward, done, False, {}


    def get_obs(self):
        position_tmp = self.data.qpos.flat.copy()
        velocity_tmp = self.data.qvel.flat.copy()
        return np.concatenate((position_tmp, velocity_tmp)).ravel()

    def reset_model(self, options=None):
        # 1) Compute the linearly updated stiffness for this episode
        self.stiffness_scaling = self.stiffness_start + (self.episode_count * self.stiffness_increment)
        self.stiffness_scaling = np.clip(self.stiffness_scaling, self.stiffness_start, self.stiffness_end)
        self.apply_stiffness(self.stiffness_scaling)

        # 2) Log it
        self.stiffness_history.append(self.stiffness_scaling)
        print(f"Episode: {self.episode_count}, (Linear) Stiffness: {self.stiffness_scaling}")

        # 3) Increment episode count (so next time, we move to the next stiffness)
        self.episode_count += 1

        # 4) Reset counters and states
        self.steps_from_reset = 0
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel + self._reset_noise_scale * np.random.randn(self.model.nv)

        # Ensure the x-position starts at zero
        qpos[0] = 0.0
        # Actually apply to simulator
        self.set_state(qpos, qvel)

        # 5) Record the new initial position as start_x_position
        self.start_x_position = self.data.qpos[0]
        self.previous_position = self.get_position()  # or just self.data.qpos[0]

        return self.get_obs()

    
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
        filename = os.path.normpath(filename)
        directory = os.path.dirname(filename)
        os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
        np.save(filename, self.distances)

    def load_distances(self, filename):
        self.distances = np.load(filename).tolist()

    def get_position(self):
    # Assuming the first element of qpos represents the x-position
        return self.data.qpos[0]


class ExpoLegEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 40,
    }

    def __init__(self, xml_file='leg.xml', render_mode='none', seed=None, 
             stiffness_start=1000, stiffness_end=10000, num_timesteps=100_000, max_episode_steps=500):
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

        self.stiffness_start = stiffness_start
        self.stiffness_end = stiffness_end
        self.num_timesteps = num_timesteps
        self.max_episode_steps = max_episode_steps  # Maximum steps in each episode (e.g., 500)
        self.stiffness_scaling = self.stiffness_start  # Current stiffness scaling factor
        self.stiffness_history = []  # Keep track of stiffness over episodes
        self.stiffness_increment = (self.stiffness_end - self.stiffness_start) / (self.num_timesteps / self.max_episode_steps)
    
        # Progress tracking
        self.total_episodes = num_timesteps/max_episode_steps
        self.total_timesteps = 0  # Total number of timesteps elapsed
        self.episode_step = 0  # Steps in the current episode
        self.steps_from_reset = 0  
        self.episode_count = 0

        self.distances = []  # List to store distances for each episode
        self.current_distance = 0  # Distance in the current episode
        self.previous_position = None  # Track the previous position
    

    def update_stiffness(self, episode_number):

        k =0.01192 #where we have [100, 7070]
        # Larger k -> faster growth earlier; smaller k -> slower initial growth
        
        # Update stiffness using the natural exponential function
        self.stiffness_scaling = self.stiffness_start + (
            (self.stiffness_end - self.stiffness_start) *  (np.exp(k * episode_number) - 1) / (np.exp(k* self.total_episodes) - 1)

        )

        # Apply stiffness and ensure it doesn't exceed the maximum
        self.stiffness_scaling = min(self.stiffness_scaling, self.stiffness_end)
        self.apply_stiffness(self.stiffness_scaling)

             # Append the stiffness value to the history
        self.stiffness_history.append(self.stiffness_scaling)
        # Debug print
        print(f"Episode: {episode_number}, Stiffness: {self.stiffness_scaling}")

    def apply_stiffness(self, stiffness_value):
        for i in range(len(self.model.tendon_stiffness)):
            self.model.tendon_stiffness[i] = stiffness_value
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

        # Update distance traveled in the episode
        self.current_distance += abs(x_position_after - x_position_before)

        # Increment step count
        self.steps_from_reset += 1

        # Check if the episode is done
        done = self.steps_from_reset >= self.max_episode_steps

        if done:
            # Update stiffness using the current episode count
            self.update_stiffness(self.episode_count)
            self.distances.append(self.current_distance)  # Save total distance for the episode
            self.current_distance = 0  # Reset for the next episode

        # Render if necessary
        if self.render_mode == "human":
            self.render()

        return observation, reward, done, False, {}


        

    def get_obs(self):
        position_tmp = self.data.qpos.flat.copy()
        velocity_tmp = self.data.qvel.flat.copy()
        return np.concatenate((position_tmp, velocity_tmp)).ravel()

 
    '''
    def reset_model(self, options=None):
        # Remove resetting of stiffness history
        # Only update the stiffness at the start of an episode
        self.stiffness_scaling = self.stiffness_start + (self.episode_count * self.stiffness_increment)
        self.stiffness_scaling = np.clip(self.stiffness_scaling, self.stiffness_start, self.stiffness_end)
        self.apply_stiffness(self.stiffness_scaling)  # Apply the updated stiffness

        # Debug log
        print(f"Episode: {self.episode_count}, Stiffness: {self.stiffness_scaling}")

        # Increment episode count
        self.episode_count += 1

        # Reset noise and position/velocity
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        self.steps_from_reset = 0

        qpos = self.init_qpos + np.random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * np.random.randn(self.model.nv)

        self.set_state(qpos, qvel)
        
        self.current_distance = 0
        self.previous_position = self.get_position()  # Initial position

        return self.get_obs()
    '''
    def reset_model(self, options=None):
        
        # Remove resetting of stiffness history
        # Only update the stiffness at the start of an episode
        self.stiffness_scaling = self.stiffness_start + (self.episode_count * self.stiffness_increment)
        self.stiffness_scaling = np.clip(self.stiffness_scaling, self.stiffness_start, self.stiffness_end)
        self.apply_stiffness(self.stiffness_scaling)  # Apply the updated stiffness

        # Increment episode count
        self.episode_count += 1

        # Reset noise and position/velocity
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        self.steps_from_reset = 0

        qpos = self.init_qpos + np.random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * np.random.randn(self.model.nv)

        # Explicitly reset x-position to zero
        #qpos[0] = 0
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
       # Construct a clean path
        filename = os.path.normpath(filename)
        directory = os.path.dirname(filename)
        os.makedirs(directory, exist_ok=True)  # Ensure the directory exists

        np.save(filename, self.distances)

    def load_distances(self, filename):
        self.distances = np.load(filename).tolist()

    def get_position(self):
    # Assuming the first element of qpos represents the x-position
        return self.data.qpos[0]




class LogLegEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 40,
    }

    def __init__(self, xml_file='leg.xml', render_mode='none', seed=None, 
             stiffness_start=1000, stiffness_end=20000, num_timesteps=100000, max_episode_steps=500):
        # Define action space: example with 3 continuous variables, with bounds from -1 to 1
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)

        #self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float32)

        # Initialize base classes
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5, render_mode=render_mode, observation_space=self.observation_space)

        if seed is not None:
            self.seed(seed)

        # Initialize stiffness parameters
        self._reset_noise_scale = 1e-8

        self.stiffness_start = stiffness_start
        self.stiffness_end = stiffness_end
        self.num_timesteps = num_timesteps
        self.max_episode_steps = max_episode_steps  # Maximum steps in each episode (e.g., 500)
        self.stiffness_scaling = self.stiffness_start  # Current stiffness scaling factor
        self.stiffness_history = []  # Keep track of stiffness over episodes
        self.stiffness_increment = (self.stiffness_end - self.stiffness_start) / (self.num_timesteps / self.max_episode_steps)
    
        # Progress tracking
        self.total_episodes = num_timesteps/max_episode_steps
        self.total_timesteps = 0  # Total number of timesteps elapsed
        self.episode_step = 0  # Steps in the current episode
        self.steps_from_reset = 0  
        self.episode_count = 0

        self.distances = []  # List to store distances for each episode
        self.current_distance = 0  # Distance in the current episode
        self.previous_position = None  # Track the previous position
    
  
    def update_stiffness(self, episode_number):
            #def update_stiffness(self, episode_number):
        # Calculate the progress


        progress = episode_number / self.total_episodes  # Ensure self.total_episodes is set

        # Adjusted exponential scaling
        growth_rate = 5  # Try reducing this to make it shallower
        adjusted_progress = 1 - np.exp(-growth_rate * progress)

        # Update stiffness
        self.stiffness_scaling = self.stiffness_start + (
            (self.stiffness_end - self.stiffness_start) * adjusted_progress  )

        self.stiffness_scaling = min(self.stiffness_scaling, self.stiffness_end)

        # Apply the updated stiffness
        self.apply_stiffness(self.stiffness_scaling)

        # Append the stiffness value to the history
        self.stiffness_history.append(self.stiffness_scaling)

        # Debug logging
        print(f"Episode: {episode_number}, Stiffness: {self.stiffness_scaling}")

    def apply_stiffness(self, stiffness_value):
        for i in range(len(self.model.tendon_stiffness)):
            self.model.tendon_stiffness[i] = stiffness_value
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

        # Update distance traveled in the episode
        self.current_distance += abs(x_position_after - x_position_before)

        # Increment step count
        self.steps_from_reset += 1

        # Check if the episode is done
        done = self.steps_from_reset >= self.max_episode_steps

        if done:
            # Update stiffness using the current episode count
            self.update_stiffness(self.episode_count)
            self.distances.append(self.current_distance)  # Save total distance for the episode
            self.current_distance = 0  # Reset for the next episode

        # Render if necessary
        if self.render_mode == "human":
            self.render()

        return observation, reward, done, False, {}



    def get_obs(self):
        position_tmp = self.data.qpos.flat.copy()
        velocity_tmp = self.data.qvel.flat.copy()
        return np.concatenate((position_tmp, velocity_tmp)).ravel()

 
    def reset_model(self, options=None):
        # Remove resetting of stiffness history
        # Only update the stiffness at the start of an episode
        self.stiffness_scaling = self.stiffness_start + (self.episode_count * self.stiffness_increment)
        self.stiffness_scaling = np.clip(self.stiffness_scaling, self.stiffness_start, self.stiffness_end)
        self.apply_stiffness(self.stiffness_scaling)  # Apply the updated stiffness

        # Debug log
        print(f"Episode: {self.episode_count}, Stiffness: {self.stiffness_scaling}")

        # Increment episode count
        self.episode_count += 1

        # Reset noise and position/velocity
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        self.steps_from_reset = 0

        qpos = self.init_qpos + np.random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * np.random.randn(self.model.nv)

        self.set_state(qpos, qvel)
        
        self.current_distance = 0
        self.previous_position = self.get_position()  # Initial position

        return self.get_obs()


    def seed(self, seed=None):
        np.random.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def save_stiffness_history(self, filename):
        # Construct a clean path
        filename = os.path.normpath(filename)
        directory = os.path.dirname(filename)
        os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
        
                    # Save the history of stiffness values to a file
        np.save(filename, np.array(self.stiffness_history))

    def save_distances(self, filename):
        # Construct a clean path
        filename = os.path.normpath(filename)
        directory = os.path.dirname(filename)
        os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
        
        np.save(filename, self.distances)

    def load_distances(self, filename):
        self.distances = np.load(filename).tolist()

    def get_position(self):
    # Assuming the first element of qpos represents the x-position
        return self.data.qpos[0]




class DynamicLegEnv_base2(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 40,
    }

    def __init__(self, xml_file='leg.xml', render_mode='none', seed=None, 
                 stiffness_start=1000, stiffness_end=10000, num_timesteps=100_000, max_episode_steps = 500):
        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5, render_mode=render_mode, observation_space=observation_space)

        if seed is not None:
            self.seed(seed)

        # Initialize stiffness parameters
        self._reset_noise_scale = 1e-8
        self.total_episodes = num_timesteps/max_episode_steps

        self.stiffness_start = 1000
        self.stiffness_end = 10_000
        self.num_timesteps = num_timesteps
        self.episode_count = 0

        #self.stiffness_increment = (self.stiffness_end - self.stiffness_start) / (self.num_timesteps / self.frame_skip)
        self.stiffness_increment = (self.stiffness_end - self.stiffness_start) / (self.total_episodes)

        self.stiffness_scaling = self.stiffness_start
        self.stiffness_history = []
        self.distances = []  # List to store distances for each episode

    def apply_stiffness(self, stiffness_value):
        for i in range(len(self.model.tendon_stiffness)):
            self.model.tendon_stiffness[i] = stiffness_value
        print(f"Applied stiffness: {self.model.tendon_stiffness[0] }")

    def step(self, action):
    
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        # Calculate reward
        ctrl_cost = 0
        forward_reward = x_velocity
        reward = forward_reward - ctrl_cost

        # Prepare observation and check if done
        observation = self.get_obs()
        done = self.steps_from_reset > 500
        self.steps_from_reset += 1




        if self.render_mode == "human":
            self.render()

        return observation, reward, done, False, {}

    def get_obs(self):
        position_tmp = self.data.qpos.flat.copy()
        velocity_tmp = self.data.qvel.flat.copy()
        return np.concatenate((position_tmp, velocity_tmp)).ravel()

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        self.steps_from_reset = 0

        qpos = self.init_qpos + np.random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * np.random.randn(self.model.nv)
        self.set_state(qpos, qvel)

        # 1) Compute the linearly updated stiffness for this episode
        self.stiffness_scaling = self.stiffness_start + (self.episode_count * self.stiffness_increment)
        self.stiffness_scaling = np.clip(self.stiffness_scaling, self.stiffness_start, self.stiffness_end)
        self.apply_stiffness(self.stiffness_scaling)

    
        self.stiffness_history.append(self.stiffness_scaling)

        self.episode_count += 1


        return self.get_obs()

    def seed(self, seed=None):
        np.random.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def save_stiffness_history(self, filename):
        np.save(filename, np.array(self.stiffness_history))
