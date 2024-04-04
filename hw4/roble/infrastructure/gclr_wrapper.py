from copy import deepcopy
from numpy.random import uniform, normal
from numpy.linalg import norm
from gym.spaces import Box

# STD_RELATIVE_GOAL = .3

    
class GoalConditionedEnv(object):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, base_env, **kwargs):
        # TODO
        # DONE
        
        # General parameters
        self._env = base_env
        self.env_name = kwargs["env_name"]
        self.distribution = kwargs["distribution"] # uniform 
        self.relative_goal = kwargs["relative_goal"] # Boolean
        self.max_episode_length = kwargs["max_episode_length"]
        
        # Parameters specific to the environment
        self.global_goal = kwargs[self.env_name]["global_goal"]
        self.goal_reached_threshold = kwargs[self.env_name]["goal_reached_threshold"] # -0.3
        self.hand_indices = kwargs[self.env_name]["hand_indices"]
        self.goal_indices = kwargs[self.env_name]["goal_indices"]

        if self.distribution == "uniform":
            self.param1, self.param2 = kwargs[self.env_name]["uniform_bounds"]
            self.random = uniform
        elif self.distribution == "normal":
            self.param1, self.param2 = kwargs[self.env_name]["gaussian_bounds"]
            self.random = normal
        else:
            raise ValueError(f"Error: Distribution '{self.distribution}' unknown.")
        if self.relative_goal:
            self.param1 = self._get_obs()[self.hand_indices]
            self.random = normal
        self.count=0
        self.reset() # reset the environment by default

    def success_fn(self, last_reward):
        # TODO
        return last_reward > self.goal_reached_threshold
    
    def random_position(self, param1, param2):
        if self.relative_goal:
            param1 = self._get_obs()[self.hand_indices]
        return self.random(param1, param2, len(param1))
    
    def set_goal(self, goal): 
        if self.env_name == "widowx":
            self.unwrapped.set_target_position(goal)
        elif self.env_name == "reacher":
            self._model.site_pos[self.unwrapped.target_sid] = goal
            self.unwrapped.sim.forward()
        else:
            raise ValueError(f"Error: Environment '{self.env_name}' unknown.")

    def reset(self): 
        # Add code to generate a goal from a distribution
        # TODO
        # DONE
        obs = self.unwrapped.reset()
        self.set_goal(self.random_position(self.param1, self.param2))
        if self.relative_goal:
            goal = self.unwrapped.get_target_dist(obs)
            obs = self.create_state(obs, goal)
        self.count = 1
        return obs

    def set_info(self, info, obs, success):
        info["success"] = success
        info["goal_relative_distance"] = self.unwrapped.get_target_dist(obs)
        return info
    
    def reward(self, obs):
        hand_pos = obs[self.hand_indices]
        if self.env_name == "widowx":
            target_pos = self.unwrapped.get_object_position()
        elif self.env_name == "reacher":
            target_pos = self._model.site_pos[self.unwrapped.target_sid]
        else:
            raise ValueError(f"Error: Environment '{self.env_name}' unknown.")
        
        reward = -1*norm(hand_pos - target_pos)
        return reward
    
    def step(self, a):
        ## Add code to compute a new goal-conditioned reward
        # TODO
        # DONE
        
        obs, _, _, info = self.unwrapped.step(a)
        
        if self.relative_goal:
            obs = self.create_state(obs, self.unwrapped.get_target_dist(obs))
        
        reward = self.reward(obs)
        
        success = self.success_fn(reward)
        truncated = (self.count % self.max_episode_length) == 0
        
        if success or truncated:
            # print("count :", self.count)
            # print(f"done:{done} // truncated : {truncated}")
            self.reset()
        self.count +=1
        info = self.set_info(info=info, obs=obs, success=success)
        return obs, reward, success, info

    def create_state(self, obs, goal):
        ## Add the goal to the state
        # TODO
        # DONE
        obs[self.goal_indices] = goal
        return obs

    @property
    def _model(self):
        return self._env.model
    @property
    def action_space(self):
        return self._env.action_space
    @property
    def observation_space(self):
        return self._env.observation_space
    @property
    def metadata(self):
        return self._env.metadata
    @property
    def unwrapped(self):
        return self._env
    @property
    def seed(self):
        return self._env.seed
    @property
    def data(self):
        return self._env.data
    @property
    def get_obs(self):
        return self._env.get_obs
    @property
    def metadata(self):
        return self._env.metadata
    @property
    def render(self):
        return self._env.render

class GoalConditionedEnvV2(GoalConditionedEnv):

    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, base_env, **kwargs):
        super().__init__(base_env, **kwargs)
        # TODO
        self.goal_frequency = kwargs["k"]
        self.count_timesteps = 0
        
    def reset_step_counter(self):
        # Add code to track how long the current goal has been used.
        # TODO
        if self.count_timesteps == self.goal_frequency:
            self.count_timesteps = 0
            self.set_goal(self.random_position(self.param1, self.param2))

    def step(self, a):
        ## Add code to control the agent for a number of timesteps and 
        ## change goals after k timesteps.
        # TODO
        
        obs, reward, done, info = super().step(a)
        
        self.count_timesteps += 1
        self.reset_step_counter()
        
        return obs, reward, done, info

        