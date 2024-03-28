
import numpy as np 
from gym.spaces import Box

    
class GoalConditionedEnv(object):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, base_env, **kwargs):
        # TODO
        self._env = base_env
        self.distribution = kwargs["distribution"]
        if self.distribution == "uniform":
            self.bounds = kwargs["uniform_bounds"]
            self.random = np.random.uniform
        elif self.distribution == "normal":
            self.bounds = kwargs["gaussian_bounds"]
            self.random = np.random.normal
        else:
            raise ValueError(f"Error: Distribution '{self.distribution}' unknown.")
        self.goal_reached_threshold = kwargs["goal_reached_threshold"]
        self.param1, self.param2 = self.bounds
        self.goal_dimensions = len(self.bounds[0])
        self.reset()

    def success_fn(self, last_reward):
        # TODO
        return last_reward > self.goal_reached_threshold
    
    def reset(self):
        # Add code to generate a goal from a distribution
        # TODO
        # DONE
        obs, info = self._env.reset()
        self.goal = self.random(self.param1, self.param2, size=self.goal_dimensions)
        obs = self.create_state(obs, self.goal)
        return obs, info
    
    def step(self, a):
        ## Add code to compute a new goal-conditioned reward
        # TODO
        # DONE
        obs, reward, done, info = self._env.step(a) # adjust the reward
        obs = self.create_state(obs, self.goal)
        info["reached_goal"] = self.success_fn(reward)
        return obs, reward, done, info

    def create_state(self, obs, goal):
        ## Add the goal to the state
        # TODO
        # DONE
        return np.concatenate([obs, goal])
    
    @property
    def action_space(self):
        return self._env.action_space
    @property
    def observation_space(self):
        return self._observation_space
    @property
    def metadata(self):
        return self._env.metadata
    @property
    def unwrapped(self):
        return self._env


class GoalConditionedEnvV2(GoalConditionedEnv):

    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, base_env, **kwargs):
        super().__init__(base_env, **kwargs)
        # TODO
        super().__init__(base_env, **kwargs)
        self._env = base_env

    def reset(self):
        # Add code to generate a goal from a distribution
        # TODO
        pass
        
    def success_fn(self,last_reward):
        # TODO
        pass
        
    def reset_step_counter(self):
        # Add code to track how long the current goal has been used.
        # TODO
        pass

    def step(self, a):
        ## Add code to control the agent for a number of timesteps and 
        ## change goals after k timesteps.
        # TODO
        pass