import numpy as np
from hw4.roble.infrastructure.gclr_wrapper import GoalConditionedEnv
from gym.spaces import Box, Discrete
import torch

LOW_GOAL_REACHER = [0, -0.6, 0]
HIGH_GOAL_REACHER = [0.3, 0.1, 0.5]
LOW_GOAL_WIDOWX = [.5, .18, -.30]
HIGH_GOAL_WIDOWX = [.7, .27, -.30]

class HRLWrapper(GoalConditionedEnv):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, base_env, low_level_policy=None, high_level_policy=None, **kwargs):
        super().__init__(base_env, **kwargs["env"])
        self._params = kwargs
        # TODO
        ## Load the policy \pi(a|s,g,\theta^{low}) you trained from Q4.
        ## Make sure the policy you load was trained with the same goal_frequency
        self.reset()
        
        self.goal_frequency = self._params["env"]["k"]
        combined_params = self.set_dict_params(self._params["env"])
        
        # Import Pretrained low level Agent
        print("Setting up Low Level Policy ...")
        self.low_level_policy = low_level_policy(self._env, **combined_params)._actor
        self.low_level_policy.load_state_dict(torch.load(self._params["env"]["low_level_agent_path"]))
        # Inference mode
        self.low_level_policy.eval()
        
    def set_dict_params(self, dict_params):
        # Is this env continuous, or self._discrete?
        self._params['alg']['discrete'] = isinstance(self._env.action_space, Discrete)

        # Observation and action sizes
        ob_dim = self._env.observation_space.shape[0]
        ac_dim = self._env.action_space.n if self._params['alg']['discrete'] else self._env.action_space.shape[0]
        self._params['alg']['ac_dim'] = ac_dim
        self._params['alg']['ob_dim'] = ob_dim
        
        combined_params = dict(self._params['alg'].copy())
        combined_params.update(dict_params)
        
        return combined_params
        
        
    def reset(self):
        obs = self.remove_goal_from_state(self.unwrapped.reset())
        self.reset_goal()
        return obs # remove goal indices
    
    def reset_goal(self):
        self.set_goal(self.global_goal)
    
    def remove_goal_from_state(self, obs):
        return np.delete(obs, self.goal_indices)
        
    def step(self, subgoal):
        """
    Take a subgoal from the high-level policy and interact with the environment for a fixed number of steps (goal_frequency) using the low-level policy conditioned on the subgoal. Return the final observation, reward, done flag, and info after the interactions.

    Args:
        subgoal (np.array): The subgoal generated by the high-level policy, with shape [3, 1] corresponding to (x, y, z).

    Returns:
        obs (np.array): The final observation after interacting with the environment for goal_frequency steps, with shape [dim_obs, 1].
        reward (float): The reward obtained after the interactions.
        done (bool): A flag indicating whether the episode is done or not.
        info (dict): Additional information about the environment.

    This function serves as the main interface between the high-level and low-level policies in a hierarchical reinforcement learning setup. It takes a subgoal from the high-level policy and uses the low-level policy to interact with the environment for a fixed number of steps (goal_frequency), optimizing for the given subgoal. After the interactions, it returns the final observation, reward, done flag, and additional information.
    """
        obs = self.get_obs()
        
        # The high level policy action \pi(g|s,\theta^{hi}) is the low level goal.
        for _ in range(self.goal_frequency):
            obs = self.create_state(obs, subgoal)
            ## Get the action to apply in the environment
            ## HINT you need to use \pi(a|s,g,\theta^{low})
            action = self.low_level_policy.get_action(obs)
            if self.env_name == "widowx": # This is a hacky way to resolve action shape mismatch for widowx action
                action = action.squeeze()
            ## Step the environment
            obs, _, done, info = super().step(action)
            if done:
                break
        
        reward = self.reward(obs)
        done = self.success_fn(reward)
        if done :
            self.reset()
        info = self.set_info(info, obs, done)
        
        obs = self.remove_goal_from_state(obs)
        # return s_{t+k}, r_{t+k}, done, info
        return obs, reward, done, info

    @property
    def observation_space(self):
        # Get the original observation space from base_env
        original_space = self._env.observation_space

        # Create the new observation space with adjusted shape
        # Use the original space's bounds, excluding the last three dimensions
        
        new_low = np.delete(original_space.low, self.goal_indices)
        new_high = np.delete(original_space.high, self.goal_indices)

        # Return the new observation space
        return Box(low=new_low, high=new_high, dtype=np.float32)
    
    @property
    def action_space(self):
        if self.env_name == "reacher":
            low = np.array(LOW_GOAL_REACHER)
            high = np.array(HIGH_GOAL_REACHER)
        elif self.env_name == "widowx":
            low = np.array(LOW_GOAL_WIDOWX)
            high = np.array(HIGH_GOAL_WIDOWX)
        return Box(low=low, high=high, dtype=np.float32)
    
"""
TODO LIST :

[X] Changing the action_space and observation space for widowx, include only the state and object_position
[X] Changing Video Quality // View Camera for WidowX
[] HER Implementation

"""