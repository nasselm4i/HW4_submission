import gymnasium
import numpy as np
from gymnasium import Wrapper # FIX DONE

class LastActionWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

        low, high = env.observation_space.low, env.observation_space.high 
        low = np.concatenate((low, env.action_space.low)) 
        high = np.concatenate((high, env.action_space.high)) 

        self.observation_space = gymnasium.spaces.Box(low=low, high=high) 

    def _make_observation(self, obs, last_act):
        # TODO: concatenate obs and last_act
        return np.concatenate((obs, last_act))


    def reset(self, **kwargs):
        obs, ORIG_INFO = super().reset(**kwargs) 
        zero_action = np.zeros(self.action_space.shape)
        return self._make_observation(obs, zero_action), ORIG_INFO

    def step(self, action):
        obs, rew, done, trunc, info = super().step(action)
        return self._make_observation(obs, action), rew, done, trunc, info
