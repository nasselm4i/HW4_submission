import collections

import gymnasium
import numpy as np
from gymnasium import Wrapper

class HistoryWrapper(Wrapper):
    def __init__(self, env, length):
        super().__init__(env)
        self.length = length

        low, high = env.observation_space.low, env.observation_space.high
        low = np.array([[low] * length]).squeeze().flatten()
        high = np.array([[high] * length]).squeeze().flatten()

        self.observation_space = gymnasium.spaces.Box(low=low,
                                                      high=high)
        self._reset_buf()

    def _reset_buf(self):
        # TODO: reset history buffer
        self.buffer = collections.deque(maxlen=self.length)
        

    def _make_observation(self):

        # TODO: concatenate history into obs
        return np.concatenate(list(self.buffer))


    def reset(self, **kwargs): # Used only for the seed
        self._reset_buf()
        obs, ORIG_INFO = super().reset(**kwargs)
        
        # TODO: add first state to history buffer
        for _ in range(self.length):
            self.buffer.append(obs)
            
        obs = self._make_observation()
        return obs, ORIG_INFO


    def step(self, action):
        obs, rew, done, trunc, info = super().step(action) # FIX DONE
        
        # TODO: add state to history buffer
        
        self.buffer.append(obs)
        obs = self._make_observation()
        return obs, rew, done, trunc, info
