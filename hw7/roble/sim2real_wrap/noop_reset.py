from typing import Tuple

import numpy as np
from gymnasium import Wrapper

class RandomActResetEnv(Wrapper):
    def __init__(self, env, max_num_random_act=4):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super().__init__(env)
        self.max_num_random_act=max_num_random_act
        self.reset_count = 0

    def _sample_num_repeat(self):
        return int(np.random.randint(0, self.max_num_random_act))

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        obs, ORIG_INFO = super().reset(**kwargs)

        num_random_act = self._sample_num_repeat()
        # TODO: sample random action and step, possibly resetting env in the process
        for _ in range(num_random_act):
            action = self.action_space.sample()
            obs, _, done, _,  _ = super().step(action)
            if done: # recursively
                self.reset_counter += 1
                if self.reset_counter > 10:  # Limit to 10 recursive calls
                    raise RecursionError("Exceeded maximum number of recursive resets (10 times)")
                obs, ORIG_INFO = self.reset(**kwargs)

        return obs, ORIG_INFO

    def step(self, ac):
        return super().step(ac)
