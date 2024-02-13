import numpy as np
from gymnasium import Wrapper


class ActionRepeatWrapper(Wrapper):
    def __init__(self, env, max_repeat):
        super().__init__(env)
        self.max_repeat= max_repeat

        assert max_repeat >= 1

    def _sample_num_repeat(self):
        return int(np.random.randint(1, self.max_repeat))

    def step(self, action):
        # TODO: repeat action a random number of times
        latest_reward = 0
        last_obs = None
        done = False
        trunc = False
        info = {}
        num_repeat_action = self._sample_num_repeat()

        for _ in range(num_repeat_action):
            last_obs, rew, done, trunc, info = super().step(action)
            latest_reward = rew
            if done or trunc:
                break

        return last_obs, latest_reward, done, trunc, info
