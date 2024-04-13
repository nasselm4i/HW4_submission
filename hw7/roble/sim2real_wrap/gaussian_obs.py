import numpy as np
from gymnasium import ObservationWrapper # Potentially to replace by gym
from gymnasium.core import ObsType, WrapperObsType


class GaussianObsWrapper(ObservationWrapper):
    def __init__(self, env, scale):
        super().__init__(env)
        self.scale = scale
        
    def observation(self, observation: ObsType) -> WrapperObsType:
        noisy_obs = self._noise_obs(observation)
        return  np.clip(noisy_obs, self.observation_space.low, self.observation_space.high)

    def _noise_obs(self, obs):
        # TODO: add noise
        # DONE
        obs = np.random.normal(obs, scale=self.scale, size=obs.shape)
        return obs
