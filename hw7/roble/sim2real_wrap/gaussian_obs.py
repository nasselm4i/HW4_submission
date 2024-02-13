import numpy as np
from gymnasium import ObservationWrapper # Potentially to replace by gym
from gymnasium.core import ObsType, WrapperObsType


class GaussianObsWrapper(ObservationWrapper):
    def __init__(self, env, scale):
        super().__init__(env)
        self.scale = scale
        
    def observation(self, observation: ObsType) -> WrapperObsType:
        return self._noise_obs(observation)

    def _noise_obs(self, obs):
        # TODO: add noise
        # DONE
        obs = np.random.normal(obs, abs(obs*self.scale), size=obs.shape)
        return obs
