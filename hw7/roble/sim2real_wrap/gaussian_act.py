import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.core import ObsType, WrapperObsType, ActionWrapper, WrapperActType, ActType


class GaussianActWrapper(ActionWrapper):
    def __init__(self, env, scale):
        super().__init__(env)
        self.scale = scale

    def action(self, action: WrapperActType) -> ActType:
        noisy_action = self._noise_act(action)
        return np.clip(noisy_action, self.action_space.low, self.action_space.high)

    def _noise_act(self, act):
        # TODO: add noise
        # DONE
        act = np.random.normal(loc=act, 
                               scale=abs(act*self.scale), 
                               size=act.shape)
        return act

