
import numpy as np 
from gym.spaces import Box

class GoalConditionedEnv(object):

    def __init__():
        # TODO
        self._env = base_env 
    
    def reset():
        # Add code to generate a goal from a distribution
        # TODO
        pass

    def step():
        ## Add code to compute a new goal-conditioned reward
        # TODO
        pass
        
    def createState():
        ## Add the goal to the state
        # TODO
        pass

    @property
    def action_space(self):
        return self._env.action_space
    @property
    def observation_space(self):
        return self._observation_space
    @property
    def metadata(self):
        return self._env.metadata
        
class GoalConditionedEnvV2(object):

    def __init__():
        # TODO
        self._env = base_env
    
    def reset():
        # Add code to generate a goal from a distribution
        # TODO
        pass

    def step():
        ## Add code to compute a new goal-conditioned reward
        # TODO
        pass
        
    def createState():
        ## Add the goal to the state
        # TODO
        pass

    @property
    def action_space(self):
        return self._env.action_space
    @property
    def observation_space(self):
        return self._observation_space
    @property
    def metadata(self):
        return self._env.metadata
