class HRLWrapper(object):

    def __init__():
        # TODO
        ## Load the policy \pi(a|s,g,\theta^{low}) you trained from Q4.
        ## Make sure the policy you load was trained with the same goal_frequency
        pass
    
    def reset():
        # TODO
        ## 
        pass

    def step(action):
        ## Add code to compute a new goal-conditioned reward
        # TODO
        sub_goal = action # The high level policy action \pi(g|s,\theta^{hi}) is the low level goal.
        for range(goal_frequency):
            ## Get the action to apply in the environment
            ## HINT you need to use \pi(a|s,g,\theta^{low})
            ## Step the environment
            pass ## Remove this
        
        # return s_{t+k}, r_{t+k}, done, info
        pass
        
    def createState():
        ## Add the goal to the state
        # TODO
        pass