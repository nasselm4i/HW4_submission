from .ddpg_critic import DDPGCritic
import torch
import torch.optim as optim
from torch.nn import utils
from torch import nn
import copy
import numpy as np

from hw1.roble.infrastructure import pytorch_util as ptu
from hw1.roble.infrastructure import utils as utilss
from hw3.roble.policies.MLP_policy import ConcatMLP


class TD3Critic(DDPGCritic):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, actor, **kwargs):
        super().__init__(actor, **kwargs)
        self.gamma = kwargs['gamma']
        
    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        """
            Update the parameters of the critic.
            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories
            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                ac_na: length: sum_of_path_lengths. The action taken at the current step.
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                reward_n: length: sum_of_path_lengths. Each element in reward_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end
            returns:
                nothing
        """
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(reward_n)
        terminal_n = ptu.from_numpy(terminal_n)

        qa_t_values = self._q_net(ob_no, ac_na)
        
        # TODO compute the Q-values from the target network 
        ## Hint: you will need to use the target policy
        qa_tp1_values = self._q_net_target(next_ob_no , self._actor_target(next_ob_no))

        # TODO compute targets for minimizing Bellman error
        # HINT: as you saw in lecture, this would be:
            #currentReward + self.gamma * qValuesOfNextTimestep * (not terminal)
        terminal_n = ~terminal_n.bool()
        
        terminal_n = terminal_n.squeeze(-1)
        qa_tp1_values = qa_tp1_values.squeeze(-1)
        reward_n = reward_n.squeeze(-1) 
        
        target = reward_n + (self.gamma * qa_tp1_values * terminal_n)
        
        target = target.unsqueeze(-1)
        target = target.detach()
        
        assert qa_t_values.shape == target.shape
        
        loss = self._loss(qa_t_values, target)

        self._optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self._q_net.parameters(), self._grad_norm_clipping)
        self._optimizer.step()
        # self.learning_rate_scheduler.step()
        return {
            "Training Loss": ptu.to_numpy(loss),
            "Q Predictions": np.mean(ptu.to_numpy(qa_t_values)),
            "Q Targets": np.mean(ptu.to_numpy(target)),
            # "Policy Actions": utilss.flatten(ptu.to_numpy(ac_na)),
            # "Actor Actions": utilss.flatten(ptu.to_numpy(self._actor(ob_no)))
        }

    def update_target_network(self):
        pass

