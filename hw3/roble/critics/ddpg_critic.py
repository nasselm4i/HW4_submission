from .base_critic import BaseCritic
import torch
import torch.optim as optim
from torch.nn import utils
from torch import nn
import copy
import numpy as np

from hw1.roble.infrastructure import pytorch_util as ptu
from hw1.roble.infrastructure import utils as utilss
from hw3.roble.policies.MLP_policy import ConcatMLP


class DDPGCritic(BaseCritic):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, actor, **kwargs):
        super().__init__()
        # self._env_name = agent_params['env']['env_name']
        self._learning_rate = self._critic_learning_rate

        if isinstance(self._ob_dim, int):
            self._input_shape = (self._ob_dim,)
        else:
            self._input_shape = kwargs["input_shape"]

        out_size = 1

        kwargs = copy.deepcopy(kwargs)
        kwargs['ob_dim'] = kwargs['ob_dim'] + kwargs['ac_dim']
        kwargs['ac_dim'] = 1
        kwargs['deterministic'] = True
        self._q_net = ConcatMLP(   
                **kwargs
            )
        self._q_net_target = ConcatMLP(   
                **kwargs
            )
        # self._learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
        #     self._optimizer,
        #     self._optimizer_spec.learning_rate_schedule,
        # 
        self._optimizer = optim.Adam(
            self._q_net.parameters(),
            self._learning_rate,
            )
        self._loss = nn.SmoothL1Loss()  # AKA Huber loss
        self._q_net.to(ptu.device)
        self._q_net_target.to(ptu.device)
        self._actor = actor
        self._actor_target = copy.deepcopy(actor) 

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
        
        ### Hint: 
        # qa_t_values = self._q_net(ob_no, ac_na)
        q_t_values = self._q_net(ob_no, ac_na).squeeze(-1)
        
        # TODO compute the Q-values from the target network 
        ## Hint: you will need to use the target policy
        q_tp1_values = self._q_net_target(next_ob_no, self._actor_target(next_ob_no)).squeeze(-1)

        # TODO compute targets for minimizing Bellman error
        # HINT: as you saw in lecture, this would be:
            #currentReward + self._gamma * qValuesOfNextTimestep * (not terminal)
        # print("---------")
        # print("terminal_n")
        # print(terminal_n.shape)
        # print("---------")
        # print("q_t_values")
        # print(q_t_values.shape)
        # print("---------")
        # print("q_tp1_values")
        # print(q_tp1_values.shape)
        # print("---------")
        # print("reward_n")
        # print(reward_n.shape)
        # print("---------")
        
        if self._step_ % 100 == 0:
            print("#################")
            print(f"Reward (from batch): {reward_n} at iteration {self._step_}")
            print("#################")
        self._step_ +=1

        target = -reward_n + self._gamma * q_tp1_values * (1 - terminal_n)
        # print("target")
        # print(target.shape)
        target = target.detach()
        
        assert q_t_values.shape == target.shape
        loss = self._loss(q_t_values, target) # Loss of the Critic

        self._optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self._q_net.parameters(), self._grad_norm_clipping)
        self._optimizer.step()
        #self._learning_rate_scheduler.step()
        # return {
        #     "(Critic) Training Loss": ptu.to_numpy(loss).item(),
        #     "Q Predictions": np.mean(ptu.to_numpy(q_t_values)),
        #     "Q Targets": np.mean(ptu.to_numpy(target)),
        #     # "Policy Actions": utilss.flatten(ptu.to_numpy(ac_na)),
        #     # "Actor Actions": utilss.flatten(ptu.to_numpy(self._actor(ob_no)))}


    def update_target_network(self):
        for target_param, param in zip(
                self._q_net_target.parameters(), self._q_net.parameters()
        ):
            ## Perform Polyak averaging
            target_param.data.copy_(self._polyak_avg * param.data + (1.0 - self._polyak_avg) * target_param.data)
        for target_param, param in zip(
                self._actor_target.parameters(), self._actor.parameters()
        ):
            ## Perform Polyak averaging for the target policy
            target_param.data.copy_(self._polyak_avg * param.data + (1.0 - self._polyak_avg) * target_param.data)

    def qa_values(self, obs):
        obs = ptu.from_numpy(obs)
        ## HINT: the q function take two arguments  
        qa_values = self._q_net(obs, self._actor(obs)).squeeze(-1)
        return ptu.to_numpy(qa_values)
