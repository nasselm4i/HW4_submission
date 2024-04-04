import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from hw1.roble.infrastructure import pytorch_util as ptu
from hw1.roble.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self,
                 *args,
                 **kwargs
                 ):
        super().__init__()

        if self._discrete:
            self._logits_na = ptu.build_mlp(input_size=self._ob_dim,
                                           output_size=self._ac_dim,
                                           params=self._network)
            self._logits_na.to(ptu.device)
            self._mean_net = None
            self._logstd = None
            self._optimizer = optim.Adam(self._logits_na.parameters(),
                                        self._learning_rate)
        else:
            self._logits_na = None
            self._mean_net = ptu.build_mlp(input_size=self._ob_dim,
                                      output_size=self._ac_dim,
                                      params=self._network)
            self._mean_net.to(ptu.device)

            if self._deterministic:
                self._optimizer = optim.Adam(
                    itertools.chain(self._mean_net.parameters()),
                    self._learning_rate
                )
            else:
                self._std = nn.Parameter(
                    torch.ones(self._ac_dim, dtype=torch.float32, device=ptu.device) * 0.1
                )
                self._std.to(ptu.device)
                if self._learn_policy_std:
                    self._optimizer = optim.Adam(
                        itertools.chain([self._std], self._mean_net.parameters()),
                        self._learning_rate
                    )
                else:
                    self._optimizer = optim.Adam(
                        itertools.chain(self._mean_net.parameters()),
                        self._learning_rate
                    )

        if self._nn_baseline:
            self._baseline = ptu.build_mlp(
                input_size=self._ob_dim,
                output_size=1,
                params=self._network
            )
            self._baseline.to(ptu.device)
            self._baseline_optimizer = optim.Adam(
                self._baseline.parameters(),
                self._critic_learning_rate,
            )
        else:
            self._baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # TODO: 
        ## Provide the logic to produce an action from the policy
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        observation = ptu.from_numpy(observation)
        action = self(observation)
        if not self._deterministic :
            action = action.sample()  # don't bother with rsample
        return ptu.to_numpy(action)

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # observation = ptu.from_numpy(observation)
        if self._discrete:
            logits = self._logits_na(observation)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        else:
            if not isinstance(observation, torch.Tensor):
                observation = ptu.from_numpy(observation)
            # observation_tensor = torch.from_numpy(observation).float().to(ptu.device)
            # print("Observation: ", type(observation))
            if self._deterministic:
                ## TODO :  output for a deterministic policy
                action = self._mean_net(observation)
                return action
                # b_mean = self._mean_net(observation)
                # # Create a scale_tril matrix with an extremely small standard deviation
                # # to approximate deterministic behavior
                # scale_tril = torch.diag(torch.exp(torch.full_like(self._logstd, -20)))  # -20 or other large negative value for near-zero std
                # b_scale_tril = scale_tril.repeat(b_mean.shape[0], 1, 1)
                # action_distribution = distributions.MultivariateNormal(
                #     b_mean,
                #     scale_tril=b_scale_tril,             
                # )
                # action_mean = self._mean_net(observation)
                # eps = 1e-11 # near-zero variance
                # action_variance = torch.full_like(action_mean, eps)
                # action_distribution = distributions.Normal(action_mean, action_variance)
                # return action_distribution
            else:
                ## DONE output for a stochastic policy
                # action_distribution = self._mean_net(observation)
                b_mean = self._mean_net(observation)
                scale_tril = torch.diag(torch.exp(self._std))
                b_scale_tril = scale_tril.repeat(b_mean.shape[0], 1, 1)
                action_distribution = distributions.MultivariateNormal(
                    b_mean,
                    scale_tril=b_scale_tril,
                )
                
        return action_distribution
    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        # pass
        raise NotImplementedError

#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._loss = nn.MSELoss()

    def update(
        self, observations, actions,
        adv_n=None, acs_labels_na=None, qvals=None
        ):
        
        # TODO: update the policy and return the loss
        loss = TODO
        return {
            'Training Loss': ptu.to_numpy(loss),
        }

    def update_idm(
        self, observations, actions, next_observations,
        adv_n=None, acs_labels_na=None, qvals=None
        ):
        
        
        # TODO: Create the full input to the IDM model (hint: it's not the same as the actor as it takes both obs and next_obs)
        
        # TODO: Transform the numpy arrays to torch tensors (for obs, next_obs and actions)
        
        # TODO: Create the full input to the IDM model (hint: it's not the same as the actor as it takes both obs and next_obs)
        
        # TODO: Get the predicted actions from the IDM model (hint: you need to call the forward function of the IDM model)
        
        # TODO: Compute the loss using the MLP_policy loss function
        
        # TODO: Update the IDM model.
        loss = TODO
        return {
            'Training Loss IDM': ptu.to_numpy(loss),
        }