import numpy as np

from hw3.roble.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer
from hw3.roble.policies.MLP_policy import MLPPolicyDeterministic
from hw3.roble.critics.ddpg_critic import DDPGCritic
import copy
import torch

class GaussianNoise:
    def __init__(self, size, std_dev=0.4, decay_rate=0.995, min_std_dev=0.05):
        self.size = size
        self.std_dev = std_dev
        self.decay_rate = decay_rate
        self.min_std_dev = min_std_dev

    def sample(self):
        return np.random.normal(0, self.std_dev, self.size)

    def decay(self):
        self.std_dev = max(self.min_std_dev, self.std_dev * self.decay_rate)


class OrnsteinUhlenbeckNoise:
    "Parameters from https://arxiv.org/pdf/1509.02971.pdf"
    def __init__(self, size, mu=0., theta=0.15, sigma=0.1):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return self.state

class DDPGAgent(object):
    
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, env, **kwargs):

        self._last_obs = self._env.reset()
        self._cumulated_rewards = 0
        self._rewards = []
        # print("---- INFO ACTION SPACE ----")
        # print(self._env.action_space.n)
        # print(self._env.action_space.shape)
        action_space_shape = self._env.action_space.shape
        if len(action_space_shape) == 0:
            # Handle or raise an error for unexpected action space configurations
            raise ValueError("Action space must have a defined shape for continuous actions.")
        else:
            self._num_actions = action_space_shape[0]
        
        self.ou_noise = OrnsteinUhlenbeckNoise(size=self._num_actions)
        self.gaussian_noise = GaussianNoise(size=self._num_actions)
        # self.noise_type = "Correlated"
        self.noise_type = ""
        self.step_ = 0 # Adding step for printing the reward every X times

        self._replay_buffer_idx = None
        
        self._actor = MLPPolicyDeterministic(
            **kwargs
        )
        ## Create the Q function
        # self._agent_params['optimizer_spec'] = self._optimizer_spec
        self._q_fun = DDPGCritic(self._actor, **kwargs)

        ## Hint: We can use the Memory optimized replay buffer but now we have continuous actions
        self._replay_buffer = MemoryOptimizedReplayBuffer(
            self._replay_buffer_size, self._frame_history_len, lander=True,
            continuous_actions=True, ac_dim=self._ac_dim)
        self._t = 0
        self._num_param_updates = 0
        self._step_counter = 0
        
    def add_to_replay_buffer(self, paths):
        pass

    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self._last_obs must always point to the new latest observation.
        """        
        # TODO store the latest observation ("frame") into the replay buffer
        # HINT: the replay buffer used here is `MemoryOptimizedReplayBuffer`
            # in dqn_utils.py
        # Modified
        # self._replay_buffer_idx = -1
        self._replay_buffer_idx = self._replay_buffer.store_frame(self._last_obs)
        
 
        # TODO add noise to the deterministic policy
        # perform_random_action = TODO
        # HINT: take random action 
        
        if self.noise_type == "Correlated":
            noise_sample = self.ou_noise.sample() # Temporal Correlated Noise (Ornstein-Uhlenbeck Process)
        else:
            noise_sample = self.gaussian_noise.sample()

        action = self._actor.get_action(self._last_obs)

        # if not isinstance(noise_sample, torch.Tensor):
        #     noise_sample = torch.tensor(noise_sample, dtype=torch.float32, device=action.device)
        action = action + noise_sample
        
        action = np.clip(action, self._env.action_space.low, self._env.action_space.high) 
        
        # TODO take a step in the environment using the action from the policy
        # HINT1: remember that self._last_obs must always point to the newest/latest observation
        # HINT2: remember the following useful function that you've seen before:
            #obs, reward, done, info = env.step(action)
        obs, reward, done, info = self._env.step(action)
        # if self.step_ % 100 == 0:
        #     print("#################")
        #     print(f"Reward (from the environment directly): {reward} at iteration {self.step_}")
        #     print("#################")
        self.step_ +=1
        # TODO store the result of taking this action into the replay buffer
        # HINT1: see your replay buffer's `store_effect` function
        # HINT2: one of the arguments you'll need to pass in is self._replay_buffer_idx from above
        self._replay_buffer.store_effect(self._replay_buffer_idx, action, -reward, done)
        # self.gaussian_noise.decay()

        # TODO if taking this step resulted in done, reset the env (and the latest observation)
        if done:
            self._last_obs = self._env.reset()
            self.ou_noise.reset()


    def get_replay_buffer(self):
        return self._replay_buffer
        
    def sample(self, batch_size):
        if self._replay_buffer.can_sample(self._train_batch_size):
            return self._replay_buffer.sample(batch_size)
        else:
            # print("Need more experience in the replay buffer")
            return [],[],[],[],[]

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        logs = {}
        if (self._t > self._learning_starts
                and self._t % self._learning_freq == 0 # Update frequency of the actor and critic in our case its 1
                and self._replay_buffer.can_sample(self._train_batch_size)
        ):
            # TODO fill in the call to the update function using the appropriate tensors
            log = self._q_fun.update(
                ob_no, ac_na, next_ob_no, re_n, terminal_n
            )
            logs.update(log)
            # print(" ------ LOG CRITIC -----")
            # print(log)
            # TODO fill in the call to the update function using the appropriate tensors
            ## Hint the actor will need a copy of the q_net to maximize the Q-function
            log = self._actor.update(
                next_ob_no,
                self._q_fun
            )
            logs.update(log)
            # print(" ------ LOG ACTOR -----")
            # print(log)
            # TODO update the target network periodically 
            # HINT: your critic already has this functionality implemented
            if self._num_param_updates % self._target_update_freq == 0:
                self._q_fun.update_target_network()
            self._num_param_updates += 1
        self._t += 1
        return logs
