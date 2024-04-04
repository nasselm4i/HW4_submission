import sys
import os
sys.path.append(os.path.abspath('/teamspace/studios/this_studio/robot_learning'))
from rich.traceback import install
import numpy as np
from hw4.roble.envs.roboverse.widowx import create_widow_env
from hw4.roble.envs.reacher.reacher_env import create_reacher_env
from hw4.roble.infrastructure.hrl_wrapper import HRLWrapper

from hw4.roble.agents.pg_agent import PGAgent

low_level_policy=PGAgent

env_config = {
    "env": {
        "env_name": "reacher",  # ['reacher', 'antmaze', 'widowx']
        "task_name": "hrl",  # ['gcrl','gcrl_v2', 'hrl']
        "distribution": "normal",  # ["uniform", "normal"]
        "relative_goal": False,
        "max_episode_length": 500,
        "exp_name": "debug",
        "atari": False,
        "reacher": {
            "hand_indices": [3, 4, 5],
            "goal_indices": [-3, -2, -1],
            "uniform_bounds": [[-0.6, -1.4, -0.4], [0.8, 0.2, 0.5]],
            "gaussian_bounds": [[0.2, 0.7, 0.0], [0.3, 0.4, 0.05]],
            "global_goal": [0.1, 0.1, 0.1],
            "k": 10,
            "goal_reached_threshold": -0.3
        },
        "widowx": {
            "hand_indices": [-6, -5, -4],
            "goal_indices": [0, 1, 2],
            "uniform_bounds": [[0.4, -0.2, -0.34], [0.8, 0.4, -0.1]],
            "gaussian_bounds": [[0.6, 0.1, -0.2], [0.2, 0.2, 0.2]],
            "global_goal": [0.6, 0.2, -0.3],
            "k": 10,
            "goal_reached_threshold": -0.3
        },
        "low_level_agent_path": "/teamspace/studios/this_studio/robot_learning/data/hw4_q4_reacher_normal_k10_reacher_31-03-2024_20-30-32/agent_itr_6000.pt"
    },
    "alg": {
        "double_q": True,
        "batch_size": 4096,
        "train_batch_size": 4096,
        "eval_batch_size": 4096,
        "num_agent_train_steps_per_iter": 1,
        "num_critic_updates_per_agent_update": 16,
        "use_gpu": False,
        "gpu_id": 0,
        "rl_alg": "pg",
        "learning_starts": 1,
        "learning_freq": 1,
        "target_update_freq": 1,
        "exploration_schedule": 0,
        "optimizer_spec": 0,
        "replay_buffer_size": 100000,
        "frame_history_len": 1,
        "gamma": 0.95,
        "critic_learning_rate": 1e-3,
        "learning_rate": 3e-4,
        "ob_dim": 0,  # do not modify
        "ac_dim": 0,  # do not modify
        "batch_size_initial": 0,  # do not modify
        "discrete": False,
        "grad_norm_clipping": True,
        "n_iter": 1000,
        "polyak_avg": 0.01,
        "td3_target_policy_noise": 0.1,
        "sac_entropy_coeff": 0.2,
        "policy_std": 0.05,
        "use_baseline": True,
        "gae_lambda": 0.9,
        "standardize_advantages": True,
        "reward_to_go": False,
        "nn_baseline": True,
        "on_policy": True,
        "learn_policy_std": False,
        "deterministic": False,
        "network": {
            "layer_sizes": [64, 32],
            "activations": ["leaky_relu", "leaky_relu"],
            "output_activation": "identity"
        },
        "logging": {
        "video_log_freq": 100,  # How often to generate a video to log/
        "scalar_log_freq": 10,  # How often to log training information and run evaluation during training.
        "save_frequency": 100,  # Frequency of agent's params saved
        "save_params": False,  # Should the parameters given to the script be saved? (Always...)
        "random_seed": 1234,
        "logdir": "",
        "debug": False
    }
    }
}


def test_hrl():
    env_config["env"]["env_name"] = "reacher"
    env = create_reacher_env()
    
    # env_config["env"]["env_name"] = "widowx"
    # env = create_widow_env()
    
    env = HRLWrapper(env, low_level_policy=low_level_policy, **env_config)

    for _ in range(30):
        subgoal = env.action_space.sample()
        obs, reward, done, info = env.step(subgoal)
        for k in range(env_config["env"]["reacher"]["k"]):
            print("---------------------------------------")
            print(f"subgoal : {subgoal} // k : {k}")
            # print("obs", obs)
            # print("env.get_observation()", env._get_obs())
            # print("obs", obs.shape)
        # print(reward)
    

install()
test_hrl()