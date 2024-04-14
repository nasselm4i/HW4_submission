#!/home/zeus/miniconda3/envs/cloudspace/bin/python

import functools
import random
import time

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

import sys
import os

from hw7.roble import puppergym
from hw7.roble.sim2real_wrap.thunk_sim2real_wrap import make_thunk
from hw7.roble import ppo, sac

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import gin
from pybullet_envs.minitaur.envs_v2 import env_loader
import puppersim
from gymnasium import Wrapper
import gymnasium
from gymnasium.utils.step_api_compatibility import convert_to_terminated_truncated_step_api
from gymnasium.wrappers import TimeLimit

from rich.traceback import install

def make_pupper_task(seed):
    CONFIG_DIR = puppersim.getPupperSimPath()
    _CONFIG_FILE = os.path.join(CONFIG_DIR, "../puppersim/config", "pupper_pmtg.gin")
    #  _NUM_STEPS = 10000
    #  _ENV_RANDOM_SEED = 2

    import puppersim.data as pd
    gin.bind_parameter("scene_base.SceneBase.data_root", pd.getDataPath() + "/")
    gin.parse_config_file(_CONFIG_FILE)
    env = env_loader.load()
    print('type env vanilla :', type(env))
    env.seed(seed)

    class GymnasiumWrapper(Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = gymnasium.spaces.Box(low=env.observation_space.low, high=env.observation_space.high)
            self.action_space = gymnasium.spaces.Box(low=env.action_space.low, high=env.action_space.high)

        @property
        def render_mode(self):
            return "rgb_array"

        def reset(self):
            env.seed(np.random.randint(0, 20000))   # change seed

            return self.env.reset(), {}

        def step(self, action):
            return convert_to_terminated_truncated_step_api(self.env.step(action))

        def render(self, render_mode=None):
            return self.env.render(mode=self.render_mode)

    env = GymnasiumWrapper(env)
    return env


@hydra.main(config_path="conf", config_name="config_hw7")
def test_env(args: DictConfig):
    os.chdir(get_original_cwd())
    
    def get_arg_dict(args):
        dico = dict(vars(args))
        return dico["_content"]

    def flatten_conf(conf1, conf2):
        dico = get_arg_dict(conf1)
        dico.update(get_arg_dict(conf2))

        args = OmegaConf.create(dico)
        return args
    
    sim2real = args.sim2real
    new_args = flatten_conf(args.meta, OmegaConf.create({"sim2real": get_arg_dict(sim2real)}))
    
    args = flatten_conf(new_args, args.sac)
    
    
    ## Env Setup 
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    sim2real_wrap = make_thunk(args.sim2real) # setup wrappers
    make_vector_env = functools.partial(puppergym.make_vector_env, sim2real_wrap=sim2real_wrap,
                                        timelimit=args.timelimit, num_vector=args.num_envs)
    env = make_pupper_task(args.seed)
    print(args)
    env = TimeLimit(env, args.timelimit)
    obs, infos = env.reset()
    # ac = env.action_space.sample()
    # obs, rew, done, trunc, info = env.step(ac)
    if TEST_TIMELIMIT:
        print("---------------------------------------------")
        print("Test timelimit...")
        ac = env.action_space.sample()
        for k in range(2000):
            obs, rew, trunc, done, info = env.step(ac)
            if trunc or done:
                print(f"k={k} and trunc={trunc} // done={done}")
                print("------------------------")
                print(obs.shape)
                print("reward", rew)
                print("truncated", trunc)
                print("done", done)
                print("info", info)
                break
    print("---------------------------------------------")
    if TEST_NOOP_RESET:
        
        # print(type(env))
        # print("env.action_space", env.action_space.shape)
        # print("env.observation_space", env.observation_space.shape)
        print("---------------------------------------------")
        print("Test Random Action Reset Wrapper ...")
        
        from hw7.roble.sim2real_wrap.noop_reset import RandomActResetEnv
        # env = RandomActResetEnv(env)
        obs, infos = env.reset()
        print("obs", obs)
        obs_2, infos = env.reset()
        print("obs", obs_2)
        print('obs diff', obs - obs_2)
        for k in range(10000):
            ac = env.action_space.sample()
            obs, rew, trunc, done, info = env.step(ac)
            # Check for None values in all variables
            if any(value is None for value in [obs, rew, trunc, done]):
                raise ValueError("None value detected in step outputs.")

            # Since obs is a np.array, check for NaN values within it
            if np.any(np.isnan(obs)):
                raise ValueError("NaN value detected in observations (obs).")

            # Since rew is a float, directly check if it is NaN
            if np.isnan(rew):
                raise ValueError("NaN value detected in reward (rew).")
            if trunc or done:
                print(f"k={k} and trunc={trunc} // done={done}")
                print("------------------------")
                print(obs.shape)
                print("ac", ac.shape)
                print("reward", rew)
                print("truncated", trunc)
                print("done", done)
                print("info", info)
                # break
        print("Test passed")
        print("---------------------------------------------")
        
    if TEST_HISTORY  and args.sim2real.history_len > 0:
        print("---------------------------------------------")
        print("Test History Wrapper ...")
        
        from hw7.roble.sim2real_wrap.history import HistoryWrapper
        env = HistoryWrapper(env, args.sim2real.history_len)
        obs, info = env.reset()
        for _ in range(5):
            # print("obs", obs)
            ac = env.action_space.sample()
            obs, rew, done, trunc, info = env.step(ac)
        print(obs.shape)
        print("Test passed")
        print("---------------------------------------------")
        
    if TEST_LAST_ACTION:
        print("---------------------------------------------")
        print("Test Last Action Wrapper ...")
        
        from hw7.roble.sim2real_wrap.last_action import LastActionWrapper
        env = LastActionWrapper(env)
        obs, info = env.reset()
        ac = env.action_space.sample()
        obs, rew, done, trunc, info = env.step(ac)
        print("ac", ac.shape)
        print("obs", obs.shape)
        print("Test passed")
        print("---------------------------------------------")
    if TEST_GAUSSIAN_OBS:
        print("---------------------------------------------")
        print("Test Gaussian Obs Wrapper ...")
        
        from hw7.roble.sim2real_wrap.gaussian_obs import GaussianObsWrapper
        env = GaussianObsWrapper(env, args.sim2real.gaussian_obs_scale)
        # obs, info = env.reset()
        # print("obs", obs)
        print("Test passed")
        print("---------------------------------------------")
    if TEST_GAUSSIAN_ACT:
        
        print("Test Gaussian Action Wrapper ...")
        
        from hw7.roble.sim2real_wrap.gaussian_act import GaussianActWrapper
        print("args.sim2real.gaussian_act_scale", args.sim2real.gaussian_act_scale)
        env = GaussianActWrapper(env, args.sim2real.gaussian_act_scale)
        # obs, info = env.reset()
        # print("obs", obs)
        print("Test passed")
    
    if TEST_REPEAT_ACT:
        
        print("Test Action Repeat Wrapper ...")
        
        from hw7.roble.sim2real_wrap.repeat_act import ActionRepeatWrapper
        env = ActionRepeatWrapper(env, args.sim2real.action_repeat_max)
        obs, info = env.reset()
        
        for _ in range(100):
            ac = env.action_space.sample()
            obs, rew, done, trunc, info = env.step(ac)
            print("ac", ac.shape)
            print("obs", obs.shape)
            print("rew", rew)
        print("Test passed")

# TEST SETTINGS

TEST_TIMELIMIT = False

## Test Wrappers
TEST_NOOP_RESET = True
TEST_HISTORY = False
TEST_LAST_ACTION = False
TEST_GAUSSIAN_OBS = False
TEST_GAUSSIAN_ACT = False
TEST_REPEAT_ACT = False
        
    
    

if __name__ == "__main__":
    install()
    test_env()