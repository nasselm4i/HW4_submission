import pytest
from hw7.roble import puppergym
from hw7.roble.sim2real_wrap.thunk_sim2real_wrap import make_thunk
from hw7.roble.sim2real_wrap.noop_reset import RandomActResetEnv
from hw7.roble.sim2real_wrap.history import HistoryWrapper
from hw7.roble.sim2real_wrap.last_action import LastActionWrapper
from hw7.roble.sim2real_wrap.gaussian_obs import GaussianObsWrapper
from hw7.roble.sim2real_wrap.gaussian_act import GaussianActWrapper
from hw7.roble.sim2real_wrap.repeat_act import ActionRepeatWrapper
import sys
import os

sys.path.append(os.path.abspath('/teamspace/studios/this_studio/robot_learning'))


# Assuming make_pupper_task is defined elsewhere as in your script
from your_script_location import make_pupper_task

@pytest.fixture
def env_args():
    # Assuming args is a dictionary of your environment arguments
    args = {
        "seed": 123,
        "timelimit": 1000,
        # Add other necessary arguments here
    }
    return args

@pytest.fixture
def pupper_env(env_args):
    env = make_pupper_task(env_args['seed'])
    return env

def test_noop_reset(pupper_env):
    wrapped_env = RandomActResetEnv(pupper_env)
    assert wrapped_env is not None
    # Perform specific tests here

def test_history_wrapper(pupper_env, env_args):
    wrapped_env = HistoryWrapper(pupper_env, env_args['history_len'])
    assert wrapped_env is not None
    # Perform specific tests here

# Add similar test functions for the other wrappers and functionalities

if __name__ == "__main__":
    pytest.main()
