import numpy as np
import time

############################################
############################################

def sample_trajectory(env, policy, max_path_length, render=False, render_mode=('rgb_array')):
    # initialize env for the beginning of a new rollout
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs, infos = [], [], [], [], [], [], []
    steps = 0
    while True:
        if render:  # feel free to ignore this for now
            if 'rgb_array' in render_mode:
                if hasattr(env.unwrapped, 'sim'):
                    if 'track' in env.unwrapped.model.camera_names:
                        image_obs.append(env.unwrapped.sim.render(camera_name='track', height=500, width=500)[::-1])
                    else:
                        image_obs.append(env.unwrapped.sim.render(height=500, width=500)[::-1])
                else:
                    image_obs.append(env.render(mode=render_mode))
            if 'human' in render_mode:
                env.render(mode=render_mode)
                time.sleep(env.model.opt.timestep)
        obs.append(ob)
        # print(policy)
        ac = np.squeeze(policy.get_action(ob))
        # ac = np.expand_dims(ac, axis=0) if ac.ndim == 0 else ac
        # print(policy)
        ob, rew, done, info = env.step(ac)
        
        # if np.isnan(ob).any() or np.isnan(rew) or np.isnan(ac).any():
        #     print("Warning: NaN values detected in the simulation!")
        #     # Handle the NaN values based on your requirements
        #     # For example, you can terminate the trajectory or skip the step
        #     break
        
        # add the observation after taking a step to next_obs
        acs.append(ac)
        next_obs.append(ob)
        rewards.append(rew)
        infos.append(info)
        steps += 1
        # If the episode ended, the corresponding terminal value is 1
        # otherwise, it is 0
        if done or steps > max_path_length:
            terminals.append(1)
            break
        else:
            terminals.append(0)
    return Path(obs, image_obs, acs, rewards, next_obs, terminals, infos)

def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False, render_mode=('rgb_array')):
    """
        Collect rollouts until we have collected min_timesteps_per_batch steps.

        TODO implement this function
        Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
        Hint2: use get_pathlength to count the timesteps collected in each path
    """
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        #collect rollout
        path = sample_trajectory(env, policy, max_path_length, render, render_mode)
        paths.append(path)
        #count steps
        timesteps_this_batch += get_pathlength(path)
        # print('At timestep:    ', timesteps_this_batch, '/', min_timesteps_per_batch, end='\r')
    return paths, timesteps_this_batch

def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False, render_mode=('rgb_array')):
    """
        Collect ntraj rollouts.

        TODO implement this function
        Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
    """
    paths = []
    for _ in range(ntraj):
        # collect rollout
        path = sample_trajectory(env, policy, max_path_length, render, render_mode)
        paths.append(path)
    return paths

############################################
############################################

def Path(obs, image_obs, acs, rewards, next_obs, terminals, infos):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32),
            "infos": infos}


def convert_listofrollouts(paths, concat_rew=True):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    if concat_rew:
        rewards = np.concatenate([path["reward"] for path in paths])
    else:
        rewards = [path["reward"] for path in paths]
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    return observations, actions, rewards, next_observations, terminals

############################################
############################################

def get_pathlength(path):
    return len(path["reward"])

def flatten(matrix):
    ## Check and fix a matrix with different length lists.
    import collections.abc
    if (isinstance(matrix, (collections.abc.Sequence,))  and 
        isinstance(matrix[0], (collections.abc.Sequence, np.ndarray))): ## Flatten possible inhomogeneous arrays
        flat_list = []
        for row in matrix:
            flat_list.extend(row)
        return flat_list
    else:
        return matrix