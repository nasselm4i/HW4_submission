import numpy as np
from gym import Wrapper
from gym.wrappers import FilterObservation, FlattenObservation

import roboverse

import roboverse.bullet as bullet


def create_widow_env(env="Widow250EEPosition-v0", **kwargs):
    env = roboverse.make(env, **kwargs)
    env = FlattenObservation(
        FilterObservation(
            RoboverseWrapper(env),
            [
                "state", # np.concatenate((ee_pos, ee_quat, gripper_state, gripper_binary_state))
                # The object is located at the goal location. For
                # non goal conditioned observations, only state
                # should be used.
                "object_position",
                # "object_orientation"
            ],
        )
    )
    return env


class RoboverseWrapper(Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated or truncated, info

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)[0]
        return obs

    def render(self, mode="rgb_array", **kwargs):
        img = self.env.render_obs()
        img = np.transpose(img, (2, 1, 0))
        return img
    
    def set_target_position(self, position):
        # self.env.original_object_positions = position
        bullet.reset_object(self.unwrapped.objects["sphere"], position, self.unwrapped.get_observation()["object_orientation"])
    
    def get_object_position(self):
        return bullet.get_object_position(self.unwrapped.objects["sphere"])[0]

    def get_obs(self):
        dict_obs = self.unwrapped.get_observation()
        state = dict_obs["state"]
        object_position = dict_obs["object_position"]
        return np.concatenate((object_position, state))
    
    def get_target_dist(self, obs):
        hand_pos = obs[:3]
        target_pos = obs[-3:]
        return np.abs(hand_pos-target_pos)

    def metadata(self):
        return {"video": {
        "frames_per_second": 100 }
                }
    
    def model(self):
        return None


"""
1. **`'object_position'`**: This is the 3D position (x, y, z coordinates) of an object of interest in the environment. This information is critical for tasks that require the robot to interact with specific objects, such as grasping or moving them to a target location.

2. **`'object_orientation'`**: This represents the orientation of the object in the environment, typically expressed as a quaternion (a four-element vector). Quaternions are a compact and non-ambiguous way to represent orientations in 3D space, which is important for understanding how objects are positioned relative to the robot.

3. **`'state'`**: This is a concatenated vector that combines several pieces of information into one flat array, including:
    - **`ee_pos`**: The 3D position (x, y, z coordinates) of the robot's end-effector (the part of the robot that interacts with objects, such as a gripper or hand). This information is essential for controlling the robot and for tasks that involve precise movements.
    - **`ee_quat`**: The orientation of the robot's end-effector, expressed as a quaternion. Knowing the orientation of the end-effector is crucial for tasks that require specific poses or angles of interaction, such as inserting a key into a lock.
    - **`gripper_state`**: The state of the robot's gripper, usually expressed as the distance between the gripper's fingers or the amount of force being applied. This helps determine whether the gripper is open, closed, or holding an object.
    - **`gripper_binary_state`**: A binary indicator (usually 0 or 1) representing whether the gripper is considered open or closed. This can be a simplified view of the gripper's state for tasks where the exact distance or force is less relevant than whether the gripper is simply open or closed.

4. **`'image'`**: An image observation from the environment, typically taken from a camera attached to the robot or situated in the environment. Image observations are used in tasks that require visual perception, such as identifying objects, navigating based on visual cues, or performing tasks that require a visual understanding of the environment's state.
"""