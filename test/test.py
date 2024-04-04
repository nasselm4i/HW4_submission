
from hw4.roble.envs.reacher.reacher_env import create_reacher_env

from hw4.roble.envs.ant.create_maze_env import create_maze_env


from hw4.roble.envs.roboverse.widowx import create_widow_env
import gym
import numpy as np
from PIL import Image
import cv2

# height, width = 48, 48  # Assuming these are the dimensions of your rendered images
# video_filename = 'widowx_video.mp4'
# fourcc = cv2.VideoWriter_fourcc(*'avc1')
# video = cv2.VideoWriter(video_filename, fourcc, 20.0, (width, height))

# Ant
# env = create_maze_env('AntMaze')
# env.reset()
# # Save the test image to a file
# # img.show()
# for i in range(50):
#     env.step(env.action_space.sample())
#     test = env.render("rgb_array")
#     img = Image.fromarray(test, 'RGB')
#     img.save(f'ant-test/img-{i}.png')
# env.close()
    
# Test Reacher-v2
# print("Test Reacher-v2")
# env_reacher = gym.make("Reacher-v2")
# print(env_reacher.observation_space)


# Test AntMaze (custom)
# print("Test AntMaze (custom)")
# env_ant = create_maze_env('AntMaze')
# print(env_ant.observation_space)
# # print(env_ant.model.site_pos[env_ant.target_sid])

# Test Reacher (custom)
# print("Test Reacher (custom)")
# env = create_reacher_env()
# print(env.observation_space)
# print(env.model.site_pos[env.target_sid])
# obs = env.reset()
# env.model.site_pos[env.target_sid] = [-0.6, -1.4, -0.4]

# # env.render()
# hand_poses = []
# print("hand_pos",obs[-6:-3])
# # env.data.site_xpos = [0.1,0.1,0.1]
# for e in range(1):
#     for i in range(10):
#         a = env.action_space.sample()
#         # print ("action", a)
#         obs, _, _, _ = env.step(a)
#         hand_pos = obs[-6:-3]
#         hand_poses.append(hand_pos)
#         print("env.model.site_pos[env.target_sid]", env.model.site_pos[env.target_sid])
#         print("obs[-3:]", obs[-3:])
#         # print ("hand_pos", hand_pos)
#         # test = env.render(mode="rgb_array")
#         # img = Image.fromarray(test, 'RGB')
#         # img.save(f'reacher-test/img-{i}.png')
# print(obs)
# print(len(obs))
# print ("Min: ", np.min(hand_poses, axis=0))
# print ("Max: ", np.max(hand_poses, axis=0))
# print ("Mean: ", np.mean(hand_poses, axis=0))
# print ("STD: ", np.std(hand_poses, axis=0))
# print("-----")
# print(env.data.site_xpos)
# print(env.model.site_name2id)

# env.close()

# Test WidowX

# print("Test WidowX")
# env_widowx = create_widow_env()
# # print(env_widowx.observation_space)

# env_widowx.reset()
# for i in range(1):
#     obs, reward, done, info = env_widowx.step(env_widowx.action_space.sample())
#     test = env_widowx.render("rgb_array")
#     img = Image.fromarray(test, 'RGB')
#     img.save(f'widowx-test/img-{i}.png')

# env_widowx.close()

env_widowx = create_widow_env()
# done = False
# for _ in range(40):
#     obs, reward, done, info = env_widowx.step(env_widowx.action_space.sample())
#     test = env_widowx.render("rgb_array")
    
#     # Ensure the image is in BGR format for OpenCV
#     test_bgr = test[:, :, ::-1]  # Convert RGB to BGR
#     video.write(test_bgr)

# # Release the video writer
# video.release()

# print("Video saved:", video_filename)
# print(env_widowx.action_space)
# print(env_widowx.observation_space)

import roboverse
env_widowx_vanilla = roboverse.make
("Widow250EEPosition-v0")

env = env_widowx_vanilla("Widow250EEPosition-v0")
print("env.action_space", env.action_space)
print("env.observation_space", env.observation_space)

print("env_widowx.observation_space",env_widowx.observation_space)
# for _ in range(100):
#     a = env_widowx.action_space.sample()
#     obs, reward, done, info = env_widowx.step(a)
# print()

# import numpy as np

# l = [1,2,3,5,6,7,8,9,10]
# print(l[:-3])

# print(np.squeeze(l))
# if 0.0:
#     print("DONE ")

# print(True == 0)

# ---

# Test 
# test0 = np.linalg.norm([-3,-1,-0.5])
# test1 = np.linalg.norm([-2,-1,-0.5])
# test1bis = np.linalg.norm([-2, 1,-0.5])
# test2 = np.linalg.norm([-0.2,-.2,0])
# print(test0)
# print(test1)
# print(test1bis)
# print(test2)


# print(True + True)


# class bank(object):
#     def __init__(self, amount) -> None:
#         self.account = amount
#     def eval(self):
#         self.account -=100
        
# AE_account = bank(amount=1000)

# allo = AE_account.eval()
# print(AE_account.eval())
# print(allo)
# # print(AE_account.account)
# # AE_account.eval()
# # print(AE_account.account)

# obs = np.arange(20)

# print(obs[:-3])
# print(obs)

# from gym.spaces import Box


# class WrapperTest(object):
#     def __init__(self, env, **kwargs):
#         self.base_env = env
    
#     @property
#     def observation_space(self):
#         # Get the original observation space from base_env
#         original_space = self.base_env.observation_space

#         # Create the new observation space with adjusted shape
#         # Use the original space's bounds, excluding the last three dimensions
#         new_low = original_space.low[:-3]
#         new_high = original_space.high[:-3]
#         print(type(original_space.dtype))
#         # Return the new observation space
#         return Box(low=new_low, high=new_high, dtype=np.float32)

#     @property
#     def action_space(self):
#         return self.base_env.action_space
    
# env = WrapperTest(env)

# print(env.observation_space)
# # print(env.observation_dim)
# print(env.action_space)
# print(env.action_dim)