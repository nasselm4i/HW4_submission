#!/bin/bash

echo "Starting tasks..."

Reacher Tasks
echo "Starting Reacher tasks..."

# Q1: Uniform distribution experiments for Reacher
echo "Running Q1 for Reacher..."
python run_hw4_gcrl.py env.env_name=reacher env.exp_name=DEBUG_q1_reacher_uniform_seed_10 env.distribution=uniform logging.random_seed=10
python run_hw4_gcrl.py env.env_name=reacher env.exp_name=DEBUG_q1_reacher_uniform_seed_20 env.distribution=uniform logging.random_seed=20
python run_hw4_gcrl.py env.env_name=reacher env.exp_name=DEBUG_q1_reacher_uniform_seed_30 env.distribution=uniform logging.random_seed=30

# Q2: Normal distribution experiments for Reacher
echo "Running Q2 for Reacher..."
python run_hw4_gcrl.py env.env_name=reacher env.exp_name=DEBUG_q2_reacher_normal env.distribution=normal logging.random_seed=10
python run_hw4_gcrl.py env.env_name=reacher env.exp_name=DEBUG_q2_reacher_normal env.distribution=normal logging.random_seed=20
python run_hw4_gcrl.py env.env_name=reacher env.exp_name=DEBUG_q2_reacher_normal_seed_30 env.distribution=normal logging.random_seed=30

# Q3: Normal distribution with relative goals for Reacher
echo "Running Q3 for Reacher..."
python run_hw4_gcrl.py env.env_name=reacher env.exp_name=DEBUG_q3_reacher_normal_relative env.distribution=normal env.relative_goal=true

Q4: Normal distribution, varying k for Reacher
echo "Running Q4 for Reacher..."
python run_hw4_gcrl.py env.env_name=reacher env.exp_name=DEBUG_q4_reacher_normal_k10 env.distribution=normal env.k=10 env.task_name=gcrl_v2 logging.save_params=true alg.n_iter=1800
python run_hw4_gcrl.py env.env_name=reacher env.exp_name=DEBUG_q4_reacher_normal_k5 env.distribution=normal env.k=5 env.task_name=gcrl_v2 logging.save_params=true alg.n_iter=1800

# Q5: Normal distribution, Hierarchical RL for Reacher with k=5
echo "Running Q5 for Reacher..."
python run_hw4_gcrl.py env.env_name=reacher env.task_name=hrl env.distribution=normal env.exp_name=DEBUG_q5_reacher_hrl_gf5 env.k=5  alg.n_iter=400 env.low_level_agent_path=//teamspace/studios/this_studio/robot_learning/data/hw4_DEBUG_q4_reacher_normal_k10_reacher_03-04-2024_23-57-31/agent.pt
python run_hw4_gcrl.py env.env_name=reacher env.task_name=hrl env.distribution=normal env.exp_name=DEBUG_q5_reacher_hrl_gf10 env.k=10 env.distribution=normal  alg.n_iter=400 low_level_agent_path=


# # WidowX Tasks
echo "Starting WidowX tasks..."

# Q1: Uniform distribution experiments for WidowX
echo "Running Q1 for WidowX..."
python run_hw4_gcrl.py env.env_name=widowx env.exp_name=DEBUG_q1_widowx_uniform_seed_10 env.distribution=uniform logging.random_seed=10
python run_hw4_gcrl.py env.env_name=widowx env.exp_name=DEBUG_q1_widowx_uniform_seed_20 env.distribution=uniform logging.random_seed=20
python run_hw4_gcrl.py env.env_name=widowx env.exp_name=DEBUG_q1_widowx_uniform_seed_30 env.distribution=uniform logging.random_seed=30

# Q2: Normal distribution experiments for WidowX
echo "Running Q2 for WidowX..."
python run_hw4_gcrl.py env.env_name=widowx env.exp_name=DEBUG_q2_widowx_normal_seed_10 env.distribution=normal logging.random_seed=10
python run_hw4_gcrl.py env.env_name=widowx env.exp_name=DEBUG_q2_widowx_normal_seed_20 env.distribution=normal logging.random_seed=20
python run_hw4_gcrl.py env.env_name=widowx env.exp_name=DEBUG_q2_widowx_normal_seed_30 env.distribution=normal logging.random_seed=30

# Q3: Normal distribution with relative goals for WidowX
echo "Running Q3 for WidowX..."
python run_hw4_gcrl.py env.env_name=widowx env.exp_name=DEBUG_q3_widowx_normal_relative env.distribution=normal env.relative_goal=true

# Q4: Normal distribution, varying k for WidowX
echo "Running Q4 for WidowX..."
python run_hw4_gcrl.py env.env_name=widowx env.exp_name=DEBUG_q4_widowx_normal_k10 env.distribution=normal env.k=5 env.task_name=gcrl_v2 logging.save_params=true
python run_hw4_gcrl.py env.env_name=widowx env.exp_name=DEBUG_q4_widowx_normal_k5 env.distribution=normal env.k=10 env.task_name=gcrl_v2

# Q5: Normal distribution, Hierarchical RL for WidowX with k=5
echo "Running Q5 for WidowX..."
python run_hw4_gcrl.py env.env_name=widowx env.task_name=hrl env.distribution=normal env.exp_name=DEBUG_q5_widowx_hrl_gf5 env.k=5 alg.n_iter=400 env.low_level_agent_path=/teamspace/studios/this_studio/robot_learning/data/hw4_DEBUG_q4_widowx_normal_k10_widowx_04-04-2024_00-31-24/agent.pt
# python run_hw4_gcrl.py env.env_name=widowx env.task_name=hrl env.distribution=normal env.exp_name=DEBUG_q5_widowx_hrl_gf10 env.k=10 env.distribution=normal alg.n_iter=400 low_level_agent_path=
# 