import numpy as np
import loco_mujoco
from loco_mujoco.task_factories import DefaultDatasetConf, LAFAN1DatasetConf, AMASSDatasetConf
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# note: we do not support parallel environments in gymnasium yet!
env = gym.make("LocoMujoco", env_name="MJHumanoid",
               amass_dataset_conf=AMASSDatasetConf(["CMU/CMU/13/13_17_poses"]),
            #    lafan1_dataset_conf=LAFAN1DatasetConf("walk1_subject1"),
               goal_type="GoalTrajMimicv2", goal_params=dict(visualize_goal=True), render_mode="human", reward_type="MimicReward")

model = PPO.load("/Users/benediktstroebl/Documents/GitHub/loco-mujoco/models/ppo_loco_mujoco_200000_steps.zip", env=env)

action_dim = env.action_space.shape[0]

# Evaluation loop
print("Starting evaluation")
env.reset()
img = env.render()
obs, _ = env.reset()
done = False
i = 0
total_reward = 0
# Evaluate the trained model
while True:
    if i == 1000 or done:
        obs, _ = env.reset()
        i = 0
        print(f"Episode completed with total reward: {total_reward}")
        total_reward = 0
    
    # Use the trained model to predict actions
    action, _ = model.predict(obs, deterministic=True)
    
    # Take a step in the environment
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward

    env.render()
    i += 1