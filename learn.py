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
               goal_type="GoalTrajMimicv2", goal_params=dict(visualize_goal=True), reward_type="MimicReward")

action_dim = env.action_space.shape[0]

seed = 1
np.random.seed(seed)

# Create the PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    gamma=0.99,
    seed=seed,
    policy_kwargs=dict(net_arch=dict(pi=[1024, 1024], vf=[1024, 1024])),
    tensorboard_log="logs/"
)

# Setup checkpoint callback to save models during training
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./models/",
    name_prefix="ppo_loco_mujoco"
)




# Train the model
TRAIN_TIMESTEPS = 500000
print("Training model for", TRAIN_TIMESTEPS, "timesteps")
model.learn(total_timesteps=TRAIN_TIMESTEPS, callback=checkpoint_callback, progress_bar=True)

# Save the final model
model.save("ppo_loco_mujoco_final")
print("Model training complete and saved")



# Evaluation loop
print("Starting evaluation")
env.reset()
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