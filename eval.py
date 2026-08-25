from env import LiftEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

RENDER = False

model = PPO.load("cts_obs_model.zip")
eval_envs = DummyVecEnv([lambda: Monitor(LiftEnv("human"))])
if RENDER:
    evaluate_policy(model, eval_envs, render=True,
                    n_eval_episodes=1, deterministic=False)
mean_reward, std_reward = evaluate_policy(
    model, eval_envs, n_eval_episodes=1000)
print(f"mean reward: {mean_reward}, std reward: {std_reward}")
mean_reward, std_reward = evaluate_policy(
    model, eval_envs, deterministic=False, n_eval_episodes=1000)
print(f"mean reward: {mean_reward}, std reward: {std_reward}")

model = PPO.load("binary_obs_model.zip")
eval_envs = DummyVecEnv([lambda: Monitor(LiftEnv("human", multi_binary=True))])
if RENDER:
    evaluate_policy(model, eval_envs, render=True,
                    n_eval_episodes=1, deterministic=False)
mean_reward, std_reward = evaluate_policy(
    model, eval_envs, n_eval_episodes=1000)
print(f"mean reward: {mean_reward}, std reward: {std_reward}")
mean_reward, std_reward = evaluate_policy(
    model, eval_envs, deterministic=False, n_eval_episodes=1000)
print(f"mean reward: {mean_reward}, std reward: {std_reward}")

model = PPO.load("dis_obs_model.zip")
eval_envs = DummyVecEnv(
    [lambda: Monitor(LiftEnv("human", multi_discrete=True))])
if RENDER:
    evaluate_policy(model, eval_envs, render=True,
                    n_eval_episodes=1, deterministic=False)
mean_reward, std_reward = evaluate_policy(
    model, eval_envs, n_eval_episodes=1000)
print(f"mean reward: {mean_reward}, std reward: {std_reward}")
mean_reward, std_reward = evaluate_policy(
    model, eval_envs, deterministic=False, n_eval_episodes=1000)
print(f"mean reward: {mean_reward}, std reward: {std_reward}")

model = PPO.load("model.zip")
eval_envs = DummyVecEnv(
    [lambda: Monitor(LiftEnv("human", multi_discrete=True))])
if RENDER:
    evaluate_policy(model, eval_envs, render=True,
                    n_eval_episodes=1, deterministic=False)
mean_reward, std_reward = evaluate_policy(
    model, eval_envs, n_eval_episodes=1000)
print(f"mean reward: {mean_reward}, std reward: {std_reward}")
mean_reward, std_reward = evaluate_policy(
    model, eval_envs, deterministic=False, n_eval_episodes=1000)
print(f"mean reward: {mean_reward}, std reward: {std_reward}")
