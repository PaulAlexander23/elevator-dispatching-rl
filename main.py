from env import LiftEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor


def main():
    envs = DummyVecEnv(
        [lambda: LiftEnv(
            reward_shaping=True, multi_binary=False, multi_discrete=False, custom_obs=True)])
    eval_envs = DummyVecEnv(
        [lambda: Monitor(LiftEnv(multi_binary=False, multi_discrete=False, custom_obs=True))])
    model = PPO("MlpPolicy", envs, device="cpu", verbose=1, n_epochs=3)
    # policy_kwargs={"net_arch":{"pi":[64],"vf":[64]
    # }}
    mean_reward, std_reward = evaluate_policy(model, eval_envs)
    print(f"mean reward: {mean_reward}, std reward: {std_reward}")

    callback = EvalCallback(eval_envs, eval_freq=50_000)
    model.learn(total_timesteps=2_000_000, callback=callback, log_interval=10)

    mean_reward, std_reward = evaluate_policy(model, eval_envs)
    print(f"mean reward: {mean_reward}, std reward: {std_reward}")

    model.save("model.zip")


if __name__ == "__main__":
    main()
