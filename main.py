from env import LiftEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor


def main(
    total_timesteps=2_000_000,
    eval_freq=50_000,
    n_eval_episodes=10,
    save_path="model.zip",
    n_steps=2048,
):
    envs = DummyVecEnv(
        [lambda: LiftEnv(
            reward_shaping=True, obs_type="custom")])
    eval_envs = DummyVecEnv(
        [lambda: Monitor(LiftEnv(obs_type="custom"))])
    model = PPO("MlpPolicy", envs, device="cpu", verbose=1, n_epochs=3,
                n_steps=n_steps)
    # policy_kwargs={"net_arch":{"pi":[64],"vf":[64]
    # }}
    mean_reward, std_reward = evaluate_policy(
        model, eval_envs, n_eval_episodes=n_eval_episodes)
    print(f"mean reward: {mean_reward}, std reward: {std_reward}")

    callback = EvalCallback(
        eval_envs, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes)
    model.learn(total_timesteps=total_timesteps,
                callback=callback, log_interval=10)

    mean_reward, std_reward = evaluate_policy(
        model, eval_envs, n_eval_episodes=n_eval_episodes)
    print(f"mean reward: {mean_reward}, std reward: {std_reward}")

    model.save(save_path)
    return model


if __name__ == "__main__":
    main()
