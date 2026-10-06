import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from elevator_rl.env import OBS_TYPES, LiftEnv


def make_model(obs_type="custom", reward_shaping=True, seed=None, n_steps=2048, verbose=1):
    """PPO on a training env; shaping only affects training, not evaluation."""
    envs = DummyVecEnv([lambda: LiftEnv(reward_shaping=reward_shaping, obs_type=obs_type)])
    # policy_kwargs={"net_arch":{"pi":[64],"vf":[64]}}
    return PPO(
        "MlpPolicy",
        envs,
        device="cpu",
        verbose=verbose,
        n_epochs=3,
        n_steps=n_steps,
        seed=seed,
    )


def make_eval_env(obs_type="custom", render_mode=None):
    """Unshaped env, so the reward is the number of passengers delivered."""
    return DummyVecEnv([lambda: Monitor(LiftEnv(render_mode, obs_type=obs_type))])


def main(
    total_timesteps=2_000_000,
    eval_freq=50_000,
    n_eval_episodes=10,
    save_path="model.zip",
    n_steps=2048,
    obs_type="custom",
):
    model = make_model(obs_type, n_steps=n_steps)
    eval_envs = make_eval_env(obs_type)
    mean_reward, std_reward = evaluate_policy(model, eval_envs, n_eval_episodes=n_eval_episodes)
    print(f"mean reward: {mean_reward}, std reward: {std_reward}")

    callback = EvalCallback(eval_envs, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes)
    model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=10)

    mean_reward, std_reward = evaluate_policy(model, eval_envs, n_eval_episodes=n_eval_episodes)
    print(f"mean reward: {mean_reward}, std reward: {std_reward}")

    model.save(save_path)
    return model


def cli():
    parser = argparse.ArgumentParser(description="Train a PPO lift dispatcher.")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--save-path", default="model.zip")
    parser.add_argument("--obs-type", choices=OBS_TYPES, default="custom")
    args = parser.parse_args()
    main(
        total_timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        save_path=args.save_path,
        obs_type=args.obs_type,
    )


if __name__ == "__main__":
    cli()
