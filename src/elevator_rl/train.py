import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from elevator_rl.building import PRESETS
from elevator_rl.building_env import BuildingEnv
from elevator_rl.env import OBS_TYPES, LiftEnv


def make_env(obs_type="custom", reward_shaping=False, render_mode=None, preset=None):
    """The original `LiftEnv`, or a `BuildingEnv` when a preset is named."""
    if preset is None:
        return LiftEnv(render_mode, reward_shaping=reward_shaping, obs_type=obs_type)
    return BuildingEnv(preset, render_mode, reward_shaping=reward_shaping, obs_type=obs_type)


def make_model(
    obs_type="custom", reward_shaping=True, seed=None, n_steps=2048, verbose=1, preset=None
):
    """PPO on a training env; shaping only affects training, not evaluation."""
    envs = DummyVecEnv([lambda: make_env(obs_type, reward_shaping, preset=preset)])
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


def make_eval_env(obs_type="custom", render_mode=None, preset=None):
    """Unshaped env, so the reward is the number of passengers delivered."""
    return DummyVecEnv([lambda: Monitor(make_env(obs_type, False, render_mode, preset))])


def main(
    total_timesteps=2_000_000,
    eval_freq=50_000,
    n_eval_episodes=10,
    save_path="model.zip",
    n_steps=2048,
    obs_type="custom",
    preset=None,
):
    model = make_model(obs_type, n_steps=n_steps, preset=preset)
    eval_envs = make_eval_env(obs_type, preset=preset)
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
    parser.add_argument(
        "--preset",
        choices=PRESETS,
        help="train on the multi-lift BuildingEnv with this config (obs type custom or box)",
    )
    args = parser.parse_args()
    main(
        total_timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        save_path=args.save_path,
        obs_type=args.obs_type,
        preset=args.preset,
    )


if __name__ == "__main__":
    cli()
