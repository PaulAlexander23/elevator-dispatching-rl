import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from elevator_rl.building import PRESETS
from elevator_rl.building_env import ACTION_MODES, SHAPINGS, BuildingEnv
from elevator_rl.building_env import OBS_TYPES as BUILDING_OBS_TYPES
from elevator_rl.env import OBS_TYPES, LiftEnv


def make_env(
    obs_type="custom", reward_shaping=False, render_mode=None, preset=None, action_mode="step"
):
    """The original `LiftEnv`, or a `BuildingEnv` when a preset is named."""
    if preset is None:
        return LiftEnv(render_mode, reward_shaping=reward_shaping, obs_type=obs_type)
    return BuildingEnv(
        preset,
        render_mode,
        reward_shaping=reward_shaping,
        obs_type=obs_type,
        action_mode=action_mode,
    )


VEC_ENVS = ("dummy", "subproc", "cpp")


def make_vec_env(
    n_envs=1,
    vec_env="dummy",
    obs_type="custom",
    reward_shaping=False,
    preset=None,
    action_mode="step",
    seed=0,
):
    """`n_envs` training envs: Python in one process, one process each, or C++."""
    if vec_env == "cpp":
        if preset is None:
            raise ValueError("the C++ env is a BuildingEnv: name a preset")
        from elevator_rl.cpp_env import CppVecEnv

        return CppVecEnv(
            preset, n_envs, obs_type, reward_shaping, action_mode=action_mode, seed=seed or 0
        )
    cls = SubprocVecEnv if vec_env == "subproc" else DummyVecEnv
    return cls(
        [lambda: make_env(obs_type, reward_shaping, preset=preset, action_mode=action_mode)]
        * n_envs
    )


def make_model(
    obs_type="custom",
    reward_shaping=True,
    seed=None,
    n_steps=2048,
    verbose=1,
    preset=None,
    action_mode="step",
    n_envs=1,
    batch_size=64,
    net_arch=None,
    learning_rate=3e-4,
    vec_env="dummy",
):
    """PPO on a training env; shaping only affects training, not evaluation.

    `n_steps` is per env, so each rollout holds n_envs x n_steps steps.
    """
    envs = make_vec_env(n_envs, vec_env, obs_type, reward_shaping, preset, action_mode, seed)
    return PPO(
        "MlpPolicy",
        envs,
        device="cpu",
        verbose=verbose,
        n_epochs=3,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        policy_kwargs={"net_arch": net_arch} if net_arch else None,
    )


def make_eval_env(obs_type="custom", render_mode=None, preset=None, action_mode="step"):
    """Unshaped env, so the reward is the number of passengers delivered."""
    return DummyVecEnv(
        [lambda: Monitor(make_env(obs_type, False, render_mode, preset, action_mode))]
    )


def main(
    total_timesteps=2_000_000,
    eval_freq=50_000,
    n_eval_episodes=10,
    save_path="model.zip",
    n_steps=2048,
    obs_type="custom",
    preset=None,
    shaping="default",
    action_mode="step",
    n_envs=1,
    batch_size=64,
    net_arch=None,
    learning_rate=3e-4,
    pretrain=0,
    vec_env="dummy",
):
    """Train PPO and save it. With `pretrain`, first imitate the collective
    heuristic on that many samples (BuildingEnv presets, step actions only)."""
    reward_shaping = True if preset is None else shaping
    model = make_model(
        obs_type,
        reward_shaping,
        n_steps=n_steps,
        preset=preset,
        action_mode=action_mode,
        n_envs=n_envs,
        batch_size=batch_size,
        net_arch=net_arch,
        learning_rate=learning_rate,
        vec_env=vec_env,
    )
    if pretrain:
        if preset is None or action_mode != "step":
            raise ValueError("pretrain needs a BuildingEnv preset and step actions")
        from elevator_rl.imitate import warm_start

        log_likelihood, _ = warm_start(model, preset, pretrain, shaping, obs_type=obs_type)
        print(f"imitation: mean log-likelihood of the heuristic's actions {log_likelihood:.3f}")
    eval_envs = make_eval_env(obs_type, preset=preset, action_mode=action_mode)
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
    parser.add_argument(
        "--obs-type",
        choices=sorted(set(OBS_TYPES) | set(BUILDING_OBS_TYPES)),
        default="custom",
        help="relative is BuildingEnv only; multi_binary and multi_discrete are LiftEnv only",
    )
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument(
        "--vec-env", choices=VEC_ENVS, default="dummy", help="cpp needs the built C++ module"
    )
    parser.add_argument("--n-steps", type=int, default=2048, help="rollout steps per env")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--net-arch", type=int, nargs="+", help="hidden layer sizes")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--pretrain",
        type=int,
        default=0,
        help="imitate the collective heuristic on this many samples first (--preset only)",
    )
    parser.add_argument(
        "--preset",
        choices=PRESETS,
        help="train on the multi-lift BuildingEnv with this config",
    )
    parser.add_argument(
        "--shaping",
        choices=SHAPINGS,
        default="default",
        help="training reward shaping for --preset (see building_env.SHAPINGS)",
    )
    parser.add_argument(
        "--action-mode",
        choices=ACTION_MODES,
        default="step",
        help="for --preset: one floor per action, or a target floor per lift",
    )
    args = parser.parse_args()
    main(
        total_timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        save_path=args.save_path,
        obs_type=args.obs_type,
        preset=args.preset,
        shaping=args.shaping,
        action_mode=args.action_mode,
        n_steps=args.n_steps,
        n_envs=args.n_envs,
        batch_size=args.batch_size,
        net_arch=args.net_arch,
        learning_rate=args.learning_rate,
        pretrain=args.pretrain,
        vec_env=args.vec_env,
    )


if __name__ == "__main__":
    cli()
