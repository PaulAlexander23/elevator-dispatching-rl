"""Record BuildingEnv trajectories for replay by the C++ port.

The C++ simulator uses its own random numbers, so it cannot match NumPy draw
for draw. Instead the trace records each step's arrivals along with the
actions, and the C++ replay injects those arrivals and checks every
observation (all three types), reward and truncation flag exactly.

Format, one record per line, space-separated:

    config n_floors=30 n_lifts=4 ... substeps=10
    env shaped=1 pickup=1.0 ... max_steps=50 action_mode=step observe_direction=0
    reset
    arrivals <n> <floor> <destination> ...      (10 warm-up rounds after a reset)
    custom <values...>
    box <values...>
    relative <values...>
    step <action per lift...>
    arrivals <n> <floor> <destination> ...
    result <reward> <truncated>
    custom <values...>
    box <values...>
    relative <values...>

    uv run python -m elevator_rl.trace --preset full --steps 400 --out full.trace
"""

import argparse
from dataclasses import fields
from pathlib import Path

import numpy as np

from elevator_rl.building import PRESETS, BuildingConfig
from elevator_rl.building_env import ACTION_MODES, SHAPINGS, BuildingEnv, Shaping


def config_line(config: BuildingConfig):
    """`key=value` pairs in field order; floats use repr, so they round-trip exactly."""
    parts = []
    for f in fields(config):
        value = getattr(config, f.name)
        if isinstance(value, bool):
            value = int(value)
        parts.append(f"{f.name}={value!r}" if isinstance(value, float) else f"{f.name}={value}")
    return "config " + " ".join(parts)


def env_line(env: BuildingEnv):
    """The env options, in the order elevator_replay reads them."""
    shaping = env.reward_shaping or Shaping()
    return (
        f"env shaped={int(env.reward_shaping is not None)} pickup={shaping.pickup!r} "
        f"empty_serve={shaping.empty_serve!r} progress={shaping.progress!r} "
        f"waiting={shaping.waiting!r} gamma={shaping.gamma!r} max_steps={env.max_steps} "
        f"action_mode={env.action_mode} observe_direction={int(env.observe_direction)}"
    )


def record(
    config,
    n_steps=400,
    seed=0,
    reward_shaping=True,
    max_steps=50,
    action_mode="step",
    observe_direction=False,
):
    """Run a random policy and return the trace as a list of lines."""
    env = BuildingEnv(
        config,
        reward_shaping=reward_shaping,
        max_steps=max_steps,
        action_mode=action_mode,
        observe_direction=observe_direction,
    )
    drawn = []
    draw = env.sim.draw_arrivals

    def draw_and_record(fraction):
        drawn.append(draw(fraction))
        return drawn[-1]

    env.sim.draw_arrivals = draw_and_record
    lines = [config_line(env.config), env_line(env)]

    def arrivals_lines():
        out = [f"arrivals {len(a)} " + " ".join(f"{f} {d}" for f, d in a) for a in drawn]
        drawn.clear()
        return [line.rstrip() for line in out]

    def observation(obs_type):
        env.obs_type = obs_type
        obs = env._observation()
        env.obs_type = "custom"
        return obs

    def observations():
        return [
            "custom " + " ".join(str(int(v)) for v in observation("custom")),
            "box " + " ".join(repr(float(v)) for v in observation("box")),
            "relative " + " ".join(repr(float(v)) for v in observation("relative")),
        ]

    rng = np.random.default_rng(seed)
    n_actions = int(env.action_space.nvec[0])
    env.reset(seed=seed)
    lines += ["reset", *arrivals_lines(), *observations()]
    for _ in range(n_steps):
        action = rng.integers(n_actions, size=env.config.n_lifts)
        _, reward, _, truncated, _ = env.step(action)
        lines.append("step " + " ".join(str(int(a)) for a in action))
        lines += arrivals_lines()
        lines.append(f"result {float(reward)!r} {int(truncated)}")
        lines += observations()
        if truncated:
            env.reset(seed=int(rng.integers(2**31)))
            lines += ["reset", *arrivals_lines(), *observations()]
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=PRESETS, default="full")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=50, help="episode length")
    parser.add_argument("--shaping", choices=SHAPINGS, default="default")
    parser.add_argument("--action-mode", choices=ACTION_MODES, default="step")
    parser.add_argument("--observe-direction", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    lines = record(
        PRESETS[args.preset],
        args.steps,
        args.seed,
        args.shaping,
        args.max_steps,
        args.action_mode,
        args.observe_direction,
    )
    Path(args.out).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
