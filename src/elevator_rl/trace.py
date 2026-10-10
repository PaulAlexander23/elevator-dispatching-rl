"""Record BuildingEnv trajectories for replay by the C++ port.

The C++ simulator uses its own random numbers, so it cannot match NumPy draw
for draw. Instead the trace records each step's arrivals along with the
actions, and the C++ replay injects those arrivals and checks every
observation (both types), reward and truncation flag exactly.

Format, one record per line, space-separated:

    config n_floors=30 n_lifts=4 ... substeps=10
    env reward_shaping=1 max_steps=50
    reset
    arrivals <n> <floor> <destination> ...      (10 warm-up rounds after a reset)
    custom <values...>
    box <values...>
    step <action per lift...>
    arrivals <n> <floor> <destination> ...
    result <reward> <truncated>
    custom <values...>
    box <values...>

    uv run python -m elevator_rl.trace --preset full --steps 400 --out full.trace
"""

import argparse
from dataclasses import fields
from pathlib import Path

import numpy as np

from elevator_rl.building import PRESETS, BuildingConfig
from elevator_rl.building_env import BuildingEnv


def config_line(config: BuildingConfig):
    """`key=value` pairs in field order; floats use repr, so they round-trip exactly."""
    parts = []
    for f in fields(config):
        value = getattr(config, f.name)
        if isinstance(value, bool):
            value = int(value)
        parts.append(f"{f.name}={value!r}" if isinstance(value, float) else f"{f.name}={value}")
    return "config " + " ".join(parts)


def record(config, n_steps=400, seed=0, reward_shaping=True, max_steps=50):
    """Run a random policy and return the trace as a list of lines."""
    env = BuildingEnv(config, reward_shaping=reward_shaping, max_steps=max_steps)
    # A second env object that only reads the first one's state for "box" obs.
    box = BuildingEnv(config, obs_type="box")
    box.sim = env.sim

    drawn = []
    draw = env.sim.draw_arrivals

    def draw_and_record(fraction):
        drawn.append(draw(fraction))
        return drawn[-1]

    env.sim.draw_arrivals = draw_and_record

    lines = [
        config_line(env.config),
        f"env reward_shaping={int(reward_shaping)} max_steps={max_steps}",
    ]

    def arrivals_lines():
        out = [f"arrivals {len(a)} " + " ".join(f"{f} {d}" for f, d in a) for a in drawn]
        drawn.clear()
        return [line.rstrip() for line in out]

    def observations():
        return [
            "custom " + " ".join(str(int(v)) for v in env._observation()),
            "box " + " ".join(repr(float(v)) for v in box._observation()),
        ]

    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    lines += ["reset", *arrivals_lines(), *observations()]
    for _ in range(n_steps):
        action = rng.integers(env.config.n_actions, size=env.config.n_lifts)
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
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    lines = record(PRESETS[args.preset], args.steps, args.seed, max_steps=args.max_steps)
    Path(args.out).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
