"""Benchmark JAX PPO throughput on the settings the SB3 sweeps used.

Preset full, relative observations, 256x256 MLPs, 3 epochs, as in
`elevator_rl.sweep`. Compilation and evaluation are not timed.

    uv run --group jax-cuda python scripts/time_jax_ppo.py --devices gpu cpu
"""

import argparse

import jax

from elevator_rl.jax_ppo import PPOConfig, train

# (n_envs, n_steps, minibatch): the SB3 sweep's settings, then larger ones.
SETTINGS = [
    (64, 64, 64),
    (64, 64, 512),
    (64, 64, 2048),
    (1024, 8, 2048),
    (4096, 8, 4096),
    (16384, 8, 16384),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--devices", nargs="+", default=["gpu"])
    parser.add_argument("--seconds", type=float, default=20, help="rough training time per row")
    args = parser.parse_args()

    print("| Device | n_envs | n_steps | Minibatch | PPO steps/s |")
    print("|---|---:|---:|---:|---:|")
    for name in args.devices:
        device = jax.devices(name)[0]
        for n_envs, n_steps, batch_size in SETTINGS:
            config = PPOConfig(
                n_envs=n_envs, n_steps=n_steps, batch_size=batch_size, net_arch=(256, 256)
            )
            rollout = n_envs * n_steps
            with jax.default_device(device):
                # A short run to estimate the speed, then one of about --seconds.
                _, history = train("full", 4 * rollout, config, eval_freq=10**12, verbose=False)
                steps = int(history[-1]["steps_per_second"] * args.seconds)
                _, history = train(
                    "full", max(steps, 4 * rollout), config, eval_freq=10**12, verbose=False
                )
            rate = history[-1]["steps_per_second"]
            print(f"| {name} | {n_envs} | {n_steps} | {batch_size} | {rate:,.0f} |", flush=True)


if __name__ == "__main__":
    main()
