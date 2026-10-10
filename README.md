# Elevator dispatching RL

Applying reinforcement learning to a lift (elevator) controller, using a
[Gymnasium](https://gymnasium.farama.org/) environment and PPO from
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/).

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/):

```sh
uv sync                     # runtime + dev dependencies
uv sync --extra bench       # + psutil for the timing scripts
uv sync --extra gnn         # + torch-geometric for scripts/gnn.py
```

## Usage

```sh
# Train (2M steps by default, roughly 30 minutes on a laptop CPU)
uv run python -m elevator_rl.train --timesteps 2000000 --save-path model.zip

# Evaluate a trained model (pass the obs type it was trained with)
uv run python scripts/eval.py model.zip --obs-type custom --render

# Baselines: random policy and an up/down sweep
uv run python scripts/eval_random.py

# Throughput benchmarks
uv run python scripts/time_env.py
uv run python scripts/time_ppo.py
```

## Benchmark

Compare the observation spaces (and the random / up-down baselines):

```sh
uv run python -m elevator_rl.benchmark --timesteps 200000 --seeds 3 --readme README.md
```

This trains PPO for each observation type and seed in parallel (one process
per CPU by default; `--workers` to change), and rewrites the table below.
Use `--obs-types custom box` to run a subset and `--json results.json` to keep
the per-seed numbers.

<!-- BENCHMARK:START -->
_No results yet: run the command above to fill in this table._
<!-- BENCHMARK:END -->

## Tests and linting

```sh
uv run pytest            # fast tests, a few seconds
uv run pytest -m slow    # also the full training run (~30 minutes)
uv run ruff check .
uv run ruff format .
```

## Layout

```
src/elevator_rl/   sim.py (simulator), env.py (Gymnasium env), train.py (PPO training),
                   baselines.py (hand-written policies), benchmark.py (obs-space comparison),
                   building.py + building_env.py (configurable multi-lift env)
tests/             pytest suite
scripts/           evaluation, baselines, benchmarks and GNN experiments
```

## Environment

There are:
- floors: 10
- lifts: one to start with
- passengers: each is at floor x and wants to get to floor y

Actions
- Lift moves up one floor
- Lift moves down one floor
- Lift serves this floor

Transition

1. Sample new passengers: each floor gets a new passenger with
   probability 0.1, going to a uniformly random other floor.
2. Apply the lift action.
   If the lift is moving up/down, then just update the lift position.
   If the lift serves this floor, then:
   - Any passengers in the lift that want this floor leave.
   - Any passengers waiting enter the lift from first to arrive to last
     until the lift is full (capacity 8).

Episodes are truncated after 200 steps.

Rewards
- +1 for each passenger delivered to their floor.
- With `reward_shaping=True`: +1 for each passenger picked up, and -0.5
  for serving a floor where nobody gets on or off.

Observations (`obs_type`)
- `"box"`: lift position (scaled to [0, 1]); whether people are waiting
  at each floor; whether people in the lift want each floor.
- `"multi_binary"`: the same as three rows of flags, with the lift
  position one-hot.
- `"multi_discrete"`: per-floor counts: lift here (0 = no, k = yes with
  k - 1 passengers), people waiting (capped at 8), people in the lift
  wanting that floor.
- `"custom"`: lift position, passengers in the lift, then per-floor
  waiting and wanted counts.

## Multi-lift building env

`BuildingEnv` (`src/elevator_rl/building_env.py`) is a configurable,
heavier version of the env for experimenting with acceleration. One central
controller picks an action for every lift (`MultiDiscrete`), and the reward
is summed over lifts. `BuildingConfig` (`src/elevator_rl/building.py`)
switches each feature on or off:

| Field | Default | Effect |
|---|---|---|
| `n_floors`, `n_lifts` | 10, 1 | Building size; lifts act in index order, so lift 0 boards first |
| `hall_calls` | off | Up/down buttons; actions become up, down, serve up, serve down |
| `traffic` | `uniform` | `up_peak`, `down_peak`, `lunch`, or `day` (morning up-peak to evening down-peak over an episode) |
| `kinematics` | off | Acceleration, braking and door times, integrated in `substeps` per `step_seconds` |

The default config matches `LiftEnv` step for step with the same seed (a test
checks this). The presets add one feature at a time:
`original` → `tall` (30 floors) → `multi` (4 lifts) → `kinematic` →
`hall_calls` → `full` (day traffic).

```sh
uv run python -m elevator_rl.train --preset multi --timesteps 500000
uv run python scripts/time_env.py --preset full
uv run python scripts/time_ppo.py --preset full
```

## License

[MIT](LICENSE)
