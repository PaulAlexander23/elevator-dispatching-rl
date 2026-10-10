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

The C++ parity tests in `tests/test_cpp.py` are skipped until the C++ port is
built (see below).

## Layout

```
src/elevator_rl/   sim.py (simulator), env.py (Gymnasium env), train.py (PPO training),
                   baselines.py (hand-written policies), benchmark.py (obs-space comparison),
                   building.py + building_env.py (configurable multi-lift env),
                   sweep.py + compare.py (PPO throughput and learning comparisons),
                   trace.py (trajectories for the C++ parity check),
                   imitate.py (warm start from the heuristic), cpp_env.py (C++ VecEnv)
cpp/               C++ port of the multi-lift env, its Python binding (python/),
                   replay, benchmark and tests
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

### Training the 30-floor presets

PPO from scratch does not learn the 30-floor presets (`tall` onwards): a
delivery takes dozens of correct steps in a row, so random exploration rarely
finds one, and the empty-serve penalty teaches it to never open the doors.
What works is to imitate a classic collective-control heuristic first, with
an observation centred on each lift, then fine-tune with a low learning rate:

```sh
uv run python -m elevator_rl.train --preset full --obs-type relative \
    --n-envs 64 --n-steps 32 --batch-size 64 --net-arch 256 256 \
    --pretrain 100000 --learning-rate 3e-5 --timesteps 500000
```

On `full` this delivers about 67 passengers per episode after 500k steps
(the heuristic, `baselines.CollectivePolicy`, delivers 69 and a random
policy 7). `BuildingEnv` also has a target-floor action mode, a
direction-of-travel observation and several reward shapings
(`building_env.SHAPINGS`) for experiments; `elevator_rl.compare` runs them.

To see where PPO's time goes as training is parallelised, sweep the number of
envs and the batch size. Each run reports PPO steps/s and the share of time
spent stepping envs, in policy forward passes and SB3 bookkeeping, and in the
gradient update:

```sh
uv run python -m elevator_rl.sweep --preset full --n-envs 1 4 16 64 \
    --batch-sizes 64 512 --vec-envs dummy subproc --repeats 2
```

`--torch-threads 1 8` and `--devices cpu cuda` also sweep the learner. On an
8-core CPU, 8 threads is fastest (16 hyperthreads is slower). A GPU only
helps with large minibatches (512+), and then the env's speed matters again:
with `--vec-env cpp`, 1024 envs and minibatch 2048, a GTX 1080 runs about
50k PPO steps/s, against 4.5k with the Python envs. `train.py --device cuda`
trains on the GPU. The project's lockfile pins the CPU build of torch, so a
CUDA build has to be installed separately.

## C++ port

`cpp/` is a C++17 port of `BuildingSim` and `BuildingEnv` with fixed-size
state (no allocation per step) and its own random number generator. It
reproduces the Python dynamics exactly when given the same arrivals:
`elevator_rl.trace` records Python trajectories with their arrivals, and
`elevator_replay` checks every observation, reward and truncation flag.
Random draws differ from NumPy's, so the random-policy statistics are
compared instead.

```sh
uv sync                                            # installs nanobind for the binding
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=.venv/bin/python
cmake --build cpp/build                            # also builds src/elevator_rl/_cpp*.so
ctest --test-dir cpp/build                         # C++ unit tests
uv run pytest tests/test_cpp.py tests/test_cpp_env.py   # parity with Python
cpp/build/elevator_bench --preset full --steps 2000000
```

The Python module `elevator_rl._cpp` exposes the C++ env, and
`elevator_rl.cpp_env.CppVecEnv` wraps it as a Stable-Baselines3 `VecEnv`:
one call steps every env (without holding the GIL) and writes into
preallocated arrays. Train on it with `--vec-env cpp`:

```sh
uv run python -m elevator_rl.train --preset full --obs-type relative --vec-env cpp \
    --n-envs 64 --n-steps 32 --batch-size 64 --net-arch 256 256 \
    --pretrain 100000 --learning-rate 3e-5
```

Rebuild with `cmake --build cpp/build` after changing the C++; the module is
not part of the installed package.

Each floor queue holds at most 512 passengers (the longest seen in Python is
about 310); extra arrivals are dropped and counted, and the parity tests
check that none are.

## License

[MIT](LICENSE)
