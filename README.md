# Elevator dispatching RL

Applying reinforcement learning to a lift (elevator) controller, using a
[Gymnasium](https://gymnasium.farama.org/) environment and PPO from
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/).

The env also has C++, EnvPool, JAX and NVIDIA Warp ports, built to learn how
far a Python env and its training loop can be accelerated. [docs/](docs/README.md)
has the plan, each phase's results and the setup notes.

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/):

```sh
uv sync                     # runtime + dev dependencies
uv sync --extra bench       # + psutil for the timing scripts
uv sync --extra gnn         # + torch-geometric for scripts/gnn.py
uv sync --no-group cpu --group cuda   # CUDA torch instead of the CPU build
```

torch comes from PyTorch's CPU index unless you ask for the `cuda` group,
which installs the CUDA 12.6 build (the newest that still runs on GTX 10xx
GPUs). The two groups conflict, so only one is installed at a time. A plain
`uv sync` or `uv run` switches back to the CPU build, so run GPU jobs with
`uv run --no-group cpu --group cuda ...` (or `uv run --no-sync ...` after
the sync above).

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
                   imitate.py (warm start from the heuristic), cpp_env.py (C++ VecEnv),
                   ctypes_env.py (the same VecEnv over the C API, through ctypes)
cpp/               C++ port of the multi-lift env, its Python binding (python/),
                   a flat C API built as a shared library (capi/), replay,
                   benchmark and tests
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
trains on the GPU, after `uv sync --no-group cpu --group cuda`.

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
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$PWD/.venv/bin/python
cmake --build cpp/build                            # also builds src/elevator_rl/_cpp*.so
                                                   # and src/elevator_rl/libelevator_c.so
ctest --test-dir cpp/build                         # C++ unit tests
uv run pytest tests/test_cpp.py tests/test_cpp_env.py tests/test_ctypes_env.py
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

The same env is also built as a plain shared library with a flat
`extern "C"` API (`cpp/capi/elevator_c.h`), which needs no Python headers
and can be loaded from any language. `elevator_rl.ctypes_env.CtypesVecEnv`
loads it with `ctypes` and gives the same results as `CppVecEnv`; train on
it with `--vec-env ctypes`. `scripts/time_ctypes.py` compares the call
overhead of the two routes.

Each floor queue holds at most 512 passengers (the longest seen in Python is
about 310); extra arrivals are dropped and counted, and the parity tests
check that none are.

The module needs Python headers. The project pins Python 3.12 (for EnvPool,
below), and Ubuntu's system 3.12 ships without them, so use uv's own Python:
`uv sync --python-preference only-managed`.

## EnvPool

[EnvPool](https://github.com/sail-sg/envpool) runs C++ envs on a thread pool.
It only takes new envs inside its own Bazel build, so `cpp/envpool/` holds
the elevator env written against EnvPool's `Env` interface, and
`scripts/build_envpool.sh` checks out EnvPool at a pinned commit, adds the
env, trims the package to EnvPool's core and builds a wheel. EnvPool needs
Python 3.12 or later and [bazelisk](https://github.com/bazelbuild/bazelisk)
(as `bazel`) on the PATH; the first build takes a few minutes.

```sh
scripts/build_envpool.sh
uv pip install cpp/build/envpool-dist/envpool-*.whl
uv run pytest tests/test_envpool_env.py                # step-for-step parity with the C++ env
uv run python scripts/time_vec_env.py                  # C++ VecEnv vs EnvPool sync/async
uv run python -m elevator_rl.train --preset full --obs-type relative --vec-env envpool
```

`uv sync` removes the wheel (it is not in the lockfile), so reinstall it
afterwards; `uv run` leaves it in place. `elevator_rl.envpool_env.EnvPoolVecEnv`
adapts the pool to SB3: EnvPool resets a finished env on the following step,
while SB3 expects the reset on the same step, so the adapter resets finished
envs straight away in one batched call. Only the `box` and `relative`
observations are available, because an EnvPool spec fixes the dtype at
compile time.

## JAX

`elevator_rl.jax_env` ports `BuildingEnv` (step actions; `box` and `relative`
observations) to pure JAX functions over fixed-size arrays, so thousands of
envs run in one `vmap` on a GPU. `elevator_rl.jax_ppo` is PPO written to
match SB3's, with the rollout and the update compiled into one XLA program.

```sh
uv sync --group jax-cuda                    # or --group jax for the CPU only
uv run pytest tests/test_jax_env.py tests/test_jax_ppo.py
uv run python -m elevator_rl.jax_ppo --preset full --n-envs 1024 --n-steps 8 \
    --batch-size 2048 --timesteps 5000000
uv run python scripts/time_jax_env.py       # env steps/s
uv run python scripts/time_jax_ppo.py       # PPO steps/s on the SB3 sweep's settings
```

The parity tests replay Python's arrivals in float64 and match every
observation exactly. That needs an `optimization_barrier` in the kinematics,
because XLA fuses `a + b * c` into a fused multiply-add, which rounds
differently from NumPy (the C++ port uses `-ffp-contract=off` for the same
reason). Training runs in float32.

On a GTX 1080, JAX PPO on `full` runs 30k steps/s with the training recipe's
settings (64 envs x 64 steps, minibatch 64), against 4.3k for SB3 with the
C++ env on the same GPU, and 238k steps/s with 16,384 envs.

If JAX falls back to the CPU with "Unable to load cuSPARSE", an older CUDA
library on `LD_LIBRARY_PATH` is shadowing the one in JAX's wheel: remove the
system library directories (`/usr/lib/x86_64-linux-gnu` and friends) from
`LD_LIBRARY_PATH`, or run with `env -u LD_LIBRARY_PATH`.

## NVIDIA Warp

`elevator_rl.warp_env` is a third GPU port, written as
[Warp](https://github.com/NVIDIA/warp) kernels: one GPU thread per env,
running ordinary loops and branches like the C++ port, rather than the masked
array code of the JAX port. The kernels run inside `jax.jit` through
`warp.jax_kernel`, so `jax_ppo` trains on either env with `--backend warp`.

```sh
uv sync --group warp                        # Warp's CUDA 12 build, plus JAX with CUDA
uv run pytest tests/test_warp_env.py
uv run python scripts/time_jax_env.py --backends warp jax --devices gpu
uv run python -m elevator_rl.jax_ppo --backend warp --preset full \
    --n-envs 16384 --n-steps 8 --batch-size 16384 --timesteps 20000000
```

On a GTX 1080 the Warp env steps 3.9M env steps/s at 16,384 envs, 5.8x the
JAX env, and PPO on it runs 363k steps/s, 1.5x PPO on the JAX env. PyPI's
`warp-lang` is built with CUDA 13, which no longer supports Pascal GPUs, so
the `warp` group installs the CUDA 12 build from Warp's GitHub release. The
kernels compile with `fuse_fp` off, for the same rounding reason as the JAX
and C++ ports; the first build of each configuration takes about a minute
(Warp caches it in `~/.cache/warp`).

## License

[MIT](LICENSE)
