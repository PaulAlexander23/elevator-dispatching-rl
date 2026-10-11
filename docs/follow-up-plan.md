# Follow-up Plan: MAPPO and the Open Backlog

Started 2026-10-11. This plan takes over the open items from the [acceleration plan](acceleration-plan.md), which is now complete.

## Summary

Six items remain: two speed-ups, one packaging change, two steps to multi-agent PPO (MAPPO), and a long training run. MAPPO with Mava is deferred to the backlog: its pinned dependencies clash with ours (see Backlog). Each is one session and one PR, in the order below. The run goes last, so its 40 minutes are spent once, on the best setup found.

| # | Item | Kind | Done when |
| --: | --- | --- | --- |
| 1 | Packaging with scikit-build-core | Build | `uv sync` builds the C++ module; `ELEVATOR_NO_CPP=1` installs pure Python |
| 2 | Cheaper arrivals in the JAX env | Speed | Arrivals under 15% of a JAX env step (now 55–59%), parity tests green |
| 3 | Smaller observation or a shared per-lift encoder | Speed | Faster PPO epochs with reward within noise of 77.0 at 50M steps |
| 4 | Multi-agent env interface | MARL | Per-lift views, global state and team reward on the JAX and Warp envs |
| 5 | Our own MAPPO in `jax_ppo` | MARL | MAPPO vs PPO on `full`, at equal steps and equal time, 3 seeds; transfer across lift counts measured |
| 6 | 200M-step run | Reward | The best configuration from items 3–5 run for 200M steps, 3 seeds |

Items 1 and 2 stand alone. Item 3's shared encoder is the same network as item 5's actor, so it comes first and gives MAPPO a centralised baseline built from the same parts. Item 4 keeps Mava's data layout, so adding Mava later needs only a thin wrapper.

## Baselines

Every result below is compared with these, on `full` with the Warp env and the GTX 1080 unless stated.

| Baseline | Value | Source |
| --- | --- | --- |
| Heuristic (collective control) | 69 passengers per episode | acceleration plan, "Longer runs" |
| Tuned PPO, 1,024 × 8 : 2,048, 10 epochs, lr 5.2e-5, 256 × 256 | 77.0 ± 2.4 after 50M steps, 9.5 minutes, 3 seeds | acceleration plan, "Longer runs" |
| The same, 20M steps | 73.6 ± 2.0 | acceleration plan, network size table |
| Tuned PPO speed | about 88k PPO steps/s (10 epochs); 230k at 3 epochs | `scripts/profile_jax_ppo.py` |
| JAX env step | 2.32 ms at 1,024 envs, 28.2 ms at 16,384; arrivals 55–59% | `scripts/profile_jax_env.py` |
| Warp env | about 3.9M env steps/s | `scripts/time_jax_env.py` |

Reward claims need 3 seeds: the 50M-step runs spread from 74.4 to 79.2, so differences of a point or two are noise.

## 1. Packaging with scikit-build-core

**Goal.** `uv sync` compiles the nanobind module (`elevator_rl._cpp`) and the C library (`libelevator_c.so`), so nobody copies built files into `src/elevator_rl/` by hand. Machines without a compiler still get the pure-Python package.

**Approach.** Replace hatchling with scikit-build-core as the build backend, building `cpp/CMakeLists.txt`. An override turns CMake off when `ELEVATOR_NO_CPP=1` is set (`[[tool.scikit-build.overrides]]` with `if.env`, setting `wheel.cmake = false`). `cpp_env.py` and `ctypes_env.py` already fall back when the module is missing.

- [x] Switch `[build-system]` to scikit-build-core with nanobind as a build requirement; install `_cpp` and `libelevator_c.so` into `elevator_rl/`.
- [x] Set `tool.uv.cache-keys` to the `cpp/` sources and `CMakeLists.txt`, so uv rebuilds when the C++ changes. Without it uv reuses a stale build.
- [x] Decide on editable rebuilds (`editable.rebuild`) against a plain reinstall after C++ edits, and write the answer in the README.
- [x] CI: let `uv sync` build the module, and keep `cmake` + `ctest` only for the C++ unit tests.
- [x] Keep `-ffp-contract=off`: parity depends on it.
- [x] Test the fallback: `ELEVATOR_NO_CPP=1 uv sync` gives a working package whose C++ tests skip.

**Risks.**

- **The EnvPool wheel.** This item was skipped overnight because it could break the EnvPool wheel. That wheel is not in `uv.lock`, so every `uv sync` removes it, and a sync that now also rebuilds the C++ makes that happen more often. Options: an `envpool` dependency group with `find-links` to `cpp/build/envpool-dist/` (but then the lock needs the wheel to exist), or documenting `uv sync --inexact`. Choose one and test that the EnvPool tests still run after a sync.
- **macOS.** `required-environments` includes Intel macOS, which has no CUDA and may lack a compiler. There the build must fall back, not fail.

### Result: done

`uv sync` now builds the C++ into the package in about 5 seconds, and the next `uv run` rebuilds it after any change to `cpp/`. All 235 tests pass on it, the C++ and ctypes parity tests included. Nobody builds or copies `.so` files by hand any more.

What was built:

- **Build backend.** scikit-build-core with `cmake.source-dir = "cpp"`, nanobind in the build environment, and a persistent build directory in `cpp/build/skbuild/`. The CMake file installs `_cpp` and `libelevator_c` into the wheel's `elevator_rl/` when `SKBUILD` is set, and skips the tools and C++ tests there (`ELEVATOR_TOOLS`).
- **Two fallbacks to pure Python,** both tested: `ELEVATOR_NO_CPP=1`, and any failed C++ build (an `if.failed` override; tested with a compiler that doesn't exist). Both give a `py3-none-any` wheel, and with it the C++ tests skip (50 passed, 33 skipped in the C++ and building tests).
- **Rebuilds.** `tool.uv.cache-keys` lists `cpp/src`, `cpp/include`, `cpp/capi`, `cpp/python`, `cpp/CMakeLists.txt` and `ELEVATOR_NO_CPP`. A `touch` of `cpp/src/building.cpp` made the next `uv run` rebuild. `editable.rebuild` was not needed: uv's cache keys do the same job, without CMake running on import.
- **Hand builds** of `cpp/` are now only for the replay and benchmark tools and the C++ tests. Their copies of the modules stay in `cpp/build/elevator_rl/`, not `src/`.
- **CI** builds the module through `uv sync`, then asserts it is there, since a failed build would otherwise fall back quietly and the C++ tests would skip.
- **The sdist** (400 KB) builds a wheel with both libraries.

What we learned:

- **An editable install splits the package.** scikit-build-core's editable mode serves the Python files from `src/` and the built files from `site-packages`, and the package's `__path__` lists both. `import elevator_rl._cpp` found the module, but `ctypes_env` looked only next to its own file, so it now searches `__path__`. A `.so` left in `src/` by an old hand build would have been found first, possibly stale. That is why hand builds no longer write there.
- **EnvPool was not at risk.** `uv run` rebuilds the C++ without removing the EnvPool wheel (checked after a rebuild), and `uv sync --inexact` keeps it too. Only a plain `uv sync` removes it, as before. The wheel stays out of the lock, because CI can't build it.
- **`wheel.exclude` matches the built files as well,** not only the sources. Excluding the hand-build copies in `src/` dropped both libraries from the wheel. It wasn't needed anyway: scikit-build-core already leaves out gitignored files.
- **Untested:** macOS. Without a compiler there, the `if.failed` override should fall back to pure Python.

## 2. Cheaper arrivals in the JAX env

**Goal.** Cut the arrival draw from 55–59% of a JAX env step. It draws a destination for 16 slots on each of 30 floors by Gumbel-max over 30 floors: about 14,400 random numbers per env per step for about 3 arrivals.

**Approach: draw the total, then place each arrival.** Independent Poisson counts per floor are the same, in distribution, as one Poisson total N ~ Poisson(Σ rates) with each arrival's origin drawn in proportion to the rates (Poisson splitting). So:

1. Draw N, capped at K slots (say 32).
2. For each of the K slots, draw an origin from the rates and a destination from that origin's row of `joint`, by inverse CDF: a uniform and a `searchsorted` over a cumulative sum, not Gumbel-max.
3. Scatter the slots into the floor queues in slot order, keeping today's caps (16 per floor per step, 512 per queue) and the `dropped` count.

That is about 2K = 64 uniforms per env per step instead of 14,400 Gumbel draws. The `uniform` traffic path is already cheap and stays as it is.

- [x] Measure the largest arrival rate on `full` (morning peak) and the chance N exceeds K; choose K so it is negligible, and count overflows in `dropped`.
- [x] Keep exact parity: the parity tests inject Python's arrivals, so they don't depend on the draw. Run them unchanged.
- [x] Add or extend a statistical test: per-floor count means and the destination histogram against the Python env's, over many steps.
- [ ] (CPU done; GPU waits for a reboot) Re-run `scripts/profile_jax_env.py` and `scripts/time_jax_env.py`: arrivals' share, env steps/s at 1,024 and 16,384 envs.
- [ ] (waits for the GPU) One reward-parity run on the JAX env (`scripts/reward_parity.py`, 3 seeds) to confirm learning is unchanged.

**Note.** The Warp env already draws only the arrivals that happen, so this speeds up the JAX env, not the Warp training path.

### Result: done on the CPU; GPU numbers to follow

`JaxBuildingEnv.draw_arrivals` now draws a Poisson total and places each arrival by inverse CDF, as planned. Details:

- **Slots.** The expected total is `arrival_probability × n_floors` whatever the traffic mix (each pattern's origins sum to one, as do the mix weights), so it is 3.0 a step on `full` at every point of the day. `arrival_slots(config)` takes the mean plus 10 standard deviations plus 10, rounded up to a multiple of 8: 32 on `full`. Poisson(3) exceeds 32 about once in 1e23 steps; any excess goes in a new `Arrivals.overflow` field and is counted in `dropped`.
- **Layout.** The slots are laid out in the old per-floor `Arrivals` form (floors × 16, in slot order), so `add_arrivals`, its caps and the parity tests' injected arrivals are unchanged. A slot's place in its floor's row is the number of earlier slots from the same floor, a 32 × 32 comparison.
- **Inverse CDF.** `_inverse_cdf` never returns a zero-weight index, even if rounding puts a uniform at the very end of the cumulative sum, so no passenger is sent to their own floor.
- **Tests.** The parity tests pass unchanged. The new `test_arrivals_match_python_in_distribution` (three points of the day, 20,000 draws) checks per-floor count means and variances against Python's rates, and a chi-square on the (origin, destination) histogram, for both the JAX and the Python draws. It fails if the destination row is off by one floor or the total rate is 5% too high. Full suite: 238 passed.

CPU profile (`scripts/profile_jax_env.py --n-envs 1024`, `full`, `relative`; the GPU was unavailable, see below):

| | Arrivals, ms per batched step | Share of step | Step, ms | Env steps/s |
| --- | --: | --: | --: | --: |
| Before | 212.2 | 91% | 234.4 | 4,370 |
| After | 7.6 | 14% | 54.0 | 18,954 |

Arrivals are 28× cheaper and the step 4.3× faster on the CPU; the goal (under 15%) is met there. On the CPU the actions and substeps are now 79% of the step.

**Still to do once the GPU is back.** On 2026-10-11 the NVIDIA driver's user-space library (580.178) was newer than the loaded kernel module (580.173), so JAX fell back to the CPU until a reboot. Still to run: the GPU profile at 1,024 and 16,384 envs (`profile_jax_env.py`, `time_jax_env.py`) and the reward-parity run.

## 3. Smaller observation or a shared per-lift encoder

**Goal.** Make the compute-bound PPO epochs cheaper without losing reward. The `relative` observation has 1,220 inputs and the first layers hold 82% of the weights. Shrinking the hidden layers is not the answer: 64 × 64 networks lost 7 passengers.

The `relative` observation is already four per-lift blocks of 305 values: a head of 10 (the lift's own state) and a window of 59 floor offsets × 5 fields (in the building, passengers wanting that floor, waiting up, waiting down, other lifts there). Two candidates use this:

- **A shared per-lift encoder.** One MLP (305 → 128) applied to each lift's block with shared weights, the four outputs concatenated (512) and passed to the existing 256 layer. The first layer drops from 312k weights to 39k and from 312k to 156k multiply-adds per sample. This is the same encoder item 5's actor will use.
- **A narrower window.** Keep ±k floors at full resolution and pool the floors beyond into a few coarse bins (for example 1, 2, 4, 8 floors wide). With k = 8, each block shrinks from 305 to about 140 values.

- [ ] Implement the shared encoder as an option of the JAX PPO's networks (`net_arch` stays the hidden layers after it).
- [ ] Implement the narrower window as an observation option in the JAX and Warp envs, with parity against Python, if the encoder alone doesn't reach the goal.
- [ ] The imitation warm start must clone into the new network, so check `jax_imitate` works with it.
- [ ] Compare at 20M steps (3 seeds, tuned settings) against 73.6 ± 2.0, then the best at 50M against 77.0 ± 2.4. Report PPO steps/s and `profile_jax_ppo.py`'s epoch share.

## 4. Multi-agent env interface

**Goal.** A functional multi-agent API over the JAX and Warp envs that our MAPPO uses, laid out as Mava expects so Mava can be added later.

**Interface.** Mava's env API is Jumanji's (`reset(key) -> (state, timestep)`, `step(state, action) -> (state, timestep)`, the key kept in the state) plus `num_agents`, `time_limit` and `action_dim`. Its observation is a named tuple with an agents axis:

| Field | Shape | From our env |
| --- | --- | --- |
| `agents_view` | (lifts, obs) | the `relative` observation reshaped to (4, 305) |
| `action_mask` | (lifts, actions) | all true in step mode; target mode the same unless a goal must be kept |
| `global_state` | (lifts, state) | the full 1,220-value observation, repeated for each lift |
| `step_count` | (lifts,) | `state.steps` |

Our JAX env already has this shape: `step(state, actions)` is pure, the PRNG key lives in `EnvState`, and the observation is per-lift blocks. So we use Mava's layout as our own and add no new abstraction, which also keeps the deferred Mava item cheap. One `MultiAgentEnv` wrapper over `JaxBuildingEnv` returns these arrays, the team reward repeated per lift, and `done`; a batched version wraps the Warp env's vectorised step for our own MAPPO. The per-lift block is the same size whatever the lift count (its "other lifts" field is scaled by lifts − 1), so one policy can run in buildings with 2, 4 or 6 lifts.

PettingZoo is not the core: it is stateful, steps one env at a time with dicts keyed by agent, and can't run inside `jit`. A `ParallelEnv` adapter over the Python or C++ env is an optional extra, for RLlib or BenchMARL.

- [ ] Write the wrapper and its `observation_spec` and `action_spec`, following the layout of Mava's `mava/wrappers/jumanji.py`, without depending on Mava.
- [ ] Tests: per-lift views reassemble the single-agent observation exactly, rewards and dones match the single-agent env, and the wrapper runs under `jit` and `vmap`.
- [ ] Optional: the PettingZoo adapter and its `parallel_api_test`.

## 5. Our own MAPPO in `jax_ppo`

**Goal.** Find out whether a policy shared across lifts, each seeing its own view, beats today's PPO. Today's PPO is already centralised: one network over the full observation with one action head per lift.

**Design.**

- **Actor:** item 3's per-lift encoder plus a head for one lift, applied to each lift's view with shared weights, plus a lift id (one-hot) in its input. Action log-probabilities sum over lifts, as now.
- **Critic:** on the global state, the full observation, as today's value network.
- **Loss:** the PPO loss, the tuned settings, the same Hydra config tree (`algo=mappo`). Both the JAX and Warp envs work.
- **Warm start:** `full` learns only from the imitation warm start, so clone the heuristic's per-lift actions into the shared actor first.

- [ ] Implement the actor, critic and config switch; test the per-lift log-probabilities against a lift-by-lift reference.
- [ ] Compare on `full` at 20M and 50M steps and at equal wall-clock time (3 seeds each) against tuned PPO and against item 3's shared-encoder PPO. That separates the effect of sharing from the effect of a centralised head.
- [ ] Transfer: train with 4 lifts, then evaluate with no further training on 2, 3, 6 and 8 lifts against the heuristic in each building. Then fine-tune briefly and compare with training from scratch.
- [ ] Without the lift id as well, to see if the lifts need to know which one they are.

## 6. 200M-step run

**Goal.** Learn where training levels off. Tuned PPO was still rising at 77.0 after 50M steps.

- [ ] Run the best configuration from items 3–5, and tuned PPO if that is a different setup, for 200M steps on 3 seeds. At about 88k PPO steps/s that is about 40 minutes per seed, less after items 2 and 3.
- [ ] Run GPU jobs one at a time: each JAX process takes 75% of the GPU's memory.
- [ ] Plot reward against steps and against minutes next to the 50M runs, and report the final level and the best eval.

## Pitfalls carried over

From the acceleration plan and its sessions:

- **Measure, then change.** Profile before and after each speed-up and report the share of time as well as the speed.
- **Exact parity first.** Any change to an env (items 2 and 3) keeps the step-for-step parity tests against Python green, in float64.
- **Warp miscompiles.** Read a closure constant inside a `wp.static` branch through `wp.static(...)`, and give each variable one type across branches. Run the Warp tests with `JAX_PLATFORMS=cpu` for fast compiles.
- **Long jobs and branches.** A background run imports from the working tree, so use a `git worktree` (copy the built `.so` files into it) rather than switching branches under it. Item 1 removes the copying.

## Backlog

Items deferred while working through this plan go here.

- [ ] **MAPPO with Mava (`id-mava`),** deferred on 2026-10-11 because of its dependencies. Train Mava's `ff_mappo` on our env as an outside check on item 5: if both agree, the result does not come from a bug in our own code. The complications:
    - **Pinned, old dependencies.** Mava pins `jax==0.5.3`, `numpy==1.26.4`, `gymnasium==1.0.0`, `scipy==1.12.0`, `flax<0.10.4` and Python below 3.13. We run JAX 0.11.2 and NumPy 2, and the Warp env needs our JAX. It can't share our venv.
    - **Not a library.** Mava is meant to be cloned and run as research code, and it installs several packages from git (a JaxMARL fork, gigastep, smaclite) plus MuJoCo and Brax.
    - **The GPU is unchecked.** Whether JAX 0.5.3's CUDA 12 plugin runs on the GTX 1080 (Pascal) is unknown.
    - **No warm start.** `full` learns only from the imitation warm start, which Mava lacks. We would load our cloned actor into Mava's network, or compare both MAPPOs from scratch on an easier preset (`hall_calls` or `kinematic`).
    - **JAX env only.** Mava vmaps a single env, and the Warp kernel is already batched, so Mava would probably train on the slower JAX env.

    To pick it up, start with a spike: a separate venv with Mava's pins, our package installed without dependencies, and a check that `jax_env` runs on JAX 0.5.3 on the GPU. If it does, register our env in Mava's `make_env` with item 4's wrapper and run `mava/systems/ppo/anakin/ff_mappo.py` at item 5's settings (3 seeds). If not, the fallback is a Mava fork with its pins raised, a bigger job. `rec_mappo` (recurrent) is an optional extra.
