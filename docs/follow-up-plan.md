# Follow-up Plan: MAPPO and the Open Backlog

Started 2026-10-11. This plan takes over the seven open items from the [acceleration plan](acceleration-plan.md), which is now complete.

## Summary

Seven items remain: two speed-ups, one packaging change, three steps to multi-agent PPO (MAPPO), and a long training run. Each is one session and one PR, in the order below. The run goes last, so its 40 minutes are spent once, on the best setup found.

| # | Item | Kind | Done when |
| --: | --- | --- | --- |
| 1 | Packaging with scikit-build-core | Build | `uv sync` builds the C++ module; `ELEVATOR_NO_CPP=1` installs pure Python |
| 2 | Cheaper arrivals in the JAX env | Speed | Arrivals under 15% of a JAX env step (now 55–59%), parity tests green |
| 3 | Smaller observation or a shared per-lift encoder | Speed | Faster PPO epochs with reward within noise of 77.0 at 50M steps |
| 4 | Multi-agent env interface | MARL | Per-lift views, global state and team reward on the JAX and Warp envs; the Mava install question answered |
| 5 | Our own MAPPO in `jax_ppo` | MARL | MAPPO vs PPO on `full`, at equal steps and equal time, 3 seeds; transfer across lift counts measured |
| 6 | MAPPO with Mava | MARL | Mava's `ff_mappo` trains on our env and checks item 5's result |
| 7 | 200M-step run | Reward | The best configuration from items 3–6 run for 200M steps, 3 seeds |

Items 1 and 2 stand alone. Item 3's shared encoder is the same network as item 5's actor, so it comes first and gives MAPPO a centralised baseline built from the same parts. Item 4 starts with a short check of Mava's dependencies, so item 6's main risk is known before items 5 and 6 are built.

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

- [ ] Switch `[build-system]` to scikit-build-core with nanobind as a build requirement; install `_cpp` and `libelevator_c.so` into `elevator_rl/`.
- [ ] Set `tool.uv.cache-keys` to the `cpp/` sources and `CMakeLists.txt`, so uv rebuilds when the C++ changes. Without it uv reuses a stale build.
- [ ] Decide on editable rebuilds (`editable.rebuild`) against a plain reinstall after C++ edits, and write the answer in the README.
- [ ] CI: let `uv sync` build the module, and keep `cmake` + `ctest` only for the C++ unit tests.
- [ ] Keep `-ffp-contract=off`: parity depends on it.
- [ ] Test the fallback: `ELEVATOR_NO_CPP=1 uv sync` gives a working package whose C++ tests skip.

**Risks.**

- **The EnvPool wheel.** This item was skipped overnight because it could break the EnvPool wheel. That wheel is not in `uv.lock`, so every `uv sync` removes it, and a sync that now also rebuilds the C++ makes that happen more often. Options: an `envpool` dependency group with `find-links` to `cpp/build/envpool-dist/` (but then the lock needs the wheel to exist), or documenting `uv sync --inexact`. Choose one and test that the EnvPool tests still run after a sync.
- **macOS.** `required-environments` includes Intel macOS, which has no CUDA and may lack a compiler. There the build must fall back, not fail.

## 2. Cheaper arrivals in the JAX env

**Goal.** Cut the arrival draw from 55–59% of a JAX env step. It draws a destination for 16 slots on each of 30 floors by Gumbel-max over 30 floors: about 14,400 random numbers per env per step for about 3 arrivals.

**Approach: draw the total, then place each arrival.** Independent Poisson counts per floor are the same, in distribution, as one Poisson total N ~ Poisson(Σ rates) with each arrival's origin drawn in proportion to the rates (Poisson splitting). So:

1. Draw N, capped at K slots (say 32).
2. For each of the K slots, draw an origin from the rates and a destination from that origin's row of `joint`, by inverse CDF: a uniform and a `searchsorted` over a cumulative sum, not Gumbel-max.
3. Scatter the slots into the floor queues in slot order, keeping today's caps (16 per floor per step, 512 per queue) and the `dropped` count.

That is about 2K = 64 uniforms per env per step instead of 14,400 Gumbel draws. The `uniform` traffic path is already cheap and stays as it is.

- [ ] Measure the largest arrival rate on `full` (morning peak) and the chance N exceeds K; choose K so it is negligible, and count overflows in `dropped`.
- [ ] Keep exact parity: the parity tests inject Python's arrivals, so they don't depend on the draw. Run them unchanged.
- [ ] Add or extend a statistical test: per-floor count means and the destination histogram against the Python env's, over many steps.
- [ ] Re-run `scripts/profile_jax_env.py` and `scripts/time_jax_env.py`: arrivals' share, env steps/s at 1,024 and 16,384 envs.
- [ ] One reward-parity run on the JAX env (`scripts/reward_parity.py`, 3 seeds) to confirm learning is unchanged.

**Note.** The Warp env already draws only the arrivals that happen, so this speeds up the JAX env, not the Warp training path.

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

**Goal.** A functional multi-agent API over the JAX and Warp envs that both MAPPO routes can use, plus a decision on how to run Mava.

**Interface.** Mava's env API is Jumanji's (`reset(key) -> (state, timestep)`, `step(state, action) -> (state, timestep)`, the key kept in the state) plus `num_agents`, `time_limit` and `action_dim`. Its observation is a named tuple with an agents axis:

| Field | Shape | From our env |
| --- | --- | --- |
| `agents_view` | (lifts, obs) | the `relative` observation reshaped to (4, 305) |
| `action_mask` | (lifts, actions) | all true in step mode; target mode the same unless a goal must be kept |
| `global_state` | (lifts, state) | the full 1,220-value observation, repeated for each lift |
| `step_count` | (lifts,) | `state.steps` |

Our JAX env already has this shape: `step(state, actions)` is pure, the PRNG key lives in `EnvState`, and the observation is per-lift blocks. So we use Mava's layout as our own and add no new abstraction. One `MultiAgentEnv` wrapper over `JaxBuildingEnv` returns these arrays, the team reward repeated per lift, and `done`; a batched version wraps the Warp env's vectorised step for our own MAPPO. The per-lift block is the same size whatever the lift count (its "other lifts" field is scaled by lifts − 1), so one policy can run in buildings with 2, 4 or 6 lifts.

PettingZoo is not the core: it is stateful, steps one env at a time with dicts keyed by agent, and can't run inside `jit`. A `ParallelEnv` adapter over the Python or C++ env is an optional extra, for RLlib or BenchMARL.

- [ ] **Spike first: can Mava run here?** Mava is meant to be cloned and run, not installed as a library. It pins `jax==0.5.3`, `numpy==1.26.4`, `gymnasium==1.0.0`, `flax<0.10.4` and Python below 3.13, and pulls several packages from git (a JaxMARL fork, gigastep, smaclite). We run JAX 0.11.2 and NumPy 2. Try a separate venv with Mava's pins plus our package installed without dependencies, and check that `jax_env` imports and runs on JAX 0.5.3 with the CUDA 12 plugin on the GTX 1080. If not, the fallback is a Mava fork with its pins raised, which is a bigger job. Record the answer before item 5.
- [ ] Write the wrapper and its `observation_spec` and `action_spec`, following Mava's `mava/wrappers/jumanji.py`.
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

## 6. MAPPO with Mava

**Goal.** Train Mava's MAPPO on our env as an outside check on item 5. If both agree, the result does not come from a bug in our own code.

- [ ] Using item 4's spike, register our env in Mava's `make_env` and add a Hydra env config, then run `mava/systems/ppo/anakin/ff_mappo.py`.
- [ ] Match settings to item 5 where Mava exposes them: envs, rollout length, epochs, minibatches, learning rate, clip, network size.
- [ ] **The warm start.** Mava has none, and `full` learns slowly from scratch. Either load our cloned actor's weights into Mava's network, or compare both MAPPOs from scratch on an easier preset (`hall_calls` or `kinematic`) as well as on `full`.
- [ ] Optional: `rec_mappo` (recurrent), since a lift may gain from remembering recent calls.
- [ ] Report reward at equal steps against item 5 (3 seeds) and steps/s on the JAX env. Mava probably can't use the Warp env: it vmaps a single env, and the Warp kernel is already batched.

## 7. 200M-step run

**Goal.** Learn where training levels off. Tuned PPO was still rising at 77.0 after 50M steps.

- [ ] Run the best configuration from items 3–6, and tuned PPO if that is a different setup, for 200M steps on 3 seeds. At about 88k PPO steps/s that is about 40 minutes per seed, less after items 2 and 3.
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
