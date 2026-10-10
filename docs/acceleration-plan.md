# Elevator Env Acceleration Plan

Last updated 2026-10-10. The live version of this plan is a Claude Docs doc; this is the repo's copy, with each chart turned into a table of its data.

## Summary

End-to-end PPO training on the heaviest preset (`full`) is now 209× faster than at the start: 362,635 PPO steps/s, up from 1,739. The start was SB3 with Python envs on one CPU thread; the end is the JAX PPO training on the Warp env, compiled into one program on a GTX 1080. With matching settings, the JAX PPO learns as well as SB3. Making the env faster gave only the first 1.7×. After the C++ port, the learner was the bottleneck, and the big gains came from moving the whole training loop onto the GPU.

**End-to-end PPO training on `full` is 209× faster than where it started.** PPO steps/s, preset full, relative obs, 256×256 MLPs, 3 epochs; 64 envs × 64 steps and minibatch 64 unless stated. EnvPool (phase 3) is not shown: the env was already 1% of SB3's training time, so it could not move this number.

| Stage | Phase | PPO steps/s | vs start |
|---|---:|---:|---:|
| Python envs (DummyVecEnv), 1 CPU thread | Phase 0 | 1,739 | 1.0× |
| C++ VecEnv, 1 CPU thread | Phase 2 | 2,886 | 1.7× |
| C++ VecEnv, 8 CPU threads, minibatch 2048 | Learner | 19,156 | 11× |
| C++ VecEnv, GPU, 1024 envs, minibatch 2048 | Learner | 50,230 | 29× |
| JAX env + PPO, GPU, 1024 envs, minibatch 2048 | Phase 4 | 184,902 | 106× |
| JAX env + PPO, GPU, 16,384 envs | Phase 4 | 238,109 | 137× |
| Warp env + JAX PPO, GPU, 16,384 envs | Phase 4 | 362,635 | 209× |

_Source: `elevator_rl.sweep` (SB3) and `scripts/time_jax_ppo.py` (JAX), Ryzen 7 3700X + GTX 1080._

Every phase is built:

1. **Heavier env (phase 0):** 30 floors, 4 lifts, kinematics, hall calls and a day of traffic, switched on one at a time by presets. PPO learns `full` only with the lift-relative observation and a behaviour-cloning warm start; it reaches 66.8 passengers per episode, against the heuristic's 69.
2. **C++ core (phase 1):** matches Python exactly, step for step, and is 70–120× faster per step.
3. **Python binding (phase 2):** a batched nanobind VecEnv. It cut the env from 40% to 1% of training time and made training 1.7× faster.
4. **EnvPool (phase 3):** 1.9–2.9× the C++ env's throughput with 16 threads, but no gain in SB3 training, as gate 2 predicted.
5. **JAX (phase 4):** 2.6–7× SB3's speed on the same GPU. The Warp env then added 1.5×, reaching 363k PPO steps/s.

Success still means a faster training loop with no loss in policy quality. Policy quality at the fastest settings is not yet checked; it is first in the backlog.

## Current state

The repo now has five implementations of the env, each checked against the Python one, and two of PPO. All of it is on the `heavier-env` branch.

| Component | Where | Role | Speed on `full` |
| --- | --- | --- | --- |
| Python envs | `building.py`, `building_env.py` (the original `LiftEnv` in `sim.py`, `env.py`) | The reference: every feature, observation and action mode | about 6.5k env steps/s (`DummyVecEnv`) |
| C++ port | `cpp/`, module `elevator_rl._cpp` (nanobind) | Exact parity with Python; `CppVecEnv` steps a batch in one call | about 0.5M env steps/s, 1 thread |
| EnvPool | `cpp/envpool/`, `scripts/build_envpool.sh`, `EnvPoolVecEnv` | The C++ env on EnvPool's thread pool, as a trimmed wheel built with Bazel | up to 1.5M env steps/s, 16 threads |
| JAX env | `jax_env.py` | Pure-function port for `jit` and `vmap`; exact parity in float64 | about 0.67M env steps/s on the GPU |
| Warp env | `warp_env.py` | One GPU thread per env, run inside jax.jit; exact parity in float64 | about 3.9M env steps/s on the GPU |
| SB3 PPO | `train.py`, `compare.py`, `sweep.py` | Training on any env backend, on the CPU or the GPU | up to 50k PPO steps/s |
| JAX PPO | `jax_ppo.py` | PPO matching SB3's, compiled into one program with the env | up to 363k PPO steps/s (on the Warp env) |

The tests (149, plus slow ones) check every port against Python, and CI builds the C++ and runs the JAX tests. The optional parts are dependency groups: CUDA torch (`cuda`) and JAX (`jax`, `jax-cuda`). EnvPool is a local wheel and needs Python 3.12 or later, which the project now uses.

## Phase 0: heavier environment

Before porting anything, make the env worth accelerating. It is now implemented as `BuildingEnv`, with a `BuildingConfig` that switches each feature on or off and one central controller choosing an action per lift. The original `LiftSim`/`LiftEnv` stay untouched as the reference. `BuildingEnv`'s default config reproduces them step for step with the same seed, and a test enforces this.

Each preset adds one feature to the previous one, so each sub-phase has its own measured cost. Figures are from this machine (16 cores, CPU, 1 env, short 8,192-step PPO runs), so treat the PPO column as rough.

| Sub-phase | Preset | Adds | Env µs/step | PPO steps/s | Env share of PPO time |
| --- | --- | --- | --: | --: | --: |
| 0a | `original` | Reference config (10 floors, 1 lift) | 16 | 1,470 | 2.3% |
| 0b | `tall` | 30 floors | 31 | 920 | 2.8% |
| 0c | `multi` | 4 lifts, `MultiDiscrete` central controller | 40 | 410 | 1.6% |
| 0d | `kinematic` | Acceleration, braking, door time, 10 substeps per 1 s step | 59 | 410 | 2.4% |
| 0e | `hall_calls` | Up/down buttons; serve up / serve down actions | 79 | 360 | 2.8% |
| 0f | `full` | `day` traffic: up-peak → lunch → down-peak, Poisson arrivals | 164 | 340 | 5.6% |

The env became 10x more expensive, but PPO slowed about 4x too, because the obs grows to 196 values and the action space to 4 lifts. So the env is still only about 6% of training time. The likely remedies are on the trainer side: more parallel envs, larger batches, or the end-to-end JAX route. Gate 2 should be judged with those in place.

For each sub-phase:

- [ ] Tests pass (`tests/test_building.py`; parity, kinematics, hall calls, traffic, lift ordering)
- [ ] Measure: `scripts/time_env.py --preset <name>` and `scripts/time_ppo.py --preset <name>`
- [ ] Train briefly (`python -m elevator_rl.train --preset <name>`) and check the policy beats random
- [ ] Note which code dominates the step: per-lift Python loops for 0c–0e, the NumPy arrival model for 0f

Design choices worth knowing before porting:

- Lifts act in index order within a step, so lift 0 wins boarding conflicts.
- With kinematics, a lift ignores actions while moving or with its doors open. The exception is a same-direction move, which extends the trip by one floor while the target is the next floor ahead. A bang-bang profile with exact braking stops each lift on the floor.
- Non-uniform traffic uses Poisson counts per floor (a busy lobby can get several arrivals a step). `uniform` keeps the original Bernoulli draws for parity.
- Queues are still unbounded Python deques. Choosing a cap remains a Phase 1 decision.

### Sweep: parallel envs and batch size

Parallelising the trainer gives the first big win, about 9–12x, before any C++. It also makes the env worth porting. With 64 in-process envs and batch 512, PPO reaches 3,254 steps/s and env stepping is 54% of the time. At 1 env it was 9%.

**With 64 envs, env stepping becomes the largest share of PPO time.** Share of PPO wall-clock time, preset full, minibatch 512, 2,048-step rollout. Gate 2's threshold is an env share over 10%.

| VecEnv | Envs | Env stepping | Policy forward + SB3 | Gradient update | PPO steps/s |
|---|---:|---:|---:|---:|---:|
| dummy | 1 | 8.6% | 88.4% | 3.0% | 376 |
| dummy | 4 | 19.2% | 72.4% | 8.4% | 1,043 |
| dummy | 16 | 38.9% | 42.7% | 18.4% | 2,282 |
| dummy | 64 | 54.2% | 19.8% | 26.0% | 3,254 |
| subproc | 1 | 12.3% | 84.9% | 2.9% | 348 |
| subproc | 4 | 13.0% | 78.2% | 8.8% | 1,036 |
| subproc | 16 | 22.3% | 56.3% | 21.3% | 2,517 |
| subproc | 64 | 31.6% | 31.5% | 36.9% | 4,326 |

_Source: `elevator_rl.sweep`, preset full, 30,000 timed steps × 2 seeds per config, 1 torch thread._

- **Gate 2 is met.** Env share is above 10% in every parallel configuration. A batched C++ VecEnv in the `dummy`/64 setup could at most roughly double throughput (1 / (1 − 0.54) ≈ 2.2x).
- **SubprocVecEnv is fastest today (4,326 steps/s)** because it steps envs in parallel. But it pays inter-process overhead on every step. An in-process, multithreaded C++ batch (Phase 2–3) removes both the Python step cost and that overhead.
- **Batch 512 vs 64** cuts update time: 12 instead of 96 gradient steps per rollout. That changes learning dynamics, not just speed, so check reward per sample before adopting it.
- **64 envs × 2,048-step rollout** leaves only 32 steps per env per rollout, against 200-step episodes. Compare learning curves before treating this as the training default.

Reproduce with `uv run python -m elevator_rl.sweep --preset full --n-envs 1 4 16 64 --batch-sizes 64 512 --vec-envs dummy subproc --repeats 2`.

### Reward comparison: does parallelism hurt learning?

More envs do not hurt learning, but bigger minibatches do. On `original`, 64 envs × 32 steps with minibatch 64 matched or beat the 1-env baseline per env step (final reward 27.2 ± 2.3 vs 24.9 ± 1.2). It got there in 1.8 minutes instead of 6.4.

**64 envs with minibatch 64 learns as well as 1 env, 3.6× sooner.** Passengers delivered per episode at the last evaluation, preset original, mean of 3 seeds. Settings are envs × steps per env : minibatch. A random policy scores 18.6 on this preset.

| Setting | Final reward | Training minutes |
|---|---:|---:|
| 1 × 2048 : 64 (baseline) | 27.2 | 6.4 |
| 64 × 32 : 64 | 31.0 | 1.8 |
| 16 × 128 : 256 | 18.2 | 1.5 |
| 64 × 128 : 512 | 14.9 | 0.9 |
| 64 × 32 : 512 | 12.3 | 0.9 |

_Source: `elevator_rl.compare`, 500,000 env steps × 3 seeds, deterministic evaluation of 10 episodes every 25,000 steps, evaluation time excluded._

| Setting (envs × steps : batch) | Final reward | PPO steps/s | Minutes for 500k steps |
| --- | --: | --: | --: |
| 1×2048 : 64 (current default) | 24.9 ± 1.2 | 1,310 | 6.4 |
| 64×32 : 64 | **27.2 ± 2.3** | 4,667 | 1.8 |
| 16×128 : 256 | 17.8 ± 1.3 | 5,439 | 1.5 |
| 64×32 : 512 | 13.0 ± 2.4 | 9,191 | 0.9 |
| 64×128 : 512 | 13.6 ± 1.6 | 9,369 | 0.9 |

Final reward is the mean of the last 20% of evaluations, ± the standard deviation across 3 seeds. A random policy scores 18.6.

- **Large minibatches learn less from the same data.** Batch 512 makes 8x fewer gradient steps per rollout. After 500k steps those runs sit below the random policy, so their 2x throughput is not worth it.
- **Short rollouts per env (32 steps) were fine.** 64×128 : 512 learned no better than 64×32 : 512, so batch size, not horizon, is the driver.
- **Suggested training default:** 64 envs × 32 steps, minibatch 64. None of the curves has flattened, so longer runs are still worth doing.

### Fixing learning on `full`

`full` now trains to 66.8 ± 1.3 passengers per episode in 500k steps, close to the heuristic's 69 and still rising. It was at 0. Reward shaping was not the fix. What made it work: a lift-relative observation, imitating a heuristic first, and a low fine-tuning learning rate.

**Only imitation plus a low learning rate learns the full preset.** Passengers delivered per episode on `full` after 500k PPO steps, mean of 2 seeds, best shaping per row. References: random policy 6.8, collective heuristic 69.

| Change | Variant | Delivered |
|---|---:|---:|
| Reward shaping | Original shaping | 0.1 |
| Reward shaping | No empty-serve penalty | 3.8 |
| Reward shaping | Progress shaping | 4.1 |
| Action design | Target-floor actions | 2.8 |
| Observation design | Lift-relative observation | 4.1 |
| Imitation warm start | Imitation only | 57.5 |
| Imitation warm start | Imitation + PPO, lr 3e-4 | 29.6 |
| Imitation warm start | **Imitation + PPO, lr 3e-5 (the recipe)** | **66.8** |

_Source: `elevator_rl.compare`, preset full, 64 envs × 32 steps, minibatch 64._

What each step showed:

1. **Learning breaks at `tall`, not later.** On 30 floors with one lift and no other features, PPO scores 0.2, below random's 4.5. So the cause is scale, not multi-lift, kinematics or hall calls.
2. **A heuristic proves the task is learnable.** Classic collective control (`CollectivePolicy`) delivers 59 on `tall` and 69 on `full`, against 5–7 for random.
3. **Reward and action fixes help only a little.** Dropping the empty-serve penalty, potential-based progress shaping, and target-floor actions each leave PPO below random.
4. **Imitation exposes an observation problem.** Copying the heuristic reaches 99% accuracy on `tall`, but only 71–75% on the decisions that matter on `full`. The per-floor absolute counts make "is anyone waiting ahead of me?" hard to learn for 4 lifts × 30 floors.
5. **A lift-relative observation fixes imitation.** Each lift sees the building centred on itself (`obs_type="relative"`). Imitation then reaches 57.5 on `full`. PPO from scratch still fails with it, so exploration is the other half of the problem.
6. **Fine-tuning needs a low learning rate.** At PPO's default 3e-4, the first updates erase half the imitated skill (57.5 → about 25). At 3e-5 it improves steadily to 66.8.

Reproduce with `python -m elevator_rl.train --preset full --obs-type relative --n-envs 64 --n-steps 32 --batch-size 64 --net-arch 256 256 --pretrain 100000 --learning-rate 3e-5`.

Next: train longer to see whether PPO beats the heuristic, and port the relative observation to C++ before Phase 2, since it is now the observation that matters.

## Phase 1: C++ core

Port the simulator as a plain C++17 struct with no Python dependency, and design it so the same layout carries over to JAX and Warp later.

**Design choices**

- **Fixed-size state.** Use `std::array<uint8_t, CAP>` for lift destinations and a ring buffer per floor with a hard cap (for example 32). Avoid `std::vector` and `std::deque`. A fixed size keeps steps allocation-free, makes EnvPool's buffers trivial, and is required later for JAX and Warp anyway.
- **Decide what happens on overflow.** Python queues are unbounded. Either drop arrivals at the cap, or prove the cap is never hit in practice by logging the max queue length over many Python episodes. Document the choice, because it changes the dynamics slightly.
- **Origin is never used.** `Passenger.origin` is never read, so only destinations need storing. FIFO order matters for boarding, so keep the queue as an ordered destination list.
- **Keep `step(action)` pure.** It should take an action and return `(served, boarded)`, with obs encoding a separate function writing into a caller-owned buffer.
- **RNG.** Use one `std::mt19937_64` (or PCG) per env, seeded from the reset seed.

**Parity testing**

You cannot get bit-identical trajectories to NumPy's `default_rng` (PCG64 + its own Bernoulli and integer algorithms) without reimplementing NumPy's samplers. Test in two layers instead:

1. **Deterministic parity.** Add an optional arrivals-injection hook to both implementations so the test supplies the arrivals. Then replay identical action and arrival sequences and assert identical obs, rewards and truncation, step for step, for every obs type.
2. **Statistical parity.** Under random and up/down-sweep policies, compare mean episode reward, arrivals per step and queue-length distributions between Python and C++ over about 1,000 episodes. Keep the existing `baselines.py` numbers as the reference.

### Phase 1 result: implemented

The C++ core in `cpp/` matches Python exactly and steps 70–120x faster on one thread. Every preset replays 2,000 Python steps with identical observations (both types), rewards and truncations, kinematics included.

| Preset | Python µs/step | C++ µs/step | Speed-up |
| --- | --: | --: | --: |
| `original` | 16 | 0.14 | 112x |
| `tall` | 31 | 0.26 | 117x |
| `multi` | 40 | 0.40 | 100x |
| `kinematic` | 59 | 0.77 | 77x |
| `hall_calls` | 79 | 1.17 | 67x |
| `full` | 164 | 1.90 | 86x |

C++ timings use `elevator_bench`: a random policy plus a custom observation every step, 2M steps, on one core.

What made exact parity work, and what to carry into Phases 2–4:

- **Injected arrivals.** `elevator_rl.trace` records each step's arrivals; `elevator_replay` feeds them in, so the different random number generators don't matter. A random-policy test then checks that C++'s own draws match Python's distributions.
- **Floating point.** The kinematics give bit-identical results only with the same operation order and `-ffp-contract=off` (no fused multiply-add). JAX and Warp will not be bit-exact, so plan on a tolerance there.
- **Rounding.** Python's `round()` rounds halves to even, and `std::round` does not. The port uses `std::nearbyint`.
- **Queue cap.** Each floor holds 512 waiting passengers; the longest seen in Python was 311. Drops are counted and tests assert there are none.
- **Cheap win found while porting.** Building the destination table only for floors with an arrival halved `full` (4.2 → 1.9 µs). Python does the same work in NumPy, so it could get a similar fix.

What this means for Phase 2: a C++ step (0.1–2 µs) is now cheaper than a single Python-to-C++ call (about 1 µs). The binding must step all envs in one call, or the call overhead eats the gain.

## Phase 2: Python binding

Use nanobind or pybind11 built with scikit-build-core, and expose a **batched** step from day one. A per-step call from Python costs about 1 µs whichever binding you use, so a single-env C++ `step` will only beat the 13 µs Python step by about 5–10x.

| Option | Pros | Cons |
| --- | --- | --- |
| nanobind / pybind11 extension | NumPy arrays zero-copy, builds as a wheel via `uv`, typed errors | C++ build in the Python package; wheels per platform |
| Plain shared library (`.dll` / `.so`) via `ctypes` or `cffi` | No Python headers; same binary usable from C#, Julia, Unity | Manual pointer/array marshalling, no exceptions across the boundary, Windows `__declspec(dllexport)` and calling-convention care |
| EnvPool's own build (Phase 3) | Threading and batching handled for you | Bazel-based, Linux-first; heavier toolchain |

What to build:

- `LiftEnvCpp(gymnasium.Env)`: a drop-in for `LiftEnv` with the same spaces, so `train.py` and the tests run unchanged against it.
- `LiftVecCpp`: a `VecEnv` that steps N envs in one C++ call, writing into preallocated `(N, obs_dim)` arrays with auto-reset. It releases the GIL during the loop (`nb::gil_scoped_release`). This alone likely gives most of Phase 3's gain.
- CI: build the extension on the existing GitHub Actions job (`uv sync` triggers the scikit-build-core build). Keep the Python env as the reference implementation and run parity tests on every push.

If you specifically want a DLL, build the core as a C-ABI library (`extern "C" lift_step(env*, int action, float* obs_out)`) and put the Python binding on top. You then get both, and the DLL stays reusable outside Python.

### Phase 2 result: implemented

The batched binding makes env stepping almost free. PPO on the training recipe trains 1.7x faster than with `DummyVecEnv` (2,886 vs 1,739 steps/s), and 2.25x faster with minibatch 512 (5,476 vs 2,433). That is the ceiling Amdahl's law predicted from the env's 40% share, and the env is now about 1% of training time.

**With the C++ env, stepping is 1% of training time.** Share of PPO wall-clock time, preset full, relative obs, 256×256 net, 64 envs, 4,096-step rollout.

| VecEnv | Minibatch | Env stepping | Policy forward + SB3 | Gradient update | PPO steps/s |
|---|---:|---:|---:|---:|---:|
| DummyVecEnv | 64 | 39.7% | 6.1% | 54.2% | 1,739 |
| SubprocVecEnv | 64 | 19.2% | 9.6% | 71.2% | 2,244 |
| **CppVecEnv** | 64 | 0.7% | 9.5% | 89.8% | 2,886 |
| DummyVecEnv | 512 | 55.9% | 8.6% | 35.5% | 2,433 |
| SubprocVecEnv | 512 | 30.6% | 15.5% | 53.8% | 3,630 |
| **CppVecEnv** | 512 | 1.4% | 18.2% | 80.4% | 5,476 |

_Source: `elevator_rl.sweep`, 40,000 timed steps × 2 seeds, 1 torch thread._

What was built:

- **C++ `VecEnv`** (`cpp/src/vec_env.cpp`): steps N envs in one call into caller-owned arrays and auto-resets finished episodes.
- **nanobind module `elevator_rl._cpp`**: built by CMake into `src/elevator_rl/`. It releases the GIL while stepping and exposes a single `Env` with injectable arrivals for parity tests.
- **`CppVecEnv`**: the SB3 adapter, selected with `--vec-env cpp` in `train`, `compare` and `sweep`. Raw stepping is about 600k env steps/s on `full` with 64 envs, against about 6.5k for `DummyVecEnv`.
- **Before the binding**, the reward shaping, target actions, direction observation and relative observation were ported to C++. Trace replay checks them all against Python exactly.

**What this means for Phase 3:** the gate is not met. Multithreaded env stepping (EnvPool) speeds up 1% of the time, so it cannot help this training setup. The time now goes to PyTorch's gradient update (80–90%), run here on one thread with a 256×256 network on a 1,220-value observation. The next levers are on the learner: more PyTorch threads, larger minibatches (with learning quality checked), a GPU, or the JAX route that compiles the env and PPO together. `SubprocVecEnv` with 256 envs was left out because it ran the machine out of memory: 256 processes each load PyTorch.

### Learner speed-ups: threads and GPU

Speeding up the learner pays off far more than the env did. A GTX 1080 with minibatch 2048 and 1024 C++ envs runs 50,230 PPO steps/s, 17× the training recipe's 2,909.

**A GPU learner with the C++ env trains 17× faster than the recipe.** PPO steps/s on full, 256×256 MLP, 64 C++ envs unless marked. With Python envs, the same GPU setup is capped by the env (92% of the time).

| Learner setting | PPO steps/s | vs recipe |
|---|---:|---:|
| GPU · minibatch 2048 · 1024 envs | 50,230 | 17.3× |
| GPU · minibatch 2048 · 64 envs | 23,199 | 8.0× |
| 8 CPU threads · minibatch 2048 | 19,156 | 6.6× |
| 8 CPU threads · minibatch 512 | 14,596 | 5.0× |
| 1 CPU thread · minibatch 2048 | 6,029 | 2.1× |
| 8 CPU threads · minibatch 64 | 5,044 | 1.7× |
| GPU · minibatch 2048 · 1024 Python envs | 4,476 | 1.5× |
| GPU · minibatch 64 | 4,342 | 1.5× |
| 1 CPU thread · minibatch 64 (recipe) | 2,909 | 1.0× |

_Source: `elevator_rl.sweep`, rollout 4,096 (8,192 for 256+ envs), 40,000–50,000 timed steps × 2 seeds, Ryzen 7 3700X + GTX 1080._

- **CPU threads.** 8 threads (the 3700X's physical cores) is fastest; 16 hyperthreads are about 20% slower. Threads barely help minibatch 64 (1.7×), because each gradient step is too small to split. At minibatch 2048 they give 3.2×.
- **GPU.** At minibatch 64 the GPU is no faster than 8 CPU threads, because kernel launches dominate. It needs minibatches of 512 or more. Then rollout collection (policy forward passes and SB3 bookkeeping) becomes the bottleneck, so more envs per step help: going from 64 to 1024 envs raises it from 23k to 50k steps/s.
- **The env matters again.** With the GPU learner, Python's `DummyVecEnv` caps training at 4,476 steps/s, with the env taking 92% of the time. The C++ env is 11× faster. At 1024 C++ envs, env stepping is back to 14% of the time, so Phase 3's multithreading could add at most about 1.16×.
- **Not yet checked: learning quality.** The recipe learns with minibatch 64 and 64 envs × 32 steps. The fast settings use minibatch 2048 and 8 steps per env, and large minibatches learned worse per sample on `original`. The next step is a compare run on `full` at these settings.

Reproduce with `uv run python -m elevator_rl.sweep --preset full --obs-type relative --net-arch 256 256 --vec-envs dummy cpp --n-envs 64 256 1024 --batch-sizes 2048 --rollout 8192 --devices cuda`. GPU runs need a CUDA build of torch (2.14.1+cu126 supports the GTX 1080), but the lockfile pins the CPU build. `train.py --device cuda` trains on the GPU.

### Phase 2 alternative: a C library through ctypes

The env also works as a plain shared library with a flat `extern "C"` API, loaded with `ctypes`. It gives identical trajectories and the same training speed as the nanobind module: 22,837 vs 22,615 PPO steps/s at the most env-heavy CPU setting. A raw batch step through ctypes is even 0.5 µs cheaper than through nanobind, but only when the pointers are built once. Built naively, a ctypes call costs 4–11× more.

What was built (branch `ctypes-dll`):

- `cpp/capi/elevator_c.h`: 12 functions over `elevator::VecEnv`, using only fixed-width integers, doubles, pointers and one plain struct (`ElevatorOptions`).
    - The env is an opaque handle, created and destroyed by the library.
    - Every function returns a status code. `elevator_last_error()` gives the message, kept per thread.
    - `elevator_abi_version()` and `elevator_options_size()` let a binding check that it matches the library.
- `libelevator_c.so` (`elevator_c.dll` on Windows), a CMake target built next to the nanobind module. It needs no Python headers and exports only the 12 API symbols.
- `ctypes_env.py`: `CtypesVecEnv` subclasses `CppVecEnv`, so it has the same SB3 interface. Only the two batch calls go through ctypes. It is available as `--vec-env ctypes` in `train`, `compare` and `sweep`.
- `tests/test_ctypes_env.py` (17 tests): step-for-step parity with `CppVecEnv` on 3 presets × 3 observation and action modes, across auto-resets and reseeding. Other tests cover out-of-range actions, bad configs, NULL handles, per-thread errors, freeing the handle, and a PPO run.
- `scripts/time_ctypes.py`: the call-overhead and stepping benchmarks below.

**A ctypes step is cheapest with cached pointers, and 4–11× dearer than nanobind without.** µs per batch step call, 1 env, preset original, best of 5 × 200,000 calls. A bare ctypes call costs 0.25 µs; a nanobind getter 0.06 µs.

| Route | µs per call | vs nanobind |
|---|---:|---:|
| ctypes, cached pointers, GIL held (`PyDLL`) | 0.855 | 0.6× |
| **ctypes, cached pointers (`CtypesVecEnv`)** | **0.934** | **0.6×** |
| nanobind (`CppVecEnv`) | 1.495 | 1.0× |
| ctypes, `array.ctypes.data` per call | 6.382 | 4.3× |
| ctypes, `ndpointer` argtypes | 16.551 | 11.1× |

_Source: `scripts/time_ctypes.py`, raw batch step calls, preset original, relative obs, Ryzen 7 3700X._

The same ranking holds on `full` (2.67 µs for nanobind, 2.17 µs cached), where the step itself is about 1.7 µs more expensive.

In training, the route makes no difference: env stepping is 1–7% of PPO time, and the calls are the same size either way.

| PPO setting (`full`, relative obs, 256×256 MLP) | nanobind | ctypes |
|---|---:|---:|
| 64 envs, minibatch 64, 1 CPU thread (the recipe) | 2,944 ± 17 | 2,940 ± 25 |
| 64 envs, minibatch 512, 1 CPU thread | 5,633 ± 22 | 5,633 ± 7 |
| 1024 envs, minibatch 2048, 8 CPU threads | 22,615 ± 447 | 22,837 ± 3 |

_Source: `elevator_rl.sweep --vec-envs cpp ctypes`, PPO steps/s, 40,000–50,000 timed steps × 2 seeds. The GPU setting was not rerun: the venv has the CPU build of torch, and installing the `cuda` group would remove the EnvPool wheel._

Through SB3's `VecEnv.step`, which adds the infos and the copies, the two are within noise at every batch size. ctypes was 1.09–1.11× nanobind at 1 env and 0.98–1.03× at 256 and 1024. At 64 envs it swung from 0.82× to 1.22× between repeats.

What we learned:

- **The cost is in converting the arguments, not in the call.** A bare ctypes call costs 0.25 µs, against 0.06 µs for a nanobind getter. Passing arrays is what gets expensive: `array.ctypes.data` builds a helper object per array (about 0.8 µs each), and `ndpointer` argtypes check each array in Python (about 2.2 µs each).
- **Caching the pointers is what makes ctypes fast.** It works only because the buffers never move. `CppVecEnv` now allocates them once and copies each step's actions into a fixed array, and the code says not to rebind them.
- **nanobind is slower here because it checks more.** It validates the shape and dtype of all 7 arrays on every call. ctypes with raw pointers checks nothing, so a wrong dtype is silent memory corruption, not an error. The library checks what it can: NULL pointers and the action range.
- **Releasing the GIL is nearly free.** `ctypes.CDLL` releases it on every call, as the nanobind binding does. Keeping it (`ctypes.PyDLL`) saves only 0.07–0.08 µs.
- **Hidden visibility didn't hide everything.** `-fvisibility=hidden` applies only to the target's own sources, so about 60 internal C++ symbols from the static `elevator` library were exported too. Linking with `--exclude-libs,ALL` cut the exports to the 12 API functions.

#### Pitfalls of the C library route

Most of what nanobind does for you, a C API makes you do by hand, and mistakes crash or corrupt memory rather than raise. Only the Linux build was run; the Windows rows are applied in the code but untested.

| Area | Pitfall | What goes wrong | What this repo does |
|---|---|---|---|
| Calling convention | `__cdecl` vs `__stdcall` | On 32-bit Windows, the wrong one (`CDLL` vs `WinDLL`) unbalances the stack. x86-64 has a single convention per OS, so it is moot there. | `ELEVATOR_CALL` is `__cdecl` on Windows; Python loads with `CDLL` |
| Calling convention | Undeclared signatures | ctypes assumes `int` for every argument and return. The 64-bit handle comes back truncated to 32 bits and crashes later, far from the cause. | `argtypes` and `restype` for all 12 functions; `c_void_p` for the handle |
| Calling convention | Struct layout and ABI drift | The `ctypes.Structure` must match field order, types and padding. A changed header that the binding doesn't follow is misread silently. | `static_assert` of no padding; the ABI version and `sizeof(ElevatorOptions)` are checked at load |
| Calling convention | Types without a fixed size | `bool`, `enum`, `long` and `size_t` vary by compiler or platform. | `int32_t`, `uint64_t` and `double` only |
| Memory ownership | Who frees the handle | The library allocates it, so the library must free it. Python's garbage collector can't see it. | `elevator_vec_destroy`, called by a `weakref.finalize` that holds no reference to the env (one would keep it alive forever); `close()` frees it once |
| Memory ownership | Arrays C points into | The caller owns every array, and C only keeps its address. A NumPy array that is rebound, freed or not contiguous makes C write into freed memory. | Buffers allocated once and never rebound; contiguity asserted when caching pointers; actions copied into a fixed array |
| Error handling | Exceptions across the boundary | A C++ exception escaping an `extern "C"` function calls `std::terminate`, which kills the Python process. | Every entry point catches everything and returns `ELEVATOR_ERROR` |
| Error handling | Reporting the error | A C function can only return a code. A global message would be overwritten by other threads. | A `thread_local` message from `elevator_last_error()`; `c_char_p` copies it at once; Python raises `RuntimeError` |
| Error handling | No type checks | ctypes passes a raw address, so a wrong dtype or shape is garbage, not an error. | The library checks NULLs and the action range, writing nothing on failure; Python fixes dtypes when allocating. `ndpointer` checks cost 17 µs a call. |
| Windows | `dllexport` | Windows exports nothing by default; Linux exports everything. | `ELEVATOR_API` is `dllexport` while building and `dllimport` for C users; hidden visibility and `--exclude-libs` on Linux |
| Windows | CRT mismatch | A DLL built with `/MD` needs the matching `vcruntime` on the machine. Each CRT has its own heap, so memory freed in a different CRT corrupts the heap. | Static CRT (`MSVC_RUNTIME_LIBRARY MultiThreaded`); no memory is allocated on one side and freed on the other |
| Windows | DLL search path | Since Python 3.8, `PATH` is not searched for a DLL's dependencies. | Loaded by full path; with the static CRT it needs only system DLLs |
| Linux | `libstdc++` version | The `.so` links the system `libstdc++.so.6`. Built with a newer compiler, it fails on older systems (`GLIBCXX_... not found`). | Not handled; `-static-libstdc++` would fix it for distribution |

## Phase 3: EnvPool

Adopt EnvPool only if Phase 2's batched VecEnv leaves the env as a measurable share of training time. For a 100 ns step, thread-pool dispatch can cost more than the step itself.

**How it works.** You implement an `Env<Spec>` C++ class (`Reset`, `Step`, `Allocate` writing into its state buffers) plus a spec of obs and action shapes. EnvPool runs N envs on a thread pool and hands Python a batch.

- **Sync mode** (`batch_size == num_envs`) is a drop-in for a VecEnv.
- **Async mode** (`batch_size < num_envs`) returns whichever envs finish first. That helps when step times vary, which they barely do here.

**Fitting it into training**

- SB3 expects its own `VecEnv` API. EnvPool ships a gym/gymnasium interface, and you would need a thin adapter (EnvPool's docs include an SB3 example). Auto-reset semantics differ between EnvPool, Gymnasium 1.x and SB3, so check where the final obs of an episode goes.
- With SB3, the bigger win is raising `n_envs` from 1 to 16–64 and enlarging the batch. That cuts per-sample policy overhead, and you can do it today with the Python env.
- A CleanRL-style PPO script (single file, EnvPool-native) is the usual pairing and is easier to profile than SB3.

**Practical blockers**

- EnvPool's build is Bazel and is effectively Linux-only for custom envs. Adding an env means forking or vendoring the repo.
- Check that EnvPool's current release supports your Python version (the project needs Python >= 3.11) before investing in it.

### Phase 3 result: implemented

The elevator env runs inside EnvPool. At 256 envs it reaches 0.96M env steps/s in sync mode and 1.5M in async mode on 16 threads, 1.9–2.9× the single-threaded C++ VecEnv. As the Phase 2 gate predicted, that doesn't speed up SB3 training, where env stepping was already 1% of the time.

**With 16 threads, EnvPool steps 256 envs 1.9–2.9× faster than the C++ VecEnv.** Env steps/s, preset full, random actions; async hands Python half the envs at a time. The single-threaded C++ VecEnv steps 516,211 env steps/s at 256 envs.

| EnvPool threads | Sync | Async |
|---|---:|---:|
| 1 | 245,600 | 388,814 |
| 2 | 400,362 | 643,455 |
| 4 | 621,782 | 1,146,054 |
| 8 | 807,596 | 1,400,261 |
| 16 | 955,928 | 1,522,563 |

_Source: `scripts/time_vec_env.py`, preset full, relative obs, 256 envs, 2M env steps per point, Ryzen 7 3700X (8 cores, 16 threads). The script also covers 16, 64 and 1024 envs._

What was built:

- `cpp/envpool/`: an `ElevatorEnv` written against EnvPool's `Env<Spec>` interface. Its spec sizes come from a throwaway C++ `Env`, so they can't drift. Each env draws its episode seeds from EnvPool's own `std::mt19937`.
- `scripts/build_envpool.sh`: checks out sail-sg/envpool at a pinned commit, copies the env in, and trims the package to EnvPool's core plus the elevator env. It then builds a 574 KB wheel with Bazel 9 in about 3 minutes. The full release build would also need Qt, SWIG and Go.
- `EnvPoolVecEnv`, the SB3 adapter, available as `--vec-env envpool`.
- Tests: each pooled env matches a lone C++ env step for step, on 1 and on 4 threads. The test reimplements `std::mt19937` in Python to know each env's seeds.

What we learned:

- **One thread loses.** On 1 thread EnvPool manages 196–245k env steps/s, against 460–520k for `CppVecEnv`. Handing work to a worker thread, plus EnvPool's Python wrapper, costs more than a 2 µs step. Threads only win from 64 envs and 4 threads; at 16 envs they never do.
- **Async helps most at mid sizes.** It is 1.5–2× faster than sync at 64–256 envs, because Python sends one half's actions while the other half steps. On-policy PPO can't use it: it needs every env at every step.
- **Reset timing differs.** EnvPool resets a finished env on the next step, ignoring that step's action. SB3 expects the reset on the same step, with the last observation in `terminal_observation`. The adapter resets finished envs straight away in one batched call; EnvPool's own SB3 example resets them one at a time.
- **Packaging was the slow part.** EnvPool needs Python 3.12 or later, so the project moved to 3.12. That broke exact parity with the C++ port, because from 3.12 `sum()` of floats uses compensated summation; a plain loop fixed it. Ubuntu's Python 3.12 has no headers for building extensions, so the venv now uses uv's managed Python.
- **`uv sync` uninstalls the wheel,** because it isn't in the lockfile. Reinstall it after a sync; `uv run` leaves it alone.

## Phase 4: JAX and NVIDIA Warp (both built)

JAX is the stronger fit for this env. It lets env, policy and PPO update compile into one XLA program (the PureJaxRL / Gymnax pattern), which removes the Python loop entirely. That is the step that changes training speed by orders of magnitude, not the env port.

|  | JAX | NVIDIA Warp |
| --- | --- | --- |
| Model | Pure functions over arrays, `jit` + `vmap` over envs | CUDA-style kernels written in Python, one thread per env |
| Hardware | CPU, GPU, TPU | NVIDIA GPU (CPU fallback for debugging) |
| RL ecosystem | Gymnax, PureJaxRL, Brax, Flax/Optax | Mostly physics/robotics; interop with PyTorch/JAX via DLPack |
| Fits this env? | Yes: integer state, small arrays, discrete actions | Possible, but it shines for physics-heavy envs |

**What the env must become (start these choices in Phase 1)**

- **Fixed shapes everywhere.** Use queues as `(n_floors, max_queue)` arrays with a length per floor, and the lift as `(capacity,)` destinations with a count. No Python lists or deques.
- **No data-dependent control flow.** The `while` boarding loop becomes a vectorised computation: board `k = min(waiting, capacity - in_lift)` passengers via index arithmetic, with `jnp.where` for action branches.
- **Explicit RNG keys.** Split a `jax.random` key per step. Results will not match NumPy or C++ draw-for-draw, so rely on statistical parity again.
- **State as a pytree.** Use an immutable `NamedTuple`/`flax.struct` state and `step(state, action, key) -> (state, obs, reward, done)`.

The C++ port and the JAX version share this array-based design. Written that way, Phase 1 is also a reference spec for Phase 4.

### Phase 4 result: JAX implemented

JAX PPO trains `full` at 30k PPO steps/s with the recipe's settings, and at up to 238k steps/s with 16,384 envs, on the GTX 1080. That is 2.6–7× SB3 with the C++ env on the same GPU, and 82× the SB3 recipe on one CPU thread. It learns as SB3 does: 138 vs 134 passengers delivered on `original` after 200k steps with matching settings.

**JAX PPO trains 2.6–7× faster than SB3 on the same GPU.** PPO steps/s, preset full, relative obs, 256×256 MLPs, 3 epochs, GTX 1080. SB3 was not run with 16,384 envs; the SB3 recipe on 1 CPU thread runs 2,909.

| Setting | SB3 + C++ env | JAX | JAX over SB3 |
|---|---:|---:|---:|
| 64 envs × 64 steps, minibatch 64 | 4,342 | 30,350 | 7.0× |
| 64 envs × 64 steps, minibatch 512 | 16,295 | 55,060 | 3.4× |
| 64 envs × 64 steps, minibatch 2048 | 23,199 | 60,653 | 2.6× |
| 1024 envs × 8 steps, minibatch 2048 | 50,230 | 184,902 | 3.7× |
| 16,384 envs × 8 steps, minibatch 16,384 | not run | 238,109 |  |

_Source: `scripts/time_jax_ppo.py` and `elevator_rl.sweep`, about 20 s of training per JAX point, compilation and evaluation excluded._

What was built:

- `jax_env.py`: `BuildingEnv` as pure functions over fixed-shape arrays, with step actions, `box` and `relative` observations, every reward shaping, and kinematics, hall calls and all traffic profiles.
  - Floor queues are `(floors, 512)` int8 arrays, and each lift holds a count of passengers per destination floor.
  - Boarding in arrival order is a cumulative sum, and the queue is compacted with a scatter.
  - Lifts act in order through a loop that is unrolled when traced.
- `jax_ppo.py`: PPO written to match SB3's, with the rollout and the update epochs compiled into one program.
- Tests:
  - exact parity with Python on every preset (float64, with Python's arrivals injected);
  - parity in distribution when JAX draws its own arrivals;
  - PPO learns `original`.
- Dependency groups `jax` (run in CI) and `jax-cuda`.

What we learned:

- **XLA fuses multiply-adds.** It turns `a + b * c` into one fused multiply-add, so 28% of results differ from NumPy in the last bit, and the kinematics diverged after a few steps. An `optimization_barrier` around the product is JAX's `-ffp-contract=off`.
- **Dtypes drift.** An int32 divided by an int divides in float32. Under x64, `sum()` of int32 returns int64, which breaks scan carries.
- **The GPU env was bound by memory traffic, not compute.**
  - The first version plateaued at 136k env steps/s: each lift's action selected between two whole states, copying the 61 KB of queues 4 times per step.
  - Updating only the rows a step touches, counting up-goers as they arrive and leave, and storing int8 queues raised it to 633k. Replacing an atomic scatter-add with a compare-and-sum raised it to 666k.
  - Two plausible fixes made it slower: an associative-scan cumsum (584k) and a `searchsorted` gather (269k). Measure each change.
- **JAX doesn't pay on the CPU.** JAX PPO manages 3.6–4.9k steps/s there, below SB3 with 8 torch threads (19k).
- **A shell setting broke CUDA.** AMD's AOCC setup script put the system library directories on `LD_LIBRARY_PATH`. JAX then loaded Ubuntu's 2023 `libnvJitLink` instead of its own and fell back to the CPU. Fixed in `~/.profile`.
- **Compiling takes 3–4 s** per configuration, paid once per run.
- **Not ported yet:** the target action mode, the `custom` observation and the imitation warm start (see the backlog).

### Phase 4 result: NVIDIA Warp implemented

The Warp env steps 3.9M envs/s at 16,384 envs, 5.8× the JAX env and 7.6× the single-threaded C++ env. PPO on it runs at 362,635 steps/s, 1.52× PPO on the JAX env. That is the most Amdahl's law allowed, since the JAX env was about a third of training time; the policy and the update are now the bottleneck again.

**The Warp env steps 3.9M envs/s, 5.8× the JAX env, at 16,384 envs.** Env steps/s on a GTX 1080, preset full, relative obs, random actions, all inside one jit. For reference, the single-threaded C++ VecEnv steps 516,211 env steps/s.

| Parallel envs | Warp | JAX | Warp over JAX |
|---|---:|---:|---:|
| 256 | 293,787 | 279,293 | 1.1× |
| 1,024 | 1,093,706 | 510,015 | 2.1× |
| 4,096 | 3,759,973 | 646,121 | 5.8× |
| 16,384 | 3,914,387 | 680,735 | 5.8× |

_Source: `scripts/time_jax_env.py --backends warp jax`, 400 steps per env, compilation excluded._

| PPO setting (`full`) | JAX env | Warp env | Speed-up |
| --- | --: | --: | --: |
| 64 envs × 64 steps, minibatch 64 | 30,350 | 34,064 | 1.12× |
| 1024 envs × 8 steps, minibatch 2048 | 184,902 | 254,095 | 1.37× |
| 16,384 envs × 8 steps, minibatch 16,384 | 238,109 | 362,635 | 1.52× |

What was built:

- `warp_env.py`: one GPU thread per env, with plain loops and branches ported from the C++ code.
  - Boarding walks only the passengers actually waiting and compacts the queue in place.
  - A finished episode resets inside its own thread, so the other threads skip it.
  - The kernels are generated per config (closures, with the sizes and feature flags as compile-time constants).
- The kernels run inside `jax.jit` through `warp.jax_kernel`, with the env state held as JAX arrays. `jax_ppo --backend warp` trains on it unchanged.
- Tests: exact float64 parity with Python on every preset, returns that match Python in distribution, auto-reset, and a PPO update.
- A `warp` dependency group.

What we learned:

- **Pascal support needs Warp's CUDA 12 build.** PyPI's `warp-lang` is built with CUDA 13, which refuses Pascal GPUs; the `warp` group installs the CUDA 12 build from Warp's GitHub release.
- **Fused multiply-add, a third time.** NVRTC fuses `a - b * c` by default (27,880 of 100,000 results differed from NumPy). Warp's `fuse_fp=False` module option turns that off; with it, parity was exact on the first run.
- **CUDA-style code suits this env.** It has a lot of data-dependent work: queues of varying length, early exits, and resets in a few envs at a time. JAX must mask all of it at full width; Warp threads simply skip it. The JAX env's speed came from rewriting the algorithm, while the Warp env reads like the C++ one.
- **Compiling is slow.** The first build of a config takes about a minute (18 s for the reset kernel, 42 s for the step kernel), against 3–4 s for JAX. Warp caches the result on disk, after which tests and runs start fast.
- **The kernel can leave outputs unwritten.** It only writes the terminal observation for envs that finished, so the PPO must mask with `where`, not multiply by `done`: NaN × 0 is NaN.

## Pitfalls and risks

The biggest risk is optimising the 2% of the training time spent in the env. Profile before each phase.

| Pitfall | Why it bites here | Mitigation |
| --- | --- | --- |
| Speeding up the wrong thing | Env is \~78k steps/s, PPO \~1.5k steps/s | Benchmark end-to-end PPO steps/s; try `n_envs=16+` with the Python env first |
| Per-call overhead | A \~1 µs binding call vs a \~0.1 µs C++ step | Batched step API; release the GIL inside it |
| Unbounded queues | Python `deque` has no limit; fixed arrays do | Measure max queue length; pick a cap and drop or clamp, and document it |
| RNG mismatch | NumPy PCG64 samplers differ from `std::` and `jax.random` | Injected-arrivals tests for exact parity; statistical tests for the rest |
| Reset semantics | `reset` pre-samples 10 arrival rounds; auto-reset differs between SB3, Gymnasium 1.x and EnvPool | Port the 10 rounds explicitly; test the terminal obs and `info` handling |
| Obs dtype drift | `box` is float32, others int64/int8 | One spec per obs type, asserted in tests against the Python spaces |
| Build/CI complexity | C++ toolchain, platform wheels, EnvPool's Bazel | scikit-build-core + nanobind; keep a pure-Python fallback import |
| Windows DLL issues | `dllexport`, CRT mismatches, `ctypes` calling conventions | Prefer the Python extension; if a DLL, keep a flat `extern "C"` API with POD types only |
| Multithreading bugs | Shared RNG or buffers across envs | One RNG and state per env; no globals; run tests under ThreadSanitizer |
| Policy quality regression | Subtle dynamics change (caps, boarding order) | Train with both envs at a fixed seed set; compare `benchmark.py` results |

## Benchmarks and milestones

The headline metric at every gate was end-to-end PPO steps/s at equal policy quality. Every gate has now been answered. On `full`, the baseline was 1,739 PPO steps/s (64 Python envs, SB3, one CPU thread), and the best is 362,635 (the JAX PPO on the Warp env, on the GPU).

**Every phase is built: gate 2 said no, gate 3 led to JAX and Warp.**

```
1 · C++ core  ──[Gate 1]──>  2 · Binding  ──[Gate 2]──>  3 · EnvPool  ──[Gate 3]──>  4 · JAX / Warp
Fixed-size state             nanobind / pybind11          Thread-pool batching          Env + PPO in one
Per-env RNG                  Batched VecEnv               SB3 adapter or                compiled program
Parity tests                 GIL released                 CleanRL-style PPO             Warp: 1.5× over JAX

Gate 1: passed, exact parity on all presets.
Gate 2: env over 10% of PPO time? No: 1%.
Gate 3: need 10× more? JAX: 2.6–7× over SB3.
```

Gate 2 answered no: after Phase 2 the env took 1% of training time, so EnvPool could not speed up SB3 training. It was built anyway as a learning exercise, and it confirmed that. Gate 3 led to JAX and then Warp, which gave the gains the env work could not. The one open milestone, reward parity for each backend across 3 seeds, is in the backlog.

- [x] Record the baseline: `scripts/time_env.py`, `scripts/time_ppo.py`, plus a `py-spy` profile of one PPO run
- [x] Quick win before any C++: rerun `time_ppo.py` with `n_envs` = 8, 16 and 32 (`SubprocVecEnv` and `DummyVecEnv`)
- [x] Phase 1: C++ env steps/s (single thread, no Python) and parity test suite green
- [x] Phase 2: batched VecEnv steps/s at N = 1, 16, 256; PPO steps/s with it
- [x] Phase 3: EnvPool sync vs async at 1–16 threads; PPO steps/s vs Phase 2
- [ ] Every phase: `benchmark.py` reward table within noise of the Python env across 3 seeds

## Backlog: to come back to

These were recommended along the way and deferred, so Phases 3 and 4 could go ahead first. Logged 2026-10-10.

- [ ] **Check learning quality at the fast settings.** On `full`, compare the recipe (minibatch 64, 64 envs × 32 steps) against minibatch 512 and 2048, and 64 vs 1024 envs, on the GPU. Plot reward against wall-clock minutes. Large minibatches learned worse per sample on `original`, so the 17× throughput may cost reward.
- [ ] **Tune PPO for large minibatches** if that check shows a gap: a higher learning rate, more gradient epochs per rollout, or a learning-rate schedule.
- [ ] **Longer training runs on `full`,** deferred until the speed-ups land. Target: beat the heuristic's 69 (the recipe reaches 66.8 ± 1.3 after 500k steps).
- [ ] **Learner defaults in `train.py`:** set torch threads to the physical core count (8 here, against 1 now), and suggest `--device cuda` when minibatch ≥ 512.
- [ ] **Replace SB3's per-step overhead.** With a GPU learner, the policy forward pass and SB3 bookkeeping take 28–68% of the time. Done in Phase 4: the JAX PPO removes it (7× at the recipe's settings).
- [ ] **Reward parity per phase:** run the `benchmark.py` reward table across 3 seeds for each env backend (Python, C++, EnvPool, JAX), as the milestone list asks.
- [x] **DLL route (open question in the comments):** a flat `extern "C"` API loaded with `ctypes`, as a comparison with the nanobind module. Done on branch ctypes-dll: see its result section under Phase 2.
- [ ] **Packaging:** build the C++ module with scikit-build-core so `uv sync` compiles it, keeping the pure-Python fallback.
- [ ] **`SubprocVecEnv` memory:** 256 subprocesses ran this 31 GB machine out of memory, because each process loads torch. Either cap it in the sweep or document the limit.
- [x] **NVIDIA Warp,** the other half of Phase 4. Done: see its result section.
- [x] **Merged as PR #2** for `heavier-env`.

Added after Phases 3 and 4:

- [ ] **Run the learning-quality check in JAX.** It is now cheap: 5M steps of `full` take under a minute at the fast settings.
- [x] **Port the imitation warm start to JAX,** so the `full` recipe (behaviour cloning, then PPO at learning rate 3e-5) can run there. Done: `jax_imitate` clones the heuristic into the JAX PPO's networks (`jax_ppo --pretrain 100000`, or `warm_start.samples=100000` in the Hydra config). It starts at 54 passengers (SB3's clone: 57.5), and the recipe then reaches 68.6 ± 1.1 after 5M steps (3 seeds), level with the heuristic's 69.
- [ ] **Port the target action mode and the `custom` observation to JAX.**
- [ ] **Profile the JAX env** with `jax.profiler` to find what limits it at about 660k env steps/s; applying the actions is half of each step.
- [ ] **Try EnvPool's XLA interface,** which lets JAX step EnvPool envs from inside `jit`, as a bridge between Phases 3 and 4.

Added after Warp:

- [ ] **Profile the JAX PPO's policy and update.** With the Warp env, stepping is only about a tenth of training time, so the learner is the bottleneck again.
- [ ] **Port the target action mode and the `custom` observation to Warp,** alongside the JAX port.
- [ ] **Run the Warp tests in CI** on Warp's CPU backend, if the compile time (about a minute per config) is acceptable there.
- [ ] **Hyperparameter testing.** Tooling done: `elevator_rl.tune` (Hydra), `tune_search` (Optuna) and `tune_report`. Results to follow. Sweep PPO's learning rate, epochs, minibatch size, entropy coefficient, clip range and network size, and compare reward across seeds. Adding Hydra may make this easier: one config tree for the env preset, backend and PPO settings, with multirun sweeps (and an Optuna sweeper) instead of the separate argparse CLIs in `train.py`, `jax_ppo.py` and `sweep.py`.
