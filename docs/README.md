# Docs

This project is a learning exercise in accelerating a Python RL environment.
The elevator env was made heavier, then ported to C++, EnvPool, JAX and
NVIDIA Warp, each port checked step for step against Python. On the heaviest
preset, PPO training now runs 209× faster than at the start: 362,635 PPO
steps/s against 1,739, on a GTX 1080.

- [Acceleration plan](acceleration-plan.md): the plan, each phase's results,
  the pitfalls, the benchmark gates and the backlog of what to do next. It
  is a copy of the live plan doc as of 2026-10-10, with each chart turned
  into a table of its data.
- [Setup notes](setup-notes.md): what it took to run every backend on the
  development machine, covering Python 3.12, the dependency groups, CUDA 12
  builds for a Pascal GPU, `LD_LIBRARY_PATH`, fused multiply-adds and
  compile times.

The main [README](../README.md) has the commands to build, test and run each
backend.

| Backend | Code | Speed on `full` |
|---|---|---|
| Python env | `src/elevator_rl/building.py`, `building_env.py` | about 6.5k env steps/s |
| C++ port | `cpp/`, `src/elevator_rl/cpp_env.py` | about 0.5M env steps/s, 1 thread |
| C++ as a C library | `cpp/capi/`, `src/elevator_rl/ctypes_env.py` | the same as the C++ port (ctypes) |
| EnvPool | `cpp/envpool/`, `scripts/build_envpool.sh`, `src/elevator_rl/envpool_env.py` | up to 1.5M env steps/s, 16 threads |
| JAX env | `src/elevator_rl/jax_env.py` | about 0.68M env steps/s on the GPU |
| Warp env | `src/elevator_rl/warp_env.py` | about 3.9M env steps/s on the GPU |
| SB3 PPO | `src/elevator_rl/train.py`, `compare.py`, `sweep.py` | up to 50k PPO steps/s |
| JAX PPO | `src/elevator_rl/jax_ppo.py` | up to 363k PPO steps/s (on the Warp env) |
