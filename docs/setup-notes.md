# Setup notes

Lessons from getting every backend running on the development machine: a
Ryzen 7 3700X (8 cores, 16 threads), 31 GB of RAM, an NVIDIA GTX 1080
(Pascal, sm_61, 8 GB) with driver 580 (CUDA 13.0), and Ubuntu 24.04. All of
it was done without sudo; tools such as `bazelisk` (as `bazel`) and `gh`
went into `~/.local/bin`. The main [README](../README.md) has the build and
run commands; this page covers what went wrong and why.

## Python

- **Python 3.12, from uv.** EnvPool needs Python 3.12 or later, so
  `.python-version` pins 3.12. Ubuntu's own 3.12 has no development headers,
  which the nanobind module needs, so the venv uses uv's managed Python:
  `uv sync --python-preference only-managed`.
- **Build the C++ module against the venv by absolute path:**
  `cmake -S cpp -B cpp/build -DPython_EXECUTABLE=$PWD/.venv/bin/python`.
  With a relative path, CMake picked up another Python on the PATH
  (miniconda) and skipped the module.
- **Float sums changed in Python 3.12.** From 3.12, `sum()` of floats uses
  compensated summation, so it rounds differently from the C++ port and
  broke exact parity. `BuildingEnv._potential` therefore adds terms with a
  plain loop.

## Optional dependency groups

| Group | What it adds | Notes |
|---|---|---|
| `cpu` (default) | torch from PyTorch's CPU index | A plain `uv sync` or `uv run` puts it back. |
| `cuda` | torch 2.14 built for CUDA 12.6 | Use `uv sync --no-group cpu --group cuda`, then `uv run --no-group cpu --group cuda ...` (or `uv run --no-sync`), or `uv run` swaps back to the CPU build. |
| `jax` | JAX (CPU) and optax | Installed in CI, so the JAX parity tests run there. |
| `jax-cuda` | JAX with the CUDA 12 plugin | |
| `warp` | NVIDIA Warp's CUDA 12 build, plus `jax-cuda` | Taken from Warp's GitHub release (Linux x86_64 only). |

EnvPool is not a dependency group: `scripts/build_envpool.sh` builds a wheel
into `cpp/build/envpool-dist/`. It isn't in `uv.lock`, so `uv sync` removes
it; reinstall it after a sync with
`uv pip install cpp/build/envpool-dist/envpool-*.whl` (`uv run` leaves it
alone).

## The GTX 1080 (Pascal)

CUDA 13 dropped Pascal GPUs, and several packages default to CUDA 13
builds, so the project pins CUDA 12 builds instead:

- **torch:** the `cuda` group uses PyTorch's cu126 index. CUDA 12.6 is the
  newest version whose torch wheels still run on sm_61.
- **Warp:** PyPI's `warp-lang` is built with CUDA 13 and refuses sm_61
  ("CUDA 13.4 requires sm_75 or higher"). The `warp` group installs the
  `+cu12` wheel from Warp's GitHub release instead.
- **JAX:** the CUDA 12 wheels (`jax[cuda12]`) work as they are.

## `LD_LIBRARY_PATH` and JAX's CUDA libraries

JAX fell back to the CPU with "Unable to load cuSPARSE". The cause was the
AMD AOCC compiler's setup script (`setenv_AOCC.sh`, sourced from
`~/.profile`), which adds `/usr/lib/x86_64-linux-gnu`, `/usr/lib64`,
`/usr/lib32` and `/usr/lib` to `LD_LIBRARY_PATH`. Directories on
`LD_LIBRARY_PATH` take priority over the paths pip wheels set for their own
libraries, so JAX's CUDA plugin loaded Ubuntu's old `libnvJitLink` (12.0,
from the apt package `libnvjitlink12`) instead of the 12.9 copy in its
wheel.

The fix strips those directories again straight after sourcing AOCC. The
loader searches them by default, so nothing else changes:

```sh
LD_LIBRARY_PATH=$(printf '%s' "$LD_LIBRARY_PATH" | tr ':' '\n' \
    | grep -vxE '/usr/lib(/x86_64-linux-gnu|64|32)?' | paste -sd: -)
export LD_LIBRARY_PATH
```

In a shell started before the fix, run GPU jobs with
`env -u LD_LIBRARY_PATH ...`.

## Compilers that fuse multiply-adds

Exact parity with Python needs `a + b * c` rounded as two operations. Every
compiled port had to turn off fusing them into one fused multiply-add:

| Port | Compiler | Switch |
|---|---|---|
| C++ | GCC | `-ffp-contract=off` (in `cpp/CMakeLists.txt` and the EnvPool Bazel build) |
| JAX | XLA (LLVM) | no flag exists; `jax.lax.optimization_barrier` around the product |
| Warp | NVRTC | the `fuse_fp=False` module option |

On the same random data, XLA's and NVRTC's fused results differed from
NumPy in 27,880 of 100,000 cases.

## Build and compile times

- **EnvPool wheel:** about 3 minutes for the first Bazel build (it downloads
  its dependencies); the Bazel cache lives in `~/.cache/bazel`.
- **JAX:** 3–4 s to compile a training configuration.
- **Warp:** about a minute per env configuration on the first build (18 s
  for the reset kernel, 42 s for the step kernel); cached in
  `~/.cache/warp`. The first run of `tests/test_warp_env.py` takes about 10
  minutes and later runs about 30 s.

## Memory

`SubprocVecEnv` with 256 envs ran this 31 GB machine out of memory: each
subprocess loads torch. Keep it to 64 envs or fewer here.
