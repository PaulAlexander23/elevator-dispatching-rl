"""Stable-Baselines3 `VecEnv` over the C++ env loaded as a plain shared library.

The library (libelevator_c.so, elevator_c.dll on Windows) is the flat C API
in cpp/capi/elevator_c.h, built by cpp/CMakeLists.txt next to this file; it
needs no Python headers. `CtypesVecEnv` behaves exactly as `CppVecEnv`, which
goes through the nanobind module instead: same seeds, same trajectories.

What ctypes does not do for you, and this module does by hand:
- declare every signature (`argtypes`/`restype`): ctypes assumes `int`
  otherwise, which truncates a 64-bit pointer returned by `elevator_vec_create`;
- check the C struct layout and the ABI version against the library's;
- turn status codes into exceptions, via `elevator_last_error`;
- free the env handle, which the garbage collector cannot see into;
- keep NumPy arrays alive and in place while C holds pointers to them.
"""

import ctypes
import sys
import weakref
from pathlib import Path

from elevator_rl.cpp_env import CppVecEnv

ABI_VERSION = 1  # ELEVATOR_ABI_VERSION in elevator_c.h
OBS_TYPES = {"custom": 0, "box": 1, "relative": 2}
ACTION_MODES = {"step": 0, "target": 1}

if sys.platform == "win32":
    LIBRARY_NAME = "elevator_c.dll"
elif sys.platform == "darwin":
    LIBRARY_NAME = "libelevator_c.dylib"
else:
    LIBRARY_NAME = "libelevator_c.so"
LIBRARY_PATH = Path(__file__).with_name(LIBRARY_NAME)


class Options(ctypes.Structure):
    """`ElevatorOptions`: field for field, in the same order and types."""

    _fields_ = [
        ("pickup", ctypes.c_double),
        ("empty_serve", ctypes.c_double),
        ("progress", ctypes.c_double),
        ("waiting", ctypes.c_double),
        ("gamma", ctypes.c_double),
        ("shaped", ctypes.c_int32),
        ("max_steps", ctypes.c_int32),
        ("action_mode", ctypes.c_int32),
        ("observe_direction", ctypes.c_int32),
    ]


def _declare(lib):
    """Gives each function its C signature."""
    handle, ptr = ctypes.c_void_p, ctypes.c_void_p
    i32 = ctypes.c_int32
    signatures = {
        "elevator_abi_version": (i32, []),
        "elevator_options_size": (i32, []),
        "elevator_last_error": (ctypes.c_char_p, []),
        "elevator_default_options": (i32, [ctypes.POINTER(Options)]),
        "elevator_vec_create": (
            handle,
            [ctypes.c_char_p, ctypes.POINTER(Options), i32, i32, ctypes.c_uint64],
        ),
        "elevator_vec_destroy": (None, [handle]),
        "elevator_vec_num_envs": (i32, [handle]),
        "elevator_vec_n_lifts": (i32, [handle]),
        "elevator_vec_n_actions": (i32, [handle]),
        "elevator_vec_observation_size": (i32, [handle]),
        "elevator_vec_reset": (i32, [handle, ptr, ptr]),
        "elevator_vec_step": (i32, [handle] + [ptr] * 7),
    }
    for name, (restype, argtypes) in signatures.items():
        function = getattr(lib, name)
        function.restype = restype
        function.argtypes = argtypes
    return lib


def load(path=LIBRARY_PATH):
    """Loads and checks the library. `ctypes.CDLL` releases the GIL during
    each call (`ctypes.PyDLL` would hold it) and uses the C calling convention
    (`ctypes.WinDLL` would use 32-bit Windows' __stdcall)."""
    lib = _declare(ctypes.CDLL(str(path)))
    version = lib.elevator_abi_version()
    if version != ABI_VERSION:
        raise ImportError(f"{path} has ABI version {version}, expected {ABI_VERSION}: rebuild it")
    if lib.elevator_options_size() != ctypes.sizeof(Options):
        raise ImportError(f"{path}: ElevatorOptions has a different layout from Options")
    return lib


try:
    _lib = load()
except OSError:  # not built
    _lib = None


def available():
    return _lib is not None


def _require():
    if _lib is None:
        raise ImportError(
            f"{LIBRARY_NAME} is not built: run "
            "`cmake -S cpp -B cpp/build -DPython_EXECUTABLE=.venv/bin/python && "
            "cmake --build cpp/build`"
        )
    return _lib


def check(status):
    """Raises the library's error message for a failed call."""
    if status != 0:
        raise RuntimeError(_lib.elevator_last_error().decode())


def options_for(env):
    """The C `ElevatorOptions` matching a Python env's constructor arguments."""
    options = Options()
    check(_require().elevator_default_options(ctypes.byref(options)))
    options.shaped = env.reward_shaping is not None
    if env.reward_shaping is not None:
        for name in ("pickup", "empty_serve", "progress", "waiting", "gamma"):
            setattr(options, name, getattr(env.reward_shaping, name))
    options.max_steps = env.max_steps
    options.action_mode = ACTION_MODES[env.action_mode]
    options.observe_direction = env.observe_direction
    return options


def create(env, n_envs, obs_type, seed=0):
    """A C env handle with `env`'s settings. Free it with
    `elevator_vec_destroy`."""
    from elevator_rl.trace import config_line

    lib = _require()
    config = config_line(env.config).removeprefix("config ").encode()
    handle = lib.elevator_vec_create(
        config, ctypes.byref(options_for(env)), n_envs, OBS_TYPES[obs_type], seed
    )
    if not handle:
        raise RuntimeError(lib.elevator_last_error().decode())
    return handle


def _pointer(array):
    assert array.flags.c_contiguous and array.flags.writeable
    return ctypes.c_void_p(array.ctypes.data)


class CtypesVecEnv(CppVecEnv):
    """`CppVecEnv` with the batch stepped through the C API instead of
    nanobind. Takes the same arguments and gives the same results."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The buffers never move (see CppVecEnv), so each call's pointers are
        # built once here, which makes a call about as cheap as ctypes gets.
        self._step_args = (self._handle,) + tuple(
            _pointer(a)
            for a in (
                self._actions,
                self._obs,
                self._rewards,
                self._dones,
                self._terminal,
                self._returns,
                self._lengths,
            )
        )
        self._c_step = _lib.elevator_vec_step

    def _open(self, n_envs, obs_type, seed):
        lib = _require()
        self._handle = ctypes.c_void_p(create(self._reference, n_envs, obs_type, seed))
        # Frees the handle with this object, or on close(). The callback must
        # not refer to self, or self would never be collected.
        self._free = weakref.finalize(self, lib.elevator_vec_destroy, self._handle)
        return lib.elevator_vec_observation_size(self._handle)

    def _reset_batch(self, seeds):
        check(_lib.elevator_vec_reset(self._handle, _pointer(seeds), self._step_args[2]))

    def _step_batch(self):
        if self._c_step(*self._step_args):
            check(1)

    def close(self):
        self._free()
