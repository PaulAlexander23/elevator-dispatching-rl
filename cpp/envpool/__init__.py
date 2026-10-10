"""The elevator dispatching env (elevator-dispatching-rl) in EnvPool."""

from envpool.python.api import py_env

from .elevator_envpool import _ElevatorEnvPool, _ElevatorEnvSpec

ElevatorEnvSpec, ElevatorDMEnvPool, ElevatorGymnasiumEnvPool = py_env(
    _ElevatorEnvSpec, _ElevatorEnvPool
)

__all__ = ["ElevatorEnvSpec", "ElevatorDMEnvPool", "ElevatorGymnasiumEnvPool"]
