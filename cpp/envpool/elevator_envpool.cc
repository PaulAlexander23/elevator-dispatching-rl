// pybind11 module elevator_envpool: EnvPool's Python wrappers around the
// elevator env spec and pool (see elevator_envpool.h).
#include "envpool/elevator/elevator_envpool.h"

#include "envpool/core/py_envpool.h"

using ElevatorEnvSpec = PyEnvSpec<elevator_envpool::ElevatorEnvSpec>;
using ElevatorEnvPool = PyEnvPool<elevator_envpool::ElevatorEnvPool>;

PYBIND11_MODULE(elevator_envpool, m) {
  REGISTER(m, ElevatorEnvSpec, ElevatorEnvPool)
}
