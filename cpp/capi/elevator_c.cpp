// The C API of elevator_c.h, over elevator::VecEnv. Every entry point catches
// all exceptions: one escaping through a C frame (or into ctypes) would call
// std::terminate and take the host process down.
#include "elevator_c.h"

#include <cstddef>
#include <exception>
#include <new>
#include <stdexcept>
#include <string>

#include "elevator/building.hpp"
#include "elevator/vec_env.hpp"

using namespace elevator;

// The handle is the C++ object itself; the C side only sees an opaque pointer.
struct ElevatorVecEnv {
  VecEnv vec;
  int n_actions;
};

static_assert(sizeof(ElevatorOptions) == 5 * sizeof(double) + 4 * sizeof(int32_t),
              "ElevatorOptions must have no padding");

namespace {

// One per thread, so concurrent callers don't overwrite each other's error.
thread_local std::string last_error;

int32_t fail(const char* message) {
  last_error = message;
  return ELEVATOR_ERROR;
}

// Runs `body`, turning any exception into ELEVATOR_ERROR and a message.
template <typename F>
int32_t guarded(F&& body) {
  try {
    body();
    return ELEVATOR_OK;
  } catch (const std::bad_alloc&) {
    return fail("out of memory");
  } catch (const std::exception& e) {
    return fail(e.what());
  } catch (...) {
    return fail("unknown C++ exception");
  }
}

EnvOptions to_options(const ElevatorOptions& o) {
  if (o.action_mode != ELEVATOR_ACTION_STEP && o.action_mode != ELEVATOR_ACTION_TARGET) {
    throw std::invalid_argument("action_mode must be ELEVATOR_ACTION_STEP or _TARGET");
  }
  if (o.max_steps < 1) throw std::invalid_argument("max_steps must be at least 1");
  EnvOptions options;
  options.shaped = o.shaped != 0;
  options.shaping = Shaping{o.pickup, o.empty_serve, o.progress, o.waiting, o.gamma};
  options.max_steps = o.max_steps;
  options.action_mode = o.action_mode == ELEVATOR_ACTION_TARGET ? ActionMode::Target
                                                                : ActionMode::Step;
  options.observe_direction = o.observe_direction != 0;
  return options;
}

ObsType to_obs_type(int32_t obs_type) {
  switch (obs_type) {
    case ELEVATOR_OBS_CUSTOM: return ObsType::Custom;
    case ELEVATOR_OBS_BOX: return ObsType::Box;
    case ELEVATOR_OBS_RELATIVE: return ObsType::Relative;
  }
  throw std::invalid_argument("obs_type must be one of ELEVATOR_OBS_*");
}

}  // namespace

extern "C" {

int32_t elevator_abi_version(void) { return ELEVATOR_ABI_VERSION; }

int32_t elevator_options_size(void) { return int32_t(sizeof(ElevatorOptions)); }

const char* elevator_last_error(void) { return last_error.c_str(); }

int32_t elevator_default_options(ElevatorOptions* out) {
  if (!out) return fail("out is NULL");
  EnvOptions d;
  *out = ElevatorOptions{d.shaping.pickup, d.shaping.empty_serve, d.shaping.progress,
                         d.shaping.waiting, d.shaping.gamma,     int32_t(d.shaped),
                         d.max_steps,       ELEVATOR_ACTION_STEP, int32_t(d.observe_direction)};
  return ELEVATOR_OK;
}

ElevatorVecEnv* elevator_vec_create(const char* config, const ElevatorOptions* options,
                                    int32_t n_envs, int32_t obs_type, uint64_t seed) {
  if (!config || !options) {
    fail("config and options must not be NULL");
    return nullptr;
  }
  ElevatorVecEnv* env = nullptr;
  guarded([&] {
    Config c = parse_config(config);
    c.validate();
    env = new ElevatorVecEnv{VecEnv(c, to_options(*options), n_envs, to_obs_type(obs_type), seed),
                             0};
    env->n_actions = env->vec.env(0).n_actions_per_lift();
  });
  return env;
}

void elevator_vec_destroy(ElevatorVecEnv* env) { delete env; }

int32_t elevator_vec_num_envs(const ElevatorVecEnv* env) { return env ? env->vec.num_envs() : -1; }

int32_t elevator_vec_n_lifts(const ElevatorVecEnv* env) { return env ? env->vec.n_lifts() : -1; }

int32_t elevator_vec_n_actions(const ElevatorVecEnv* env) { return env ? env->n_actions : -1; }

int32_t elevator_vec_observation_size(const ElevatorVecEnv* env) {
  return env ? env->vec.observation_size() : -1;
}

int32_t elevator_vec_reset(ElevatorVecEnv* env, const int64_t* seeds, void* obs) {
  if (!env || !obs) return fail("env and obs must not be NULL");
  return guarded([&] { env->vec.reset(seeds, obs); });
}

int32_t elevator_vec_step(ElevatorVecEnv* env, const int64_t* actions, void* obs, float* rewards,
                          uint8_t* dones, void* terminal_obs, float* episode_returns,
                          int32_t* episode_lengths) {
  if (!env || !actions || !obs || !rewards || !dones || !terminal_obs || !episode_returns ||
      !episode_lengths) {
    return fail("no argument may be NULL");
  }
  // A C caller can pass anything, and a target floor out of range would index
  // past the env's arrays, so check before stepping anything.
  const size_t n = size_t(env->vec.num_envs()) * size_t(env->vec.n_lifts());
  for (size_t i = 0; i < n; ++i) {
    if (actions[i] < 0 || actions[i] >= env->n_actions) return fail("action out of range");
  }
  return guarded([&] {
    env->vec.step(actions, obs, rewards, dones, terminal_obs, episode_returns, episode_lengths);
  });
}

}  // extern "C"
