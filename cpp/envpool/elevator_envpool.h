// The elevator env (cpp/include/elevator/building.hpp) as an EnvPool env.
//
// scripts/build_envpool.sh copies this directory into a checkout of
// sail-sg/envpool as envpool/elevator/, next to EnvPool's own envs, and builds
// a wheel. EnvPool then owns the threads: each worker takes an env id off the
// action queue, calls Step (or Reset), and writes the result into a slot of a
// preallocated state buffer that Python receives as one batch.
//
// Only the float observations (box and relative) are exposed: an EnvPool spec
// fixes the observation dtype at compile time, and training uses relative.
#ifndef ENVPOOL_ELEVATOR_ELEVATOR_ENVPOOL_H_
#define ENVPOOL_ELEVATOR_ELEVATOR_ENVPOOL_H_

#include <stdexcept>
#include <string>

#include "elevator/building.hpp"
#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

namespace elevator_envpool {

// Everything the elevator Env needs, from EnvPool's config dict.
inline elevator::Config BuildingConfig(const std::string& preset,
                                       const std::string& building) {
  // `building` is elevator_rl.trace.config_line without "config "; when set it
  // overrides the preset, so any BuildingConfig can be run.
  return building.empty() ? elevator::preset(preset)
                          : elevator::parse_config(building);
}

inline elevator::ObsType ParseObsType(const std::string& name) {
  if (name == "box") {
    return elevator::ObsType::Box;
  }
  if (name == "relative") {
    return elevator::ObsType::Relative;
  }
  throw std::invalid_argument("EnvPool elevator obs_type must be box or relative, got " +
                              name);
}

template <typename Config>
elevator::EnvOptions Options(const Config& conf) {
  elevator::EnvOptions options;
  options.shaped = conf["shaped"_];
  options.shaping.pickup = conf["pickup"_];
  options.shaping.empty_serve = conf["empty_serve"_];
  options.shaping.progress = conf["progress"_];
  options.shaping.waiting = conf["waiting"_];
  options.shaping.gamma = conf["gamma"_];
  options.max_steps = conf["max_episode_steps"_];
  const std::string& mode = conf["action_mode"_];
  if (mode == "step") {
    options.action_mode = elevator::ActionMode::Step;
  } else if (mode == "target") {
    options.action_mode = elevator::ActionMode::Target;
  } else {
    throw std::invalid_argument("action_mode must be step or target, got " + mode);
  }
  options.observe_direction = conf["observe_direction"_];
  return options;
}

class ElevatorEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    // Strings must be std::string, not const char* (see EnvPool's new_env docs).
    return MakeDict("preset"_.Bind(std::string("full")),
                    "building"_.Bind(std::string("")),
                    "obs_type"_.Bind(std::string("relative")),
                    "shaped"_.Bind(false), "pickup"_.Bind(1.0),
                    "empty_serve"_.Bind(0.5), "progress"_.Bind(0.0),
                    "waiting"_.Bind(0.0), "gamma"_.Bind(0.99),
                    "action_mode"_.Bind(std::string("step")),
                    "observe_direction"_.Bind(false));
  }

  // The spec sizes come from a throwaway Env, so they cannot drift from it.
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    elevator::Env env(BuildingConfig(conf["preset"_], conf["building"_]),
                      Options(conf));
    int size = env.observation_size(ParseObsType(conf["obs_type"_]));
    return MakeDict("obs"_.Bind(Spec<float>({size})),
                    "info:delivered"_.Bind(Spec<int>({-1})),
                    "info:boarded"_.Bind(Spec<int>({-1})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    elevator::Env env(BuildingConfig(conf["preset"_], conf["building"_]),
                      Options(conf));
    int lifts = env.sim().config().n_lifts;
    // One action per lift; -1 is EnvPool's player axis (one player here).
    return MakeDict("action"_.Bind(
        Spec<int>({-1, lifts}, {0, env.n_actions_per_lift() - 1})));
  }
};

using ElevatorEnvSpec = EnvSpec<ElevatorEnvFns>;

class ElevatorEnv : public Env<ElevatorEnvSpec> {
 protected:
  elevator::Env env_;
  elevator::ObsType obs_type_;
  bool done_{true};

 public:
  ElevatorEnv(const Spec& spec, int env_id)
      : Env<ElevatorEnvSpec>(spec, env_id),
        env_(BuildingConfig(spec.config["preset"_], spec.config["building"_]),
             Options(spec.config)),
        obs_type_(ParseObsType(spec.config["obs_type"_])) {}

  bool IsDone() override { return done_; }

  void Reset() override {
    // gen_ is EnvPool's per-env std::mt19937, seeded with seed + env_id, so
    // each episode gets a fresh, reproducible seed.
    env_.reset(gen_());
    done_ = false;
    WriteState(0.0F, 0, 0);
  }

  void Step(const Action& action) override {
    const int* actions = static_cast<const int*>(action["action"_].Data());
    elevator::EnvStep out = env_.step(actions);
    done_ = out.truncated;
    WriteState(static_cast<float>(out.reward), out.delivered, out.boarded);
  }

 private:
  void WriteState(float reward, int delivered, int boarded) {
    State state = Allocate();
    auto* obs = static_cast<float*>(state["obs"_].Data());
    if (obs_type_ == elevator::ObsType::Box) {
      env_.observe_box(obs);
    } else {
      env_.observe_relative(obs);
    }
    state["reward"_] = reward;
    state["info:delivered"_] = delivered;
    state["info:boarded"_] = boarded;
  }
};

using ElevatorEnvPool = AsyncEnvPool<ElevatorEnv>;

}  // namespace elevator_envpool

#endif  // ENVPOOL_ELEVATOR_ELEVATOR_ENVPOOL_H_
