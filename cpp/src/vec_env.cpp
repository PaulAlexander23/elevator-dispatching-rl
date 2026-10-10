#include "elevator/vec_env.hpp"

#include <stdexcept>

namespace elevator {

VecEnv::VecEnv(const Config& config, const EnvOptions& options, int n_envs, ObsType obs_type,
               uint64_t seed)
    : obs_type_(obs_type) {
  if (n_envs < 1) throw std::invalid_argument("n_envs must be at least 1");
  envs_.reserve(n_envs);
  for (int i = 0; i < n_envs; ++i) {
    envs_.emplace_back(config, options);
    seeders_.emplace_back(seed + uint64_t(i));
  }
  obs_size_ = envs_[0].observation_size(obs_type);
  returns_.assign(n_envs, 0.0);
  lengths_.assign(n_envs, 0);
}

void VecEnv::observe(int i, void* base) const {
  const Env& env = envs_[i];
  switch (obs_type_) {
    case ObsType::Custom:
      env.observe_custom(static_cast<int64_t*>(base) + size_t(i) * obs_size_);
      break;
    case ObsType::Box:
      env.observe_box(static_cast<float*>(base) + size_t(i) * obs_size_);
      break;
    case ObsType::Relative:
      env.observe_relative(static_cast<float*>(base) + size_t(i) * obs_size_);
      break;
  }
}

void VecEnv::reset_one(int i) {
  envs_[i].reset(seeders_[i].next());
  returns_[i] = 0.0;
  lengths_[i] = 0;
}

void VecEnv::reset(const int64_t* seeds, void* obs) {
  for (int i = 0; i < num_envs(); ++i) {
    if (seeds && seeds[i] >= 0) seeders_[i].seed_with(uint64_t(seeds[i]));
    reset_one(i);
    observe(i, obs);
  }
}

void VecEnv::step(const int64_t* actions, void* obs, float* rewards, uint8_t* dones,
                  void* terminal_obs, float* episode_returns, int32_t* episode_lengths) {
  const int lifts = n_lifts();
  int lift_actions[kMaxLifts];
  for (int i = 0; i < num_envs(); ++i) {
    for (int l = 0; l < lifts; ++l) lift_actions[l] = int(actions[size_t(i) * lifts + l]);
    EnvStep out = envs_[i].step(lift_actions);
    returns_[i] += out.reward;
    ++lengths_[i];
    rewards[i] = float(out.reward);
    dones[i] = out.truncated;
    if (out.truncated) {
      observe(i, terminal_obs);
      episode_returns[i] = float(returns_[i]);
      episode_lengths[i] = lengths_[i];
      reset_one(i);
    }
    observe(i, obs);
  }
}

}  // namespace elevator
