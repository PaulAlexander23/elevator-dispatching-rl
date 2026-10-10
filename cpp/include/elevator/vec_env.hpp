// Many Envs stepped together, writing into caller-owned arrays: the batched
// API the Python binding exposes, so one call from Python steps every env.
#pragma once

#include <cstdint>
#include <vector>

#include "elevator/building.hpp"

namespace elevator {

class VecEnv {
 public:
  // Env i's first episode seeds its seed stream with `seed + i`; later episodes
  // continue that stream, so a run is reproducible from `seed` alone.
  VecEnv(const Config& config, const EnvOptions& options, int n_envs, ObsType obs_type,
         uint64_t seed = 0);

  int num_envs() const { return int(envs_.size()); }
  int n_lifts() const { return envs_[0].sim().config().n_lifts; }
  int observation_size() const { return obs_size_; }
  ObsType obs_type() const { return obs_type_; }
  // Observation element type: int64 for Custom, float32 otherwise.
  bool integer_observations() const { return obs_type_ == ObsType::Custom; }

  // `seeds` holds one entry per env; a negative entry keeps that env's stream.
  // `obs` is num_envs x observation_size.
  void reset(const int64_t* seeds, void* obs);

  // `actions` is num_envs x n_lifts. Envs whose episode ends are reset at once:
  // `obs` gets the new episode's first observation, and `terminal_obs` (same
  // shape) the last observation of the old one, with its return and length.
  // Rows of `terminal_obs`, `episode_returns` and `episode_lengths` for envs
  // that did not finish are left untouched.
  void step(const int64_t* actions, void* obs, float* rewards, uint8_t* dones,
            void* terminal_obs, float* episode_returns, int32_t* episode_lengths);

  Env& env(int i) { return envs_[i]; }

 private:
  void observe(int i, void* base) const;
  void reset_one(int i);

  ObsType obs_type_;
  int obs_size_;
  std::vector<Env> envs_;
  std::vector<Rng> seeders_;
  std::vector<double> returns_;
  std::vector<int32_t> lengths_;
};

}  // namespace elevator
