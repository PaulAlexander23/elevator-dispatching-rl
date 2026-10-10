// Single-threaded throughput of the C++ env under a random policy, plus the
// statistics the Python tests compare against (reward, arrivals per step).
//
//   elevator_bench --preset full --steps 1000000 --seed 0
//   elevator_bench --preset full --print-config
//
// Prints key=value pairs, one per line.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "elevator/building.hpp"

using namespace elevator;

int main(int argc, char** argv) {
  std::string preset_name = "full";
  long n_steps = 1'000'000;
  uint64_t seed = 0;
  int max_steps = 200;
  bool print_config = false;
  for (int i = 1; i < argc; ++i) {
    auto arg = [&](const char* name) { return std::strcmp(argv[i], name) == 0 && i + 1 < argc; };
    if (arg("--preset")) preset_name = argv[++i];
    else if (arg("--steps")) n_steps = std::atol(argv[++i]);
    else if (arg("--seed")) seed = std::strtoull(argv[++i], nullptr, 10);
    else if (arg("--max-steps")) max_steps = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--print-config") == 0) print_config = true;
    else {
      std::fprintf(stderr,
                   "usage: %s [--preset NAME] [--steps N] [--seed S] [--max-steps N] "
                   "[--print-config]\n",
                   argv[0]);
      return 2;
    }
  }

  Config config = preset(preset_name);
  if (print_config) {
    std::printf("config %s\n", to_string(config).c_str());
    return 0;
  }

  Env env(config, /*reward_shaping=*/false, max_steps);
  Rng policy(seed ^ 0xabcdefULL);
  std::vector<int> actions(config.n_lifts);
  std::vector<int64_t> obs(env.observation_size(ObsType::Custom));
  uint64_t episode_seed = seed;
  env.reset(episode_seed++);

  double episode_reward = 0.0, reward_sum = 0.0;
  long episodes = 0, arrivals = 0;
  auto queued = [&] {
    long total = 0;
    for (int f = 0; f < config.n_floors; ++f) total += env.sim().queue(f).size;
    for (int i = 0; i < config.n_lifts; ++i) total += env.sim().lift(i).n_passengers;
    return total;
  };
  long before = queued();

  auto t0 = std::chrono::steady_clock::now();
  for (long s = 0; s < n_steps; ++s) {
    for (int& a : actions) a = policy.integer(config.n_actions());
    EnvStep out = env.step(actions.data());
    env.observe_custom(obs.data());
    // Arrivals = growth in passengers present + those delivered this step.
    long now = queued();
    arrivals += now - before + out.delivered;
    episode_reward += out.reward;
    if (out.truncated) {
      reward_sum += episode_reward;
      episode_reward = 0.0;
      ++episodes;
      env.reset(episode_seed++);
      now = queued();
    }
    before = now;
  }
  double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  std::printf("preset=%s\n", preset_name.c_str());
  std::printf("steps=%ld\n", n_steps);
  std::printf("seconds=%.6f\n", seconds);
  std::printf("steps_per_second=%.1f\n", n_steps / seconds);
  std::printf("microseconds_per_step=%.4f\n", 1e6 * seconds / n_steps);
  std::printf("episodes=%ld\n", episodes);
  std::printf("mean_episode_reward=%.6f\n", episodes ? reward_sum / episodes : 0.0);
  std::printf("arrivals_per_step=%.6f\n", double(arrivals) / n_steps);
  std::printf("dropped=%ld\n", env.sim().dropped());
  return 0;
}
