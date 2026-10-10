// Replays a trace written by `python -m elevator_rl.trace` and checks that the
// C++ env produces exactly the same observations, rewards and truncations.
//
//   elevator_replay full.trace [more.trace ...]
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "elevator/building.hpp"

using namespace elevator;

namespace {

struct Reader {
  std::ifstream in;
  std::string path;
  int line_number = 0;

  // Next line's tag and the rest of it; false at the end of the file.
  bool next(std::string& tag, std::istringstream& rest) {
    std::string line;
    if (!std::getline(in, line)) return false;
    ++line_number;
    rest.clear();
    rest.str(line);
    rest >> tag;
    return true;
  }

  [[noreturn]] void fail(const std::string& message) const {
    std::fprintf(stderr, "%s:%d: %s\n", path.c_str(), line_number, message.c_str());
    std::exit(1);
  }

  void expect(const char* want, std::istringstream& rest) {
    std::string tag;
    if (!next(tag, rest) || tag != want) fail(std::string("expected '") + want + "'");
  }

  std::vector<Arrival> arrivals() {
    std::istringstream rest;
    expect("arrivals", rest);
    int n;
    rest >> n;
    std::vector<Arrival> out(n);
    for (auto& a : out) rest >> a.floor >> a.destination;
    if (!rest) fail("bad arrivals line");
    return out;
  }

  void check_observations(const Env& env) {
    std::istringstream rest;
    expect("custom", rest);
    std::vector<int64_t> custom(env.observation_size(ObsType::Custom));
    env.observe_custom(custom.data());
    for (size_t i = 0; i < custom.size(); ++i) {
      long long want;
      if (!(rest >> want)) fail("custom observation too short");
      if (want != custom[i]) {
        fail("custom[" + std::to_string(i) + "]: python " + std::to_string(want) + ", c++ " +
             std::to_string(custom[i]));
      }
    }
    std::string extra;
    if (rest >> extra) fail("custom observation too long");
    check_floats("box", ObsType::Box, env, rest);
    check_floats("relative", ObsType::Relative, env, rest);
  }

  void check_floats(const char* tag, ObsType type, const Env& env, std::istringstream& rest) {
    expect(tag, rest);
    std::vector<float> values(env.observation_size(type));
    if (type == ObsType::Box) env.observe_box(values.data());
    else env.observe_relative(values.data());
    for (size_t i = 0; i < values.size(); ++i) {
      std::string token;
      if (!(rest >> token)) fail(std::string(tag) + " observation too short");
      // The trace holds the float32 value as an exact double.
      float want = float(std::strtod(token.c_str(), nullptr));
      if (want != values[i]) {
        char message[128];
        std::snprintf(message, sizeof(message), "%s[%zu]: python %.9g, c++ %.9g", tag, i, want,
                      values[i]);
        fail(message);
      }
    }
    std::string extra;
    if (rest >> extra) fail(std::string(tag) + " observation too long");
  }
};

EnvOptions parse_env_line(std::istringstream& rest) {
  EnvOptions options;
  std::string token;
  while (rest >> token) {
    auto eq = token.find('=');
    std::string key = token.substr(0, eq), value = token.substr(eq + 1);
    if (key == "shaped") options.shaped = std::stoi(value) != 0;
    else if (key == "pickup") options.shaping.pickup = std::stod(value);
    else if (key == "empty_serve") options.shaping.empty_serve = std::stod(value);
    else if (key == "progress") options.shaping.progress = std::stod(value);
    else if (key == "waiting") options.shaping.waiting = std::stod(value);
    else if (key == "gamma") options.shaping.gamma = std::stod(value);
    else if (key == "max_steps") options.max_steps = std::stoi(value);
    else if (key == "action_mode") {
      options.action_mode = value == "target" ? ActionMode::Target : ActionMode::Step;
    } else if (key == "observe_direction") options.observe_direction = std::stoi(value) != 0;
    else throw std::invalid_argument("unknown env key: " + key);
  }
  return options;
}

long replay(const char* path) {
  Reader r;
  r.path = path;
  r.in.open(path);
  if (!r.in) r.fail("cannot open");

  std::istringstream rest;
  r.expect("config", rest);
  std::string pairs;
  std::getline(rest, pairs);
  Config config = parse_config(pairs);
  if (" " + to_string(config) != pairs) {
    r.fail("config does not round-trip:\n  python:" + pairs + "\n  c++:    " + to_string(config));
  }

  r.expect("env", rest);
  Env env(config, parse_env_line(rest));

  long steps = 0;
  std::string tag;
  std::vector<int> actions(config.n_lifts);
  while (r.next(tag, rest)) {
    if (tag == "reset") {
      std::vector<std::vector<Arrival>> warmup;
      for (int i = 0; i < 10; ++i) warmup.push_back(r.arrivals());
      env.reset(0, &warmup);
      r.check_observations(env);
    } else if (tag == "step") {
      for (int& a : actions) rest >> a;
      if (!rest) r.fail("bad step line");
      std::vector<Arrival> arrivals = r.arrivals();
      EnvStep out = env.step(actions.data(), &arrivals);
      r.expect("result", rest);
      std::string reward_token;
      int truncated;
      rest >> reward_token >> truncated;
      double reward = std::strtod(reward_token.c_str(), nullptr);
      if (reward != out.reward || bool(truncated) != out.truncated) {
        r.fail("result: python " + reward_token + " " + std::to_string(truncated) + ", c++ " +
               std::to_string(out.reward) + " " + std::to_string(out.truncated));
      }
      r.check_observations(env);
      ++steps;
    } else {
      r.fail("unexpected '" + tag + "'");
    }
  }
  if (env.sim().dropped() != 0) r.fail("arrivals were dropped: the queue cap was hit");
  return steps;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s TRACE [TRACE ...]\n", argv[0]);
    return 2;
  }
  for (int i = 1; i < argc; ++i) {
    long steps = replay(argv[i]);
    std::printf("%s: %ld steps match\n", argv[i], steps);
  }
  return 0;
}
