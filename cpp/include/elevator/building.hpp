// C++ port of elevator_rl.building (BuildingSim) and elevator_rl.building_env
// (BuildingEnv).
//
// State is fixed-size: no allocation after construction, so many envs can be
// stepped side by side, and the same layout carries over to JAX or Warp.
// The dynamics match the Python version exactly given the same arrivals; the
// random arrivals themselves come from this file's own generator, so they
// match Python only in distribution.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace elevator {

constexpr int kMaxFloors = 64;
constexpr int kMaxLifts = 16;
constexpr int kMaxCapacity = 32;
// Longest queue per floor. The longest seen in Python is about 310 (the
// lobby during the "full" preset's up-peak); arrivals beyond this are dropped
// and counted in Sim::dropped(), which is never reset.
constexpr int kMaxQueue = 512;

enum class Traffic { Uniform, UpPeak, DownPeak, Lunch, Day };
enum Status { kIdle = 0, kMoving = 1, kDoors = 2 };
enum Action { kUp = 0, kDown = 1, kServe = 2, kServeUp = 2, kServeDown = 3 };
enum class ObsType { Custom, Box, Relative };

struct Config {
  int n_floors = 10;
  int n_lifts = 1;
  int lift_capacity = 8;
  double arrival_probability = 0.1;
  Traffic traffic = Traffic::Uniform;
  bool hall_calls = false;
  bool kinematics = false;
  double floor_height = 3.5;
  double max_speed = 2.5;
  double acceleration = 1.0;
  double door_time = 4.0;
  double step_seconds = 1.0;
  int substeps = 10;

  int n_actions() const { return hall_calls ? 4 : 3; }
  // Throws std::invalid_argument when a field is out of range.
  void validate() const;
};

// The presets of elevator_rl.building.PRESETS, by name.
Config preset(const std::string& name);
// "key=value" pairs in the same order and format as elevator_rl.trace.config_line
// (without its leading "config ").
std::string to_string(const Config& config);
Config parse_config(const std::string& pairs);

// xoshiro256** seeded with splitmix64: small, fast and the same on every platform.
class Rng {
 public:
  explicit Rng(uint64_t seed = 0) { seed_with(seed); }
  void seed_with(uint64_t seed);
  uint64_t next();
  double uniform();             // [0, 1)
  int integer(int n);           // [0, n)
  int poisson(double mean);     // Knuth's method; fine for the small means here
  int categorical(const double* weights, int n);  // weights need not sum to 1

 private:
  uint64_t s_[4];
};

struct Arrival {
  int floor;
  int destination;
};

struct Lift {
  // In floors and seconds, as in Python.
  double position = 0.0;
  double velocity = 0.0;
  int target = 0;
  double door_timer = 0.0;
  int n_passengers = 0;
  std::array<uint8_t, kMaxCapacity> passengers{};  // destination floors

  int floor() const;
  Status status() const;
};

struct FloorQueue {
  int size = 0;
  std::array<uint8_t, kMaxQueue> destinations{};  // in arrival order
};

struct StepResult {
  std::array<int, kMaxLifts> delivered{};
  std::array<int, kMaxLifts> boarded{};
  std::array<bool, kMaxLifts> served{};
};

class Sim {
 public:
  explicit Sim(const Config& config, uint64_t seed = 0);

  void reset();
  void seed(uint64_t seed) { rng_.seed_with(seed); }

  // Draws this step's arrivals into `out` (cleared first).
  void draw_arrivals(double episode_fraction, std::vector<Arrival>& out);
  void add_arrivals(const std::vector<Arrival>& arrivals);
  // Expected arrivals per floor and each floor's destination distribution
  // (row-major n_floors x n_floors), for the non-uniform traffic profiles.
  void arrival_model(double episode_fraction, double* rates, double* destinations) const;

  StepResult apply_actions(const int* actions);

  const Config& config() const { return config_; }
  const Lift& lift(int i) const { return lifts_[i]; }
  Lift& lift(int i) { return lifts_[i]; }
  const FloorQueue& queue(int floor) const { return queues_[floor]; }
  FloorQueue& queue(int floor) { return queues_[floor]; }
  long dropped() const { return dropped_; }

 private:
  void move(Lift& lift, int direction);
  void serve(Lift& lift, int direction, int& delivered, int& boarded);
  void integrate(Lift& lift);

  Config config_;
  Rng rng_;
  double max_speed_, acceleration_, dt_;  // in floors and seconds
  std::array<Lift, kMaxLifts> lifts_{};
  std::array<FloorQueue, kMaxFloors> queues_{};
  long dropped_ = 0;
  // origins_[k][f], destinations_[k][f][g] for the up-peak, interfloor and
  // down-peak patterns.
  std::vector<double> origins_, destinations_;
  // Scratch space for draw_arrivals: one destination row.
  std::vector<double> joint_;
};

// Training-only reward terms, as building_env.Shaping.
struct Shaping {
  double pickup = 1.0;
  double empty_serve = 0.5;
  double progress = 0.0;
  double waiting = 0.0;
  double gamma = 0.99;
};

enum class ActionMode { Step, Target };

// The BuildingEnv constructor arguments other than the config.
struct EnvOptions {
  bool shaped = false;  // false = reward is passengers delivered only
  Shaping shaping{};
  int max_steps = 200;
  ActionMode action_mode = ActionMode::Step;
  bool observe_direction = false;
};

// Per-lift and per-floor-offset widths of the relative observation.
constexpr int kRelativeLiftFields = 10;
constexpr int kRelativeFloorFields = 5;

struct EnvStep {
  double reward = 0.0;
  bool truncated = false;
  int delivered = 0;
  int boarded = 0;
};

class Env {
 public:
  explicit Env(const Config& config, const EnvOptions& options = EnvOptions{});

  // Resets and runs the 10 warm-up arrival rounds. With `warmup` (10 rounds),
  // those arrivals are used instead of drawing them, to replay a trace.
  void reset(uint64_t seed, const std::vector<std::vector<Arrival>>* warmup = nullptr);
  // `actions` holds one action per lift (a target per lift in target mode).
  // With `arrivals`, they replace the draw.
  EnvStep step(const int* actions, const std::vector<Arrival>* arrivals = nullptr);

  int observation_size(ObsType type) const;
  int n_actions_per_lift() const;
  void observe_custom(int64_t* out) const;
  void observe_box(float* out) const;
  void observe_relative(float* out) const;

  const Sim& sim() const { return sim_; }
  Sim& sim() { return sim_; }
  const EnvOptions& options() const { return options_; }
  int steps() const { return steps_; }

 private:
  struct Goal {
    bool set = false;
    int floor = 0;
    int direction = 1;
  };
  void primitive_actions(const int* actions, int* out);
  double potential() const;
  int extra_fields() const;  // free flags and directions at the end of custom/box

  Sim sim_;
  EnvOptions options_;
  int steps_ = 0;
  double last_potential_ = 0.0;
  std::array<Goal, kMaxLifts> goals_{};
  std::array<int, kMaxLifts> directions_{};
  std::vector<Arrival> arrivals_;
};

}  // namespace elevator
