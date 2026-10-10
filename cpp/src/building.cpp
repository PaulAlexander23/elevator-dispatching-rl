#include "elevator/building.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace elevator {

namespace {

// Mixture weights over the (up-peak, interfloor, down-peak) patterns.
struct Weights {
  double up, inter, down;
};
constexpr Weights kUpPeak{0.8, 0.2, 0.0};
constexpr Weights kDownPeak{0.0, 0.2, 0.8};
constexpr Weights kLunch{0.4, 0.2, 0.4};
constexpr Weights kInterfloor{0.0, 1.0, 0.0};

struct Keyframe {
  double t;
  Weights w;
};
// "day": linearly interpolated, as _DAY_KEYFRAMES in building.py.
constexpr Keyframe kDay[] = {
    {0.0, kUpPeak}, {0.3, kUpPeak}, {0.5, kLunch},
    {0.7, kInterfloor}, {0.9, kDownPeak}, {1.0, kDownPeak},
};

Weights traffic_weights(Traffic traffic, double fraction) {
  switch (traffic) {
    case Traffic::Uniform: return kInterfloor;
    case Traffic::UpPeak: return kUpPeak;
    case Traffic::DownPeak: return kDownPeak;
    case Traffic::Lunch: return kLunch;
    case Traffic::Day: break;
  }
  double t = std::min(std::max(fraction, 0.0), 1.0);
  constexpr int n = sizeof(kDay) / sizeof(kDay[0]);
  for (int i = 0; i + 1 < n; ++i) {
    const Keyframe& a = kDay[i];
    const Keyframe& b = kDay[i + 1];
    if (t <= b.t) {
      double x = b.t > a.t ? (t - a.t) / (b.t - a.t) : 1.0;
      return {(1 - x) * a.w.up + x * b.w.up, (1 - x) * a.w.inter + x * b.w.inter,
              (1 - x) * a.w.down + x * b.w.down};
    }
  }
  return kDay[n - 1].w;
}

const char* traffic_name(Traffic traffic) {
  switch (traffic) {
    case Traffic::Uniform: return "uniform";
    case Traffic::UpPeak: return "up_peak";
    case Traffic::DownPeak: return "down_peak";
    case Traffic::Lunch: return "lunch";
    case Traffic::Day: return "day";
  }
  return "?";
}

Traffic parse_traffic(const std::string& name) {
  for (Traffic t : {Traffic::Uniform, Traffic::UpPeak, Traffic::DownPeak, Traffic::Lunch,
                    Traffic::Day}) {
    if (name == traffic_name(t)) return t;
  }
  throw std::invalid_argument("unknown traffic profile: " + name);
}

// Shortest round-trip form, written like Python's float repr ("1.0", "0.1").
std::string float_repr(double value) {
  char buffer[64];
  auto [end, ec] = std::to_chars(buffer, buffer + sizeof(buffer), value);
  std::string text(buffer, end);
  if (text.find_first_of(".en") == std::string::npos) text += ".0";
  return text;
}

uint64_t splitmix64(uint64_t& x) {
  uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

}  // namespace

void Config::validate() const {
  if (n_floors < 2 || n_floors > kMaxFloors) throw std::invalid_argument("n_floors out of range");
  if (n_lifts < 1 || n_lifts > kMaxLifts) throw std::invalid_argument("n_lifts out of range");
  if (lift_capacity < 1 || lift_capacity > kMaxCapacity) {
    throw std::invalid_argument("lift_capacity out of range");
  }
  if (substeps < 1) throw std::invalid_argument("substeps must be at least 1");
}

Config preset(const std::string& name) {
  // Each preset adds one feature to the one before, as in building.py.
  Config c;
  if (name == "original") return c;
  c.n_floors = 30;
  if (name == "tall") return c;
  c.n_lifts = 4;
  if (name == "multi") return c;
  c.kinematics = true;
  if (name == "kinematic") return c;
  c.hall_calls = true;
  if (name == "hall_calls") return c;
  c.traffic = Traffic::Day;
  if (name == "full") return c;
  throw std::invalid_argument("unknown preset: " + name);
}

std::string to_string(const Config& c) {
  std::ostringstream out;
  out << "n_floors=" << c.n_floors << " n_lifts=" << c.n_lifts
      << " lift_capacity=" << c.lift_capacity
      << " arrival_probability=" << float_repr(c.arrival_probability)
      << " traffic=" << traffic_name(c.traffic) << " hall_calls=" << int(c.hall_calls)
      << " kinematics=" << int(c.kinematics) << " floor_height=" << float_repr(c.floor_height)
      << " max_speed=" << float_repr(c.max_speed)
      << " acceleration=" << float_repr(c.acceleration)
      << " door_time=" << float_repr(c.door_time)
      << " step_seconds=" << float_repr(c.step_seconds) << " substeps=" << c.substeps;
  return out.str();
}

Config parse_config(const std::string& pairs) {
  Config c;
  std::istringstream in(pairs);
  std::string token;
  while (in >> token) {
    auto eq = token.find('=');
    if (eq == std::string::npos) throw std::invalid_argument("expected key=value: " + token);
    std::string key = token.substr(0, eq);
    std::string value = token.substr(eq + 1);
    if (key == "n_floors") c.n_floors = std::stoi(value);
    else if (key == "n_lifts") c.n_lifts = std::stoi(value);
    else if (key == "lift_capacity") c.lift_capacity = std::stoi(value);
    else if (key == "arrival_probability") c.arrival_probability = std::stod(value);
    else if (key == "traffic") c.traffic = parse_traffic(value);
    else if (key == "hall_calls") c.hall_calls = std::stoi(value) != 0;
    else if (key == "kinematics") c.kinematics = std::stoi(value) != 0;
    else if (key == "floor_height") c.floor_height = std::stod(value);
    else if (key == "max_speed") c.max_speed = std::stod(value);
    else if (key == "acceleration") c.acceleration = std::stod(value);
    else if (key == "door_time") c.door_time = std::stod(value);
    else if (key == "step_seconds") c.step_seconds = std::stod(value);
    else if (key == "substeps") c.substeps = std::stoi(value);
    else throw std::invalid_argument("unknown config key: " + key);
  }
  c.validate();
  return c;
}

// --- Rng ---------------------------------------------------------------------

void Rng::seed_with(uint64_t seed) {
  for (auto& s : s_) s = splitmix64(seed);
}

uint64_t Rng::next() {
  uint64_t result = rotl(s_[1] * 5, 7) * 9;
  uint64_t t = s_[1] << 17;
  s_[2] ^= s_[0];
  s_[3] ^= s_[1];
  s_[1] ^= s_[2];
  s_[0] ^= s_[3];
  s_[2] ^= t;
  s_[3] = rotl(s_[3], 45);
  return result;
}

double Rng::uniform() { return double(next() >> 11) * 0x1.0p-53; }

int Rng::integer(int n) { return int(((next() >> 32) * uint64_t(n)) >> 32); }

int Rng::poisson(double mean) {
  if (mean <= 0) return 0;
  double limit = std::exp(-mean);
  int k = 0;
  double p = 1.0;
  do {
    ++k;
    p *= uniform();
  } while (p > limit);
  return k - 1;
}

int Rng::categorical(const double* weights, int n) {
  double total = 0.0;
  for (int i = 0; i < n; ++i) total += weights[i];
  double u = uniform() * total;
  double cumulative = 0.0;
  const double* probabilities = weights;
  int last = 0;
  for (int i = 0; i < n; ++i) {
    if (probabilities[i] <= 0) continue;
    cumulative += probabilities[i];
    last = i;
    if (u < cumulative) return i;
  }
  return last;  // rounding left u just above the total
}

// --- Lift --------------------------------------------------------------------

int Lift::floor() const {
  // Python's round() rounds halves to even; so does nearbyint by default.
  return int(std::nearbyint(position));
}

Status Lift::status() const {
  if (door_timer > 0) return kDoors;
  if (position != target || velocity != 0) return kMoving;
  return kIdle;
}

// --- Sim ---------------------------------------------------------------------

Sim::Sim(const Config& config, uint64_t seed) : config_(config), rng_(seed) {
  config_.validate();
  const Config& c = config_;
  max_speed_ = c.max_speed / c.floor_height;
  acceleration_ = c.acceleration / c.floor_height;
  dt_ = c.step_seconds / c.substeps;

  int n = c.n_floors;
  double upper = 1.0 / (n - 1);
  origins_.assign(3 * n, 0.0);
  destinations_.assign(3 * n * n, 0.0);
  auto dest = [&](int k, int f, int g) -> double& { return destinations_[(k * n + f) * n + g]; };
  // Up-peak: lobby to any upper floor.
  origins_[0] = 1.0;
  for (int g = 1; g < n; ++g) dest(0, 0, g) = upper;
  // Interfloor: any floor to any other floor.
  for (int f = 0; f < n; ++f) {
    origins_[n + f] = 1.0 / n;
    for (int g = 0; g < n; ++g) dest(1, f, g) = f == g ? 0.0 : upper;
  }
  // Down-peak: any upper floor to the lobby.
  for (int f = 1; f < n; ++f) {
    origins_[2 * n + f] = upper;
    dest(2, f, 0) = 1.0;
  }
  joint_.assign(n, 0.0);
  reset();
}

void Sim::reset() {
  for (auto& lift : lifts_) lift = Lift{};
  for (auto& queue : queues_) queue.size = 0;
}

void Sim::arrival_model(double fraction, double* rates, double* destinations) const {
  const int n = config_.n_floors;
  Weights w = traffic_weights(config_.traffic, fraction);
  const double weights[3] = {w.up, w.inter, w.down};
  const double total = config_.arrival_probability * n;
  for (int f = 0; f < n; ++f) {
    double mix[3];
    double rate = 0.0;
    for (int k = 0; k < 3; ++k) {
      mix[k] = weights[k] * origins_[k * n + f];
      rate += mix[k];
    }
    rates[f] = total * rate;
    double sum = 0.0;
    double* row = destinations + f * n;
    for (int g = 0; g < n; ++g) {
      row[g] = 0.0;
      for (int k = 0; k < 3; ++k) row[g] += mix[k] * destinations_[(k * n + f) * n + g];
      sum += row[g];
    }
    for (int g = 0; g < n; ++g) row[g] = sum > 0 ? row[g] / sum : 1.0 / n;
  }
}

void Sim::draw_arrivals(double fraction, std::vector<Arrival>& out) {
  out.clear();
  const int n = config_.n_floors;
  if (config_.traffic == Traffic::Uniform) {
    for (int floor = 0; floor < n; ++floor) {
      if (rng_.uniform() < config_.arrival_probability) {
        int destination = rng_.integer(n - 1);
        if (destination >= floor) ++destination;
        out.push_back({floor, destination});
      }
    }
    return;
  }
  // As arrival_model, but a floor's destination row is only built when
  // someone arrives there: most floors see nobody in a given step.
  Weights w = traffic_weights(config_.traffic, fraction);
  const double weights[3] = {w.up, w.inter, w.down};
  const double total = config_.arrival_probability * n;
  double* row = joint_.data();
  for (int floor = 0; floor < n; ++floor) {
    double mix[3];
    double rate = 0.0;
    for (int k = 0; k < 3; ++k) {
      mix[k] = weights[k] * origins_[k * n + floor];
      rate += mix[k];
    }
    int count = rng_.poisson(total * rate);
    if (count == 0) continue;
    for (int g = 0; g < n; ++g) {
      row[g] = 0.0;
      for (int k = 0; k < 3; ++k) row[g] += mix[k] * destinations_[(k * n + floor) * n + g];
    }
    // categorical() takes unnormalised weights.
    for (int i = 0; i < count; ++i) out.push_back({floor, rng_.categorical(row, n)});
  }
}

void Sim::add_arrivals(const std::vector<Arrival>& arrivals) {
  for (const Arrival& a : arrivals) {
    FloorQueue& queue = queues_[a.floor];
    if (queue.size == kMaxQueue) {
      ++dropped_;
      continue;
    }
    queue.destinations[queue.size++] = uint8_t(a.destination);
  }
}

StepResult Sim::apply_actions(const int* actions) {
  StepResult result;
  const Config& c = config_;
  for (int i = 0; i < c.n_lifts; ++i) {
    Lift& lift = lifts_[i];
    int action = actions[i];
    if (action == kUp || action == kDown) {
      move(lift, action == kUp ? 1 : -1);
    } else if (!c.kinematics || lift.status() == kIdle) {
      int direction = 0;
      if (c.hall_calls) direction = action == kServeUp ? 1 : -1;
      serve(lift, direction, result.delivered[i], result.boarded[i]);
      result.served[i] = true;
      if (c.kinematics) lift.door_timer = c.door_time;
    }
  }
  if (c.kinematics) {
    for (int s = 0; s < c.substeps; ++s) {
      for (int i = 0; i < c.n_lifts; ++i) integrate(lifts_[i]);
    }
  }
  return result;
}

void Sim::move(Lift& lift, int direction) {
  const int top = config_.n_floors - 1;
  auto clamp = [top](int floor) { return std::min(std::max(floor, 0), top); };
  if (!config_.kinematics) {
    lift.position = double(clamp(lift.floor() + direction));
    lift.target = lift.floor();
    return;
  }
  Status status = lift.status();
  if (status == kIdle) {
    lift.target = clamp(lift.floor() + direction);
  } else if (status == kMoving) {
    // Carry on one floor further, but only while the current target is the
    // next floor ahead, and never reverse mid-flight.
    double ahead = (double(lift.target) - lift.position) * direction;
    if (0 < ahead && ahead <= 1) lift.target = clamp(lift.target + direction);
  }
}

void Sim::serve(Lift& lift, int direction, int& delivered, int& boarded) {
  const int floor = lift.floor();
  int kept = 0;
  for (int i = 0; i < lift.n_passengers; ++i) {
    if (lift.passengers[i] != floor) lift.passengers[kept++] = lift.passengers[i];
  }
  delivered = lift.n_passengers - kept;
  lift.n_passengers = kept;

  FloorQueue& queue = queues_[floor];
  const int space = config_.lift_capacity - lift.n_passengers;
  int n_board = 0;
  int staying = 0;
  for (int i = 0; i < queue.size; ++i) {
    uint8_t destination = queue.destinations[i];
    bool wants = direction == 0 || (int(destination) - floor) * direction > 0;
    if (n_board < space && wants) {
      lift.passengers[lift.n_passengers++] = destination;
      ++n_board;
    } else {
      queue.destinations[staying++] = destination;
    }
  }
  queue.size = staying;
  boarded = n_board;
}

void Sim::integrate(Lift& lift) {
  // The same operations, in the same order, as BuildingSim._integrate, so the
  // results are bit-identical (built with -ffp-contract=off: no fused multiply-add).
  const double dt = dt_;
  if (lift.door_timer > 0) {
    lift.door_timer = std::max(lift.door_timer - dt, 0.0);
    return;
  }
  double distance = lift.target - lift.position;
  if (distance == 0 && lift.velocity == 0) return;
  const double sign = distance > 0 ? 1.0 : -1.0;
  distance = std::abs(distance);
  if (distance < 1e-9) {
    lift.position = double(lift.target);
    lift.velocity = 0.0;
    return;
  }
  const double speed = lift.velocity * sign;
  const double braking_distance = speed * speed / (2 * acceleration_);
  double new_speed;
  if (distance <= braking_distance) {
    // Constant deceleration that stops exactly on the target.
    const double deceleration = speed * speed / (2 * distance);
    new_speed = std::max(speed - deceleration * dt, 0.0);
  } else {
    new_speed = std::min(speed + acceleration_ * dt, max_speed_);
  }
  const double travelled = (speed + new_speed) / 2 * dt;
  if (travelled >= distance || new_speed == 0) {
    lift.position = double(lift.target);
    lift.velocity = 0.0;
  } else {
    lift.position += sign * travelled;
    lift.velocity = sign * new_speed;
  }
}

// --- Env ---------------------------------------------------------------------
//
// Mirrors BuildingEnv in building_env.py, including the order of floating
// point operations in the reward and observations, so results are identical.

Env::Env(const Config& config, const EnvOptions& options) : sim_(config), options_(options) {
  arrivals_.reserve(kMaxFloors * 8);
}

void Env::reset(uint64_t seed, const std::vector<std::vector<Arrival>>* warmup) {
  sim_.seed(seed);
  sim_.reset();
  steps_ = 0;
  goals_.fill(Goal{});
  directions_.fill(0);
  for (int round = 0; round < 10; ++round) {
    if (warmup) {
      sim_.add_arrivals(warmup->at(round));
    } else {
      sim_.draw_arrivals(0.0, arrivals_);
      sim_.add_arrivals(arrivals_);
    }
  }
  last_potential_ = potential();
}

int Env::n_actions_per_lift() const {
  const Config& c = sim_.config();
  if (options_.action_mode == ActionMode::Target) return c.n_floors * (c.hall_calls ? 2 : 1);
  return c.n_actions();
}

void Env::primitive_actions(const int* actions, int* out) {
  const Config& c = sim_.config();
  const int n = c.n_floors;
  for (int i = 0; i < c.n_lifts; ++i) {
    const Lift& lift = sim_.lift(i);
    Goal& goal = goals_[i];
    if (!goal.set) {
      goal.set = true;
      goal.floor = actions[i] % n;
      goal.direction = c.hall_calls && actions[i] >= n ? -1 : 1;
    }
    const int serve = c.hall_calls ? (goal.direction > 0 ? kServeUp : kServeDown) : kServe;
    const Status status = lift.status();
    if (status == kDoors) {
      out[i] = serve;  // ignored until the doors close
    } else if (status == kMoving) {
      // Ask to carry on while the goal lies beyond the current target.
      const int heading = lift.target > lift.position ? kUp : kDown;
      const bool beyond = (goal.floor - lift.target) * (heading == kUp ? 1 : -1) > 0;
      out[i] = beyond ? heading : serve;
    } else if (lift.floor() != goal.floor) {
      out[i] = goal.floor > lift.floor() ? kUp : kDown;
    } else {
      out[i] = serve;
    }
  }
}

double Env::potential() const {
  if (!options_.shaped || options_.shaping.progress == 0) return 0.0;
  const Config& c = sim_.config();
  double remaining = 0.0;
  for (int i = 0; i < c.n_lifts; ++i) {
    const Lift& lift = sim_.lift(i);
    for (int p = 0; p < lift.n_passengers; ++p) {
      remaining += std::abs(lift.passengers[p] - lift.position);
    }
  }
  return -options_.shaping.progress * remaining / (c.n_floors - 1);
}

EnvStep Env::step(const int* actions, const std::vector<Arrival>* arrivals) {
  const Config& c = sim_.config();
  if (arrivals) {
    sim_.add_arrivals(*arrivals);
  } else {
    sim_.draw_arrivals(double(steps_) / options_.max_steps, arrivals_);
    sim_.add_arrivals(arrivals_);
  }
  std::array<double, kMaxLifts> before{};
  for (int i = 0; i < c.n_lifts; ++i) before[i] = sim_.lift(i).position;

  StepResult r;
  if (options_.action_mode == ActionMode::Target) {
    int primitive[kMaxLifts];
    primitive_actions(actions, primitive);
    r = sim_.apply_actions(primitive);
    for (int i = 0; i < c.n_lifts; ++i) {
      if (r.served[i] && primitive[i] != kUp && primitive[i] != kDown) goals_[i].set = false;
    }
  } else {
    r = sim_.apply_actions(actions);
  }
  for (int i = 0; i < c.n_lifts; ++i) {
    double position = sim_.lift(i).position;
    if (position != before[i]) directions_[i] = position > before[i] ? 1 : -1;
  }

  EnvStep out;
  int empty_serves = 0;
  for (int i = 0; i < c.n_lifts; ++i) {
    out.delivered += r.delivered[i];
    out.boarded += r.boarded[i];
    empty_serves += r.served[i] && r.delivered[i] == 0 && r.boarded[i] == 0;
  }
  out.reward = double(out.delivered);
  if (options_.shaped) {
    const Shaping& s = options_.shaping;
    if (s.pickup != 0) out.reward += s.pickup * out.boarded;
    if (s.empty_serve != 0) out.reward -= s.empty_serve * empty_serves;
    if (s.progress != 0) {
      double now = potential();
      out.reward += s.gamma * now - last_potential_;
      last_potential_ = now;
    }
    if (s.waiting != 0) {
      long waiting = 0;
      for (int f = 0; f < c.n_floors; ++f) waiting += sim_.queue(f).size;
      out.reward -= s.waiting * waiting / c.n_floors;
    }
  }
  ++steps_;
  out.truncated = steps_ >= options_.max_steps;
  return out;
}

int Env::extra_fields() const {
  const int lifts = sim_.config().n_lifts;
  return (options_.action_mode == ActionMode::Target ? lifts : 0) +
         (options_.observe_direction ? lifts : 0);
}

int Env::observation_size(ObsType type) const {
  const Config& c = sim_.config();
  if (type == ObsType::Relative) {
    return c.n_lifts * (kRelativeLiftFields + (2 * c.n_floors - 1) * kRelativeFloorFields);
  }
  int lift_fields = c.kinematics ? 4 : (type == ObsType::Custom ? 2 : 1);
  int waiting = (c.hall_calls ? 2 : 1) * c.n_floors;
  return c.n_lifts * lift_fields + waiting + c.n_lifts * c.n_floors + extra_fields();
}

namespace {

// Waiting counts per floor (up then down with hall calls), uncapped.
int waiting_counts(const Sim& sim, int* out) {
  const Config& c = sim.config();
  for (int f = 0; f < c.n_floors; ++f) {
    const FloorQueue& queue = sim.queue(f);
    if (!c.hall_calls) {
      out[f] = queue.size;
      continue;
    }
    int up = 0;
    for (int i = 0; i < queue.size; ++i) up += queue.destinations[i] > f;
    out[f] = up;
    out[c.n_floors + f] = queue.size - up;
  }
  return (c.hall_calls ? 2 : 1) * c.n_floors;
}

}  // namespace

void Env::observe_custom(int64_t* out) const {
  const Config& c = sim_.config();
  int k = 0;
  for (int i = 0; i < c.n_lifts; ++i) {
    const Lift& lift = sim_.lift(i);
    out[k++] = lift.floor();
    out[k++] = lift.n_passengers;
    if (c.kinematics) {
      out[k++] = lift.target;
      out[k++] = lift.status();
    }
  }
  int waiting[2 * kMaxFloors];
  int n_waiting = waiting_counts(sim_, waiting);
  for (int j = 0; j < n_waiting; ++j) out[k++] = std::min(waiting[j], c.lift_capacity);
  for (int i = 0; i < c.n_lifts; ++i) {
    int64_t* wanted = out + k;
    std::fill(wanted, wanted + c.n_floors, 0);
    const Lift& lift = sim_.lift(i);
    for (int p = 0; p < lift.n_passengers; ++p) ++wanted[lift.passengers[p]];
    k += c.n_floors;
  }
  if (options_.action_mode == ActionMode::Target) {
    for (int i = 0; i < c.n_lifts; ++i) out[k++] = !goals_[i].set;
  }
  if (options_.observe_direction) {
    for (int i = 0; i < c.n_lifts; ++i) out[k++] = directions_[i] == -1 ? 2 : directions_[i];
  }
}

void Env::observe_box(float* out) const {
  const Config& c = sim_.config();
  const int n = c.n_floors;
  const double speed = c.max_speed / c.floor_height;
  int k = 0;
  for (int i = 0; i < c.n_lifts; ++i) {
    const Lift& lift = sim_.lift(i);
    out[k++] = float(lift.position / (n - 1));
    if (c.kinematics) {
      out[k++] = float(double(lift.target) / (n - 1));
      out[k++] = float(lift.velocity / speed);
      out[k++] = float(lift.door_timer / c.door_time);
    }
  }
  int waiting[2 * kMaxFloors];
  int n_waiting = waiting_counts(sim_, waiting);
  for (int j = 0; j < n_waiting; ++j) out[k++] = waiting[j] > 0 ? 1.0f : 0.0f;
  for (int i = 0; i < c.n_lifts; ++i) {
    float* wanted = out + k;
    std::fill(wanted, wanted + n, 0.0f);
    const Lift& lift = sim_.lift(i);
    for (int p = 0; p < lift.n_passengers; ++p) wanted[lift.passengers[p]] = 1.0f;
    k += n;
  }
  if (options_.action_mode == ActionMode::Target) {
    for (int i = 0; i < c.n_lifts; ++i) out[k++] = goals_[i].set ? 0.0f : 1.0f;
  }
  if (options_.observe_direction) {
    for (int i = 0; i < c.n_lifts; ++i) out[k++] = float(directions_[i]);
  }
}

void Env::observe_relative(float* out) const {
  // Float32 arithmetic where Python uses float32 NumPy arrays, double where
  // it uses Python floats, so the values match exactly.
  const Config& c = sim_.config();
  const int n = c.n_floors;
  const float cap = float(c.lift_capacity);
  int waiting[2 * kMaxFloors];
  waiting_counts(sim_, waiting);
  float up[kMaxFloors], down[kMaxFloors], lifts_at[kMaxFloors] = {};
  for (int f = 0; f < n; ++f) {
    up[f] = std::min(float(waiting[f]) / cap, 1.0f);
    down[f] = c.hall_calls ? std::min(float(waiting[n + f]) / cap, 1.0f) : up[f];
  }
  for (int i = 0; i < c.n_lifts; ++i) lifts_at[sim_.lift(i).floor()] += 1.0f;
  const float other_lifts = float(std::max(c.n_lifts - 1, 1));
  const double speed = c.max_speed / c.floor_height;
  const int window = 2 * n - 1;

  float* block = out;
  for (int i = 0; i < c.n_lifts; ++i) {
    const Lift& lift = sim_.lift(i);
    const Status status = lift.status();
    const bool free = options_.action_mode != ActionMode::Target || !goals_[i].set;
    float* head = block;
    head[0] = float(lift.position / (n - 1));
    head[1] = float(double(lift.n_passengers) / c.lift_capacity);
    head[2] = float(directions_[i]);
    head[3] = status == kIdle;
    head[4] = status == kMoving;
    head[5] = status == kDoors;
    head[6] = float((lift.target - lift.position) / (n - 1));
    head[7] = float(lift.velocity / speed);
    head[8] = c.kinematics ? float(lift.door_timer / c.door_time) : 0.0f;
    head[9] = free;

    int wanted[kMaxFloors] = {};
    for (int p = 0; p < lift.n_passengers; ++p) ++wanted[lift.passengers[p]];
    const int floor = lift.floor();
    float* cells = block + kRelativeLiftFields;
    for (int j = 0; j < window; ++j) {
      float* cell = cells + j * kRelativeFloorFields;
      const int g = floor + j - (n - 1);
      if (g < 0 || g >= n) {
        std::fill(cell, cell + kRelativeFloorFields, 0.0f);
        continue;
      }
      cell[0] = 1.0f;
      cell[1] = std::min(float(wanted[g]) / cap, 1.0f);
      cell[2] = up[g];
      cell[3] = down[g];
      cell[4] = (lifts_at[g] - (g == floor ? 1.0f : 0.0f)) / other_lifts;
    }
    block += kRelativeLiftFields + window * kRelativeFloorFields;
  }
}

}  // namespace elevator
