// Unit tests for the C++ env. Exact parity with Python is checked separately,
// by replaying traces (tests/test_cpp.py); these cover the behaviour the
// traces exercise rarely, and the random number generator.
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "elevator/building.hpp"

using namespace elevator;

namespace {

int failures = 0;

#define CHECK(condition)                                                           \
  do {                                                                             \
    if (!(condition)) {                                                            \
      std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #condition); \
      ++failures;                                                                  \
    }                                                                              \
  } while (0)

Config quiet(Config c = Config{}) {
  c.arrival_probability = 0.0;
  return c;
}

void test_presets_round_trip() {
  for (const char* name : {"original", "tall", "multi", "kinematic", "hall_calls", "full"}) {
    Config c = preset(name);
    CHECK(to_string(parse_config(to_string(c))) == to_string(c));
  }
  CHECK(to_string(preset("full")) ==
        "n_floors=30 n_lifts=4 lift_capacity=8 arrival_probability=0.1 traffic=day "
        "hall_calls=1 kinematics=1 floor_height=3.5 max_speed=2.5 acceleration=1.0 "
        "door_time=4.0 step_seconds=1.0 substeps=10");
  bool threw = false;
  try {
    parse_config("n_lifts=0");
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
}

void test_rng_moments() {
  Rng rng(42);
  const int n = 200000;
  double sum = 0;
  long poisson_sum = 0;
  int counts[5] = {};
  for (int i = 0; i < n; ++i) {
    double u = rng.uniform();
    CHECK(u >= 0 && u < 1);
    sum += u;
    poisson_sum += rng.poisson(2.4);
    ++counts[rng.integer(5)];
  }
  CHECK(std::abs(sum / n - 0.5) < 0.005);
  CHECK(std::abs(double(poisson_sum) / n - 2.4) < 0.02);
  for (int c : counts) CHECK(std::abs(c / double(n) - 0.2) < 0.005);

  Rng a(7), b(7);
  for (int i = 0; i < 100; ++i) CHECK(a.next() == b.next());
}

void test_arrival_rate_matches_every_profile() {
  for (Traffic t : {Traffic::Uniform, Traffic::UpPeak, Traffic::DownPeak, Traffic::Lunch,
                    Traffic::Day}) {
    Config c = preset("multi");
    c.traffic = t;
    Sim sim(c, 3);
    std::vector<Arrival> arrivals;
    long total = 0;
    const int steps = 20000;
    for (int s = 0; s < steps; ++s) {
      sim.draw_arrivals(double(s % 200) / 200, arrivals);
      total += long(arrivals.size());
      for (const Arrival& a : arrivals) {
        CHECK(a.floor != a.destination);
        CHECK(a.destination >= 0 && a.destination < c.n_floors);
      }
    }
    double expected = c.arrival_probability * c.n_floors;
    CHECK(std::abs(double(total) / steps - expected) < 0.05 * expected);
  }
}

void test_lower_numbered_lift_boards_first() {
  Config c = quiet();
  c.n_lifts = 2;
  c.lift_capacity = 2;
  Sim sim(c);
  sim.add_arrivals({{0, 3}, {0, 4}, {0, 5}});
  int actions[] = {kServe, kServe};
  StepResult r = sim.apply_actions(actions);
  CHECK(r.boarded[0] == 2 && r.boarded[1] == 1);
  CHECK(sim.lift(1).passengers[0] == 5);
}

void test_hall_calls_board_one_direction_in_order() {
  Config c = quiet();
  c.hall_calls = true;
  c.lift_capacity = 2;
  Sim sim(c);
  sim.lift(0).position = 5;
  sim.lift(0).target = 5;
  sim.add_arrivals({{5, 2}, {5, 8}, {5, 1}, {5, 9}, {5, 7}});
  int up[] = {kServeUp};
  StepResult r = sim.apply_actions(up);
  CHECK(r.boarded[0] == 2);
  CHECK(sim.lift(0).passengers[0] == 8 && sim.lift(0).passengers[1] == 9);
  const FloorQueue& q = sim.queue(5);
  CHECK(q.size == 3 && q.destinations[0] == 2 && q.destinations[1] == 1 &&
        q.destinations[2] == 7);
}

void test_one_floor_trip_takes_four_steps() {
  Config c = quiet();
  c.kinematics = true;
  Sim sim(c);
  int up[] = {kUp}, down[] = {kDown};
  sim.apply_actions(up);
  int steps = 1;
  while (sim.lift(0).status() != kIdle) {
    sim.apply_actions(down);  // ignored mid-flight
    ++steps;
  }
  CHECK(steps == 4);
  CHECK(sim.lift(0).position == 1.0 && sim.lift(0).velocity == 0.0);
}

void test_doors_block_for_door_time() {
  Config c = quiet();
  c.kinematics = true;
  Sim sim(c);
  int serve[] = {kServe}, up[] = {kUp};
  sim.apply_actions(serve);
  int steps = 1;
  while (sim.lift(0).status() != kIdle) {
    StepResult r = sim.apply_actions(up);
    CHECK(!r.served[0]);
    ++steps;
  }
  CHECK(steps == 4);
  CHECK(sim.lift(0).floor() == 0);
}

void test_queue_cap_drops_and_counts() {
  Sim sim(quiet());
  std::vector<Arrival> many(kMaxQueue + 5, Arrival{0, 1});
  sim.add_arrivals(many);
  CHECK(sim.queue(0).size == kMaxQueue);
  CHECK(sim.dropped() == 5);
}

void test_floor_rounds_half_to_even() {
  Lift lift;
  lift.position = 2.5;
  CHECK(lift.floor() == 2);  // Python: round(2.5) == 2
  lift.position = 3.5;
  CHECK(lift.floor() == 4);
}

void test_env_reset_is_reproducible() {
  Env a(preset("full"), true), b(preset("full"), true);
  a.reset(11);
  b.reset(11);
  std::vector<int64_t> oa(a.observation_size(ObsType::Custom)), ob(oa.size());
  int actions[] = {0, 2, 3, 1};
  for (int s = 0; s < 300; ++s) {
    EnvStep ra = a.step(actions), rb = b.step(actions);
    CHECK(ra.reward == rb.reward);
    actions[s % 4] = (actions[s % 4] + 1) % 4;
  }
  a.observe_custom(oa.data());
  b.observe_custom(ob.data());
  CHECK(oa == ob);
}

}  // namespace

int main() {
  test_presets_round_trip();
  test_rng_moments();
  test_arrival_rate_matches_every_profile();
  test_lower_numbered_lift_boards_first();
  test_hall_calls_board_one_direction_in_order();
  test_one_floor_trip_takes_four_steps();
  test_doors_block_for_door_time();
  test_queue_cap_drops_and_counts();
  test_floor_rounds_half_to_even();
  test_env_reset_is_reproducible();
  if (failures) {
    std::fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  std::printf("all C++ tests passed\n");
  return 0;
}
