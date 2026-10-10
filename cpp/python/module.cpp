// Python bindings: elevator_rl._cpp. The Python side is elevator_rl/cpp_env.py.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "elevator/building.hpp"
#include "elevator/vec_env.hpp"

namespace nb = nanobind;
using namespace elevator;

namespace {

using Array = nb::ndarray<nb::c_contig, nb::device::cpu>;
using Pairs = std::vector<std::pair<int, int>>;

std::vector<Arrival> to_arrivals(const Pairs& pairs) {
  std::vector<Arrival> out;
  out.reserve(pairs.size());
  for (auto [floor, destination] : pairs) out.push_back({floor, destination});
  return out;
}

// Checks shape and element type, then returns the data pointer.
template <typename T>
T* data(Array& array, size_t rows, size_t cols, const char* name) {
  bool shape_ok = cols == 0 ? array.ndim() == 1 && array.shape(0) == rows
                            : array.ndim() == 2 && array.shape(0) == rows && array.shape(1) == cols;
  if (!shape_ok) throw std::invalid_argument(std::string(name) + " has the wrong shape");
  if (array.dtype() != nb::dtype<T>()) {
    throw std::invalid_argument(std::string(name) + " has the wrong dtype");
  }
  return static_cast<T*>(array.data());
}

void* observation_data(VecEnv& v, Array& array, const char* name) {
  size_t rows = size_t(v.num_envs()), cols = size_t(v.observation_size());
  if (v.integer_observations()) return data<int64_t>(array, rows, cols, name);
  return data<float>(array, rows, cols, name);
}

}  // namespace

NB_MODULE(_cpp, m) {
  m.doc() = "C++ port of elevator_rl.building_env.BuildingEnv";

  nb::class_<Config>(m, "Config")
      .def_ro("n_floors", &Config::n_floors)
      .def_ro("n_lifts", &Config::n_lifts)
      .def("__str__", [](const Config& c) { return to_string(c); });
  m.def("parse_config", &parse_config, "Config from 'key=value ...' pairs (trace.config_line)");
  m.def("preset", &preset);

  nb::enum_<ObsType>(m, "ObsType")
      .value("custom", ObsType::Custom)
      .value("box", ObsType::Box)
      .value("relative", ObsType::Relative);
  nb::enum_<ActionMode>(m, "ActionMode")
      .value("step", ActionMode::Step)
      .value("target", ActionMode::Target);

  nb::class_<Shaping>(m, "Shaping")
      .def(nb::init<>())
      .def_rw("pickup", &Shaping::pickup)
      .def_rw("empty_serve", &Shaping::empty_serve)
      .def_rw("progress", &Shaping::progress)
      .def_rw("waiting", &Shaping::waiting)
      .def_rw("gamma", &Shaping::gamma);
  nb::class_<EnvOptions>(m, "EnvOptions")
      .def(nb::init<>())
      .def_rw("shaped", &EnvOptions::shaped)
      .def_rw("shaping", &EnvOptions::shaping)
      .def_rw("max_steps", &EnvOptions::max_steps)
      .def_rw("action_mode", &EnvOptions::action_mode)
      .def_rw("observe_direction", &EnvOptions::observe_direction);

  // One env, with injectable arrivals: used to check the binding against Python.
  nb::class_<Env>(m, "Env")
      .def(nb::init<const Config&, const EnvOptions&>())
      .def(
          "reset",
          [](Env& env, uint64_t seed, std::optional<std::vector<Pairs>> warmup) {
            if (!warmup) return env.reset(seed);
            std::vector<std::vector<Arrival>> rounds;
            for (const Pairs& round : *warmup) rounds.push_back(to_arrivals(round));
            env.reset(seed, &rounds);
          },
          nb::arg("seed"), nb::arg("warmup") = nb::none())
      .def(
          "step",
          [](Env& env, std::vector<int> actions, std::optional<Pairs> arrivals) {
            if (int(actions.size()) != env.sim().config().n_lifts) {
              throw std::invalid_argument("need one action per lift");
            }
            std::vector<Arrival> injected;
            if (arrivals) injected = to_arrivals(*arrivals);
            EnvStep out = env.step(actions.data(), arrivals ? &injected : nullptr);
            return std::make_tuple(out.reward, out.truncated, out.delivered, out.boarded);
          },
          nb::arg("actions"), nb::arg("arrivals") = nb::none())
      .def("observe", [](const Env& env, ObsType type) {
        size_t size = size_t(env.observation_size(type));
        if (type == ObsType::Custom) {
          auto* values = new int64_t[size];
          env.observe_custom(values);
          nb::capsule owner(values, [](void* p) noexcept { delete[] static_cast<int64_t*>(p); });
          return nb::cast(nb::ndarray<nb::numpy, int64_t, nb::ndim<1>>(values, {size}, owner));
        }
        auto* values = new float[size];
        if (type == ObsType::Box) env.observe_box(values);
        else env.observe_relative(values);
        nb::capsule owner(values, [](void* p) noexcept { delete[] static_cast<float*>(p); });
        return nb::cast(nb::ndarray<nb::numpy, float, nb::ndim<1>>(values, {size}, owner));
      });

  nb::class_<VecEnv>(m, "VecEnv")
      .def(nb::init<const Config&, const EnvOptions&, int, ObsType, uint64_t>(), nb::arg("config"),
           nb::arg("options"), nb::arg("n_envs"), nb::arg("obs_type"), nb::arg("seed") = 0)
      .def_prop_ro("num_envs", &VecEnv::num_envs)
      .def_prop_ro("n_lifts", &VecEnv::n_lifts)
      .def_prop_ro("observation_size", &VecEnv::observation_size)
      .def(
          "reset",
          [](VecEnv& v, Array seeds, Array obs) {
            int64_t* s = data<int64_t>(seeds, size_t(v.num_envs()), 0, "seeds");
            void* o = observation_data(v, obs, "obs");
            nb::gil_scoped_release release;
            v.reset(s, o);
          },
          nb::arg("seeds"), nb::arg("obs"))
      .def(
          "step",
          [](VecEnv& v, Array actions, Array obs, Array rewards, Array dones, Array terminal_obs,
             Array episode_returns, Array episode_lengths) {
            size_t n = size_t(v.num_envs());
            const int64_t* a = data<int64_t>(actions, n, size_t(v.n_lifts()), "actions");
            void* o = observation_data(v, obs, "obs");
            void* t = observation_data(v, terminal_obs, "terminal_obs");
            float* r = data<float>(rewards, n, 0, "rewards");
            uint8_t* d = data<uint8_t>(dones, n, 0, "dones");
            float* er = data<float>(episode_returns, n, 0, "episode_returns");
            int32_t* el = data<int32_t>(episode_lengths, n, 0, "episode_lengths");
            // The whole batch runs without the GIL.
            nb::gil_scoped_release release;
            v.step(a, o, r, d, t, er, el);
          },
          "Step every env once; arrays are written in place (see vec_env.hpp).");
}
