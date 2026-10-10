from time import sleep

import numpy as np
from gymnasium import Env, spaces

from elevator_rl.building import DOORS, PRESETS, BuildingConfig, BuildingSim

OBS_TYPES = ("custom", "box")


class BuildingEnv(Env):
    """Multi-lift dispatching with one central controller.

    The action is one choice per lift (`MultiDiscrete`): 0 = up, 1 = down,
    2 = serve; with hall calls, 2 = serve up and 3 = serve down. The reward is
    summed over lifts. `config` (a `BuildingConfig` or a name from `PRESETS`)
    selects the features; the default matches `LiftEnv`, and its "custom" and
    "box" observations are laid out the same way.

    Observation types (``obs_type``), lift fields first, then floors:
    - "custom" (`MultiDiscrete`): per lift its floor and passenger count, plus
      its target floor and status (idle, moving, doors) with kinematics; then
      waiting counts per floor (per floor and direction with hall calls),
      capped at the lift capacity; then, per lift, passengers wanting each floor.
    - "box" (`Box`): per lift its position scaled to [0, 1], plus its target
      floor, velocity (scaled to [-1, 1]) and door timer with kinematics; then
      flags for "someone is waiting" and, per lift, "someone wants this floor".
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        config=None,
        render_mode=None,
        reward_shaping=False,
        obs_type="custom",
        max_steps=200,
    ):
        super().__init__()
        if isinstance(config, str):
            config = PRESETS[config]
        self.config = config if config is not None else BuildingConfig()
        if obs_type not in OBS_TYPES:
            raise ValueError(f"obs_type must be one of {OBS_TYPES}, got {obs_type!r}")
        self.render_mode = render_mode
        self.obs_type = obs_type
        self.reward_shaping = reward_shaping
        self.max_steps = max_steps
        self.steps = 0
        self.sim = BuildingSim(self.config)

        c = self.config
        self.action_space = spaces.MultiDiscrete([c.n_actions] * c.n_lifts)
        # Box drops the passenger count, so both types have 4 fields with kinematics.
        lift_fields = 4 if c.kinematics else (2 if obs_type == "custom" else 1)
        n_waiting = (2 if c.hall_calls else 1) * c.n_floors
        if obs_type == "custom":
            per_lift = [c.n_floors, c.lift_capacity + 1]
            if c.kinematics:
                per_lift += [c.n_floors, 3]
            nvec = np.full(
                c.n_lifts * lift_fields + n_waiting + c.n_lifts * c.n_floors, c.lift_capacity + 1
            )
            nvec[: c.n_lifts * lift_fields] = np.tile(per_lift, c.n_lifts)
            self.observation_space = spaces.MultiDiscrete(nvec)
        else:
            size = c.n_lifts * lift_fields
            low = np.zeros(size + n_waiting + c.n_lifts * c.n_floors, dtype=np.float32)
            if c.kinematics:
                # The velocity is the third field of each lift.
                low[2:size:lift_fields] = -1
            self.observation_space = spaces.Box(low, 1, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.sim.rng = self.np_random

        self.steps = 0
        self.sim.reset()
        for _ in range(10):
            self.sim.sample_passengers(0.0)

        return self._observation(), {}

    def step(self, action):
        self.sim.sample_passengers(self.steps / self.max_steps)
        delivered, boarded, served = self.sim.apply_actions(action)

        reward = float(sum(delivered))
        if self.reward_shaping:
            reward += sum(boarded)
            reward -= 0.5 * sum(
                s and d == 0 and b == 0 for s, d, b in zip(served, delivered, boarded, strict=True)
            )

        self.steps += 1
        truncated = self.steps >= self.max_steps
        info = {
            "delivered": sum(delivered),
            "boarded": sum(boarded),
            "waiting": sum(len(queue) for queue in self.sim.state().floor_queues),
        }
        return self._observation(), reward, False, truncated, info

    def _observation(self):
        c = self.config
        state = self.sim.state()
        n = c.n_floors

        if c.hall_calls:
            up = [sum(d > f for d in queue) for f, queue in enumerate(state.floor_queues)]
            down = [len(queue) - u for queue, u in zip(state.floor_queues, up, strict=True)]
            waiting = up + down
        else:
            waiting = [len(queue) for queue in state.floor_queues]
        wanted = []
        for lift in state.lifts:
            counts = [0] * n
            for destination in lift.passengers:
                counts[destination] += 1
            wanted += counts

        if self.obs_type == "custom":
            obs = []
            for lift in state.lifts:
                obs += [lift.floor, len(lift.passengers)]
                if c.kinematics:
                    obs += [lift.target, lift.status]
            obs += [min(w, c.lift_capacity) for w in waiting] + wanted
            return np.array(obs, dtype=np.int64)

        speed = c.max_speed / c.floor_height
        obs = []
        for lift in state.lifts:
            obs.append(lift.position / (n - 1))
            if c.kinematics:
                obs += [lift.target / (n - 1), lift.velocity / speed, lift.door_timer / c.door_time]
        obs += [w > 0 for w in waiting] + [w > 0 for w in wanted]
        return np.array(obs, dtype=np.float32)

    def render(self):
        if self.render_mode == "human":
            print(self.steps)
            for i, lift in enumerate(self.sim.state().lifts):
                if lift.status == DOORS:
                    print(f"lift {i}: doors open at floor {lift.floor}")
            self.sim.render()
            sleep(1 / self.metadata["render_fps"])
        return None
