from time import sleep

import numpy as np
from gymnasium import Env, spaces

from sim import LiftSim

OBS_TYPES = ("box", "multi_binary", "multi_discrete", "custom")


class LiftEnv(Env):
    """Single-lift dispatching environment.

    Actions: 0 = move up, 1 = move down, 2 = serve the current floor.

    Observation types (``obs_type``):
    - "box": lift position scaled to [0, 1], then one flag per floor for
      "someone is waiting here" and one per floor for "someone in the lift
      wants this floor".
    - "multi_binary": 3 x n_floors flags: lift position (one-hot), waiting,
      wanted.
    - "multi_discrete": 3 x n_floors counts: lift position (0 = lift not here,
      k = lift here with k - 1 passengers), waiting (capped), wanted.
    - "custom": lift position, passengers in lift, waiting per floor (capped),
      wanted per floor.
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, render_mode=None, reward_shaping=False, obs_type="box", max_steps=200):
        super().__init__()
        if obs_type not in OBS_TYPES:
            raise ValueError(f"obs_type must be one of {OBS_TYPES}, got {obs_type!r}")
        self.render_mode = render_mode
        self.obs_type = obs_type
        self.reward_shaping = reward_shaping
        self.max_steps = max_steps
        self.steps = 0

        self.sim = LiftSim()
        n_floors = self.sim.n_floors
        capacity = self.sim.lift_capacity
        self.action_space = spaces.Discrete(3)
        if obs_type == "custom":
            nvec = np.full(2 + 2 * n_floors, capacity + 1)
            nvec[0] = n_floors
            self.observation_space = spaces.MultiDiscrete(nvec)
        elif obs_type == "multi_discrete":
            nvec = np.full(3 * n_floors, capacity + 1)
            nvec[:n_floors] = capacity + 2
            self.observation_space = spaces.MultiDiscrete(nvec)
        elif obs_type == "multi_binary":
            self.observation_space = spaces.MultiBinary([3, n_floors])
        else:
            self.observation_space = spaces.Box(0, 1, shape=(1 + 2 * n_floors,))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.sim.rng = self.np_random

        self.steps = 0
        self.sim.reset()
        for _ in range(10):
            self.sim.sample_passengers()

        return self._map_state_to_obs(self.sim.state()), {}

    def step(self, action):
        n_passengers_served = 0
        n_passengers_who_got_on = 0
        self.sim.sample_passengers()
        if action == 0:
            self.sim.move_up()
        elif action == 1:
            self.sim.move_down()
        else:
            n_passengers_served, n_passengers_who_got_on = self.sim.serve_floor()

        obs = self._map_state_to_obs(self.sim.state())

        # Reward for serving passengers
        reward = n_passengers_served

        if self.reward_shaping:
            # Reward for picking up passengers
            reward += n_passengers_who_got_on

            # Punish for serving the floor when not needed
            if action == 2 and n_passengers_served == 0 and n_passengers_who_got_on == 0:
                reward -= 0.5

        # The task never ends on its own; episodes are cut off by a time limit.
        self.steps += 1
        terminated = False
        truncated = self.steps >= self.max_steps
        return obs, reward, terminated, truncated, {}

    def _map_state_to_obs(self, state):
        capacity = self.sim.lift_capacity
        n_floors = len(state.floor_passengers)
        waiting = [len(queue) for queue in state.floor_passengers]
        wanted = [0] * n_floors
        for passenger in state.lift_passengers:
            wanted[passenger.destination] += 1

        if self.obs_type == "custom":
            obs = [state.lift_position, len(state.lift_passengers)]
            obs += [min(n, capacity) for n in waiting] + wanted
            return np.array(obs, dtype=np.int64)
        if self.obs_type == "multi_discrete":
            position = [0] * n_floors
            position[state.lift_position] = 1 + len(state.lift_passengers)
            obs = position + [min(n, capacity) for n in waiting] + wanted
            return np.array(obs, dtype=np.int64)
        if self.obs_type == "multi_binary":
            position = [int(floor == state.lift_position) for floor in range(n_floors)]
            obs = [position, [n > 0 for n in waiting], [n > 0 for n in wanted]]
            return np.array(obs, dtype=np.int8)
        obs = [state.lift_position / (n_floors - 1)]
        obs += [n > 0 for n in waiting] + [n > 0 for n in wanted]
        return np.array(obs, dtype=np.float32)

    def render(self):
        if self.render_mode == "human":
            print(self.steps)
            self.sim.render()
            sleep(1 / self.metadata["render_fps"])
        return None
