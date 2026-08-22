from gymnasium import Env, spaces
import numpy as np
from sim import LiftSim
from time import sleep


class LiftEnv(Env):
    def __init__(self):
        super().__init__()
        self.metadata = {"render_modes": ["human", "none"], "render_fps": 1}
        self.render_mode = "none"
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(0, 1, shape=(2,))
        self.spec = None
        self.sim = LiftSim()
        self.steps = 0
        self.max_steps = 200

    def reset(self,
              seed=None,
              options=None,
              ):
        super().reset(seed=seed, options=options)

        self.steps = 0
        self.sim.reset()

        state = self.sim.state()
        for n in range(10):
            self.sim.sample_passengers()

        obs = self._map_state_to_obs(state)
        info = {}

        return obs, info

    def step(self, action):
        n_passengers_served = 0
        self.sim.sample_passengers()
        if action == 0:
            self.sim.move_up()
        elif action == 1:
            self.sim.move_down()
        else:
            n_passengers_served = self.sim.serve_floor()

        state = self.sim.state()

        obs = self._map_state_to_obs(state)
        reward = n_passengers_served
        terminated = False
        if self.steps >= self.max_steps:
            terminated = True
        else:
            self.steps += 1
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def _map_state_to_obs(self, state):
        return np.array([0, 0], dtype=np.float32)

    def render(self):
        if self.render_mode == "human":
            self.sim.render()
            sleep(1/self.metadata["render_fps"])
        return None
