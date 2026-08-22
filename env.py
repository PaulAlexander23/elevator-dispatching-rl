from gymnasium import Env, spaces
import numpy as np


class LiftEnv(Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(0, 1, shape=(2,))
        self.spec = None

    def reset(self,
              seed=None,
              options=None,
              ):
        super().reset(seed=seed, options=options)
        obs = np.array([0, 0], dtype=np.float32)
        info = {}

        return obs, info

    def step(self, action):
        obs = np.array([0, 0], dtype=np.float32)
        reward = 0
        terminated = True
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info
