from gymnasium import Env, spaces
import numpy as np
from sim import LiftSim
from time import sleep
import random


class LiftEnv(Env):
    def __init__(self,
                 render_mode="none",
                 reward_shaping=False,
                 multi_binary=False,
                 multi_discrete=False):
        super().__init__()
        self.metadata = {"render_modes": ["human", "none"], "render_fps": 1}
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(3)
        self.multi_binary = False
        self.multi_discrete = False
        if multi_discrete:
            self.multi_discrete = True
            self.observation_space = spaces.MultiDiscrete(
                9*np.ones(30).squeeze())
        elif multi_binary:
            self.multi_binary = True
            self.observation_space = spaces.MultiBinary([3, 10])
        else:
            self.observation_space = spaces.Box(0, 1, shape=(21,))
        self.spec = None
        self.sim = LiftSim()
        self.steps = 0
        self.max_steps = 200
        self.reward_shaping = reward_shaping

    def reset(self,
              seed=None,
              options=None,
              ):
        super().reset(seed=seed, options=options)
        random.seed(seed)

        self.steps = 0
        self.sim.reset()

        state = self.sim.state()
        for n in range(10):
            self.sim.sample_passengers()

        obs = self._map_state_to_obs(state)
        info = {}

        return obs, info

    def step(self, action):
        previous_state = self.sim.state()
        n_passengers_served = 0
        n_passengers_who_got_on = 0
        self.sim.sample_passengers()
        if action == 0:
            self.sim.move_up()
        elif action == 1:
            self.sim.move_down()
        else:
            n_passengers_served, n_passengers_who_got_on = self.sim.serve_floor()

        state = self.sim.state()

        obs = self._map_state_to_obs(state)

        # Reward for serving passengers
        reward = n_passengers_served
        r1 = n_passengers_served

        if self.reward_shaping:
            # Reward for picking up passengers
            reward += n_passengers_who_got_on
            r2 = n_passengers_who_got_on

            # Reward for moving towards

            # Punish for moving up/down out of bounds
            # if (action == 0 or action == 1) and state.lift_position == previous_state.lift_position:
            #     reward -= 0.5

            # Punish for serving the floor when not needed
            r3 = 0
            if (action == 2 and
                    n_passengers_served == 0 and
                    n_passengers_who_got_on == 0):
                reward -= 0.5
                r3 -= 0.5

            # print(f"{r1}, {r2}, {r3}")

        terminated = False
        truncated = False
        if self.steps >= self.max_steps:
            terminated = True
            # truncated = True
        else:
            self.steps += 1
        info = {}
        return obs, reward, terminated, truncated, info

    def _map_state_to_obs(self, state):
        if self.multi_discrete:
            lift_position = [
                0 if n != state.lift_position else len(state.lift_passengers)
                for n in range(10)]

            lift_wanted = [min(len(state.floor_passengers[floor]), 8)
                           for floor in range(len(state.floor_passengers))]

            floor_wanted = []
            for floor in range(len(state.floor_passengers)):
                is_floor_wanted = 0
                for passenger in state.lift_passengers:
                    if passenger.destination == floor:
                        is_floor_wanted += 1
                floor_wanted.append(is_floor_wanted)

            np_obs = np.array(
                lift_position + lift_wanted + floor_wanted, np.integer)
        elif self.multi_binary:
            lift_position = [
                0 if n != state.lift_position else 1 for n in range(10)]

            lift_wanted = [len(state.floor_passengers[floor]) >
                           0 for floor in range(len(state.floor_passengers))]

            floor_wanted = []
            for floor in range(len(state.floor_passengers)):
                is_floor_wanted = False
                for passenger in state.lift_passengers:
                    if passenger.destination == floor:
                        is_floor_wanted = True
                floor_wanted.append(is_floor_wanted)

            np_obs = np.array(
                [lift_position, lift_wanted, floor_wanted], np.int8)
        else:
            obs = [state.lift_position / 9]
            for floor in range(len(state.floor_passengers)):
                obs.append(len(state.floor_passengers[floor]) > 0)
            for floor in range(len(state.floor_passengers)):
                is_floor_wanted = False
                for passenger in state.lift_passengers:
                    if passenger.destination == floor:
                        is_floor_wanted = True
                obs.append(is_floor_wanted)
            np_obs = np.array(obs, dtype=np.float32)

        return np_obs

    def render(self):
        if self.render_mode == "human":
            print(self.steps)
            self.sim.render()
            sleep(1/self.metadata["render_fps"])
        return None
