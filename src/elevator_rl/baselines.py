"""Hand-written policies with the Stable-Baselines3 `predict` interface."""

from gymnasium.spaces import Discrete


class RandomPolicy:
    def predict(self, observations, state=None, episode_start=None, deterministic=False):
        action_space = Discrete(3)
        action = [action_space.sample() for n in range(len(observations))]
        return action, state


class UpDownPolicy:
    """Sweep the building, serving every floor. Expects obs_type="box"."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.going_up = True
        self.served = False

    def predict(self, observations, state=None, episode_start=None, deterministic=False):
        if episode_start is not None and episode_start[0]:
            self.reset()

        if observations[0][0] == 1 and self.going_up:
            self.going_up = False

        if observations[0][0] == 0 and not self.going_up:
            self.going_up = True

        if not self.served:
            action = 2
            self.served = True
        else:
            self.served = False
            if self.going_up:
                action = 0
            else:
                action = 1
        return [action], state
