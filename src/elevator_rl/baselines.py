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


class CollectivePolicy:
    """Collective control for `BuildingEnv` with obs_type="custom".

    The classic elevator algorithm: each lift sweeps in one direction,
    stopping where someone on board wants to get off or someone is waiting to
    go its way, and turns round when there is nothing more ahead. With no work
    at all it heads for the lobby. Lifts do not coordinate, so several may
    chase the same call; it is a reference point, not a strong dispatcher.
    """

    def __init__(self, config):
        from elevator_rl.building import PRESETS, BuildingConfig

        if isinstance(config, str):
            config = PRESETS[config]
        self.config = config if config is not None else BuildingConfig()
        self.directions = [1] * self.config.n_lifts

    def reset(self):
        self.directions = [1] * self.config.n_lifts

    def predict(self, observations, state=None, episode_start=None, deterministic=False):
        if episode_start is not None and episode_start[0]:
            self.reset()
        return [self._act(obs) for obs in observations], state

    def _parse(self, obs):
        c = self.config
        n, fields = c.n_floors, 4 if c.kinematics else 2
        lifts = [obs[i * fields : (i + 1) * fields] for i in range(c.n_lifts)]
        k = c.n_lifts * fields
        if c.hall_calls:
            up, down = obs[k : k + n], obs[k + n : k + 2 * n]
            k += 2 * n
        else:
            up = down = obs[k : k + n]
            k += n
        wanted = [obs[k + i * n : k + (i + 1) * n] for i in range(c.n_lifts)]
        return lifts, up, down, wanted

    def _act(self, obs):
        lifts, up, down, wanted = self._parse(obs)
        return [
            self._act_lift(i, lift, up, down, want)
            for i, (lift, want) in enumerate(zip(lifts, wanted, strict=True))
        ]

    def _act_lift(self, i, lift, up, down, want):
        from elevator_rl.building import DOWN, IDLE, MOVING, SERVE, SERVE_DOWN, SERVE_UP, UP

        c = self.config
        n = c.n_floors
        floor, load = int(lift[0]), int(lift[1])
        room = load < c.lift_capacity

        def stop_at(f, direction):
            calls = up[f] if direction > 0 or not c.hall_calls else down[f]
            return want[f] > 0 or (room and calls > 0)

        def work_beyond(f, direction):
            floors = range(f + 1, n) if direction > 0 else range(f - 1, -1, -1)
            return any(want[g] > 0 or up[g] > 0 or down[g] > 0 for g in floors)

        def serve(direction):
            if not c.hall_calls:
                return SERVE
            return SERVE_UP if direction > 0 else SERVE_DOWN

        def go(direction):
            return UP if direction > 0 else DOWN

        d = self.directions[i]
        if c.kinematics and int(lift[3]) != IDLE:
            # Carry on past the target only if there is no reason to stop there.
            target = int(lift[2])
            if int(lift[3]) == MOVING and not stop_at(target, d) and work_beyond(target, d):
                return go(d)
            return serve(d)  # ignored while busy
        if stop_at(floor, d):
            return serve(d)
        if work_beyond(floor, d):
            return go(d)
        if stop_at(floor, -d) or work_beyond(floor, -d):
            self.directions[i] = d = -d
            return serve(d) if stop_at(floor, d) else go(d)
        self.directions[i] = -1
        return DOWN  # park at the lobby
