"""NVIDIA Warp port of `BuildingEnv`: one GPU thread per env.

Where the JAX port (`jax_env`) recasts the simulator as array code, with
every branch computed and masked and every queue scanned to its full 512
slots, Warp kernels are CUDA-style: each thread runs one env with ordinary
loops and `if`s, close to the C++ port. Boarding walks only the passengers
actually waiting, and the auto-reset runs only in the threads whose episode
ended.

The kernels are generated per configuration (closures over the config, so
sizes and feature flags are compile-time constants, as JAX tracing makes
them) and run inside `jax.jit` through `warp.jax_kernel`. `make_vec_env`
gives the same (reset, step) pair as `jax_env.make_vec_env`, so `jax_ppo`
trains on either.

State lives in JAX arrays shaped like `jax_env.EnvState`, batched over envs.
Floats are float64 when `jax_enable_x64` is on (the parity tests) and float32
otherwise. Kernels are compiled with `fuse_fp` off: like XLA and the C++
compiler, NVRTC would otherwise fuse `a + b * c` into a multiply-add that
rounds differently from Python.

Differences from `jax_env`: random arrivals come from Warp's per-thread
generator (statistically the same, not the same draws), and a step's Poisson
count is not capped. Ported: step actions, "box" and "relative" observations.
Needs the `warp` dependency group and a CUDA GPU (or Warp's CPU backend).
"""

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp

from elevator_rl.building import _DAY_KEYFRAMES, _PROFILE_WEIGHTS, PRESETS, BuildingConfig
from elevator_rl.building import _traffic_patterns as traffic_patterns
from elevator_rl.building_env import RELATIVE_FLOOR_FIELDS, RELATIVE_LIFT_FIELDS, SHAPINGS
from elevator_rl.jax_env import MAX_QUEUE, OBS_TYPES, WARMUP_ROUNDS, Arrivals

UP, DOWN, SERVE_UP = 0, 1, 2
IDLE, MOVING, DOORS = 0, 1, 2

# State fields, in kernel argument order, with their per-env shape and dtype
# ("float" follows jax_enable_x64).
STATE_FIELDS = (
    "position",
    "velocity",
    "target",
    "door_timer",
    "passengers",
    "queue",
    "queue_len",
    "queue_up",
    "direction",
    "steps",
    "last_potential",
    "dropped",
    "rng",
    "episode_return",
    "episode_length",
)


def _keyframes(traffic):
    """(times, weights) of the traffic mix over an episode, for the kernels."""
    if traffic == "day":
        times = [t for t, _ in _DAY_KEYFRAMES]
        weights = [w for _, w in _DAY_KEYFRAMES]
    else:
        times, weights = [0.0, 1.0], [_PROFILE_WEIGHTS[traffic]] * 2
    return np.array(times), np.array(weights)


def build_kernels(env, ft):
    """(reset_kernel, step_kernel) for this env's settings and float type `ft`."""
    c = env.config
    F, L, CAP = c.n_floors, c.n_lifts, c.lift_capacity
    Q, W = MAX_QUEUE, WARMUP_ROUNDS
    KIN, HALL, UNIFORM = c.kinematics, c.hall_calls, c.traffic == "uniform"
    RELATIVE, DIRECTION = env.obs_type == "relative", env.observe_direction
    MAX_STEPS = env.max_steps
    shaping = env.reward_shaping
    SHAPED = shaping is not None
    PICKUP = ft(shaping.pickup if SHAPED else 0.0)
    EMPTY = ft(shaping.empty_serve if SHAPED else 0.0)
    PROGRESS = ft(shaping.progress if SHAPED else 0.0)
    WAITING = ft(shaping.waiting if SHAPED else 0.0)
    GAMMA = ft(shaping.gamma if SHAPED else 0.0)
    HAS_PICKUP = SHAPED and shaping.pickup != 0
    HAS_EMPTY = SHAPED and shaping.empty_serve != 0
    HAS_PROGRESS = SHAPED and shaping.progress != 0
    HAS_WAITING = SHAPED and shaping.waiting != 0
    P = float(c.arrival_probability)
    TOTAL = ft(c.arrival_probability * c.n_floors)
    TOP = F - 1
    ZERO, ONE, HALF, TWO = ft(0.0), ft(1.0), ft(0.5), ft(2.0)
    DT = ft(c.step_seconds / c.substeps)
    ACCEL = ft(c.acceleration / c.floor_height)
    VMAX = ft(c.max_speed / c.floor_height)
    ACCEL_DT = ft((c.acceleration / c.floor_height) * (c.step_seconds / c.substeps))
    DOOR_TIME = ft(c.door_time)
    SUBSTEPS = c.substeps
    N_KEY = len(_keyframes(c.traffic)[0])
    OTHERS = float(max(L - 1, 1))
    LIFT_BLOCK = RELATIVE_LIFT_FIELDS + (2 * F - 1) * RELATIVE_FLOOR_FIELDS
    floats = wp.types.vector(length=L, dtype=ft)
    opts = {"fuse_fp": False}

    @wp.func
    def round_even(x: ft) -> int:
        # Python's round(): halves go to the even neighbour.
        r = wp.floor(x)
        d = x - r
        n = int(r)
        if d > HALF:
            n = n + 1
        elif d == HALF and n % 2 != 0:
            n = n + 1
        return n

    @wp.func
    def status(
        e: int,
        i: int,
        position: wp.array2d(dtype=ft),
        velocity: wp.array2d(dtype=ft),
        target: wp.array2d(dtype=int),
        door_timer: wp.array2d(dtype=ft),
    ) -> int:
        if door_timer[e, i] > ZERO:
            return DOORS
        if position[e, i] != ft(target[e, i]) or velocity[e, i] != ZERO:
            return MOVING
        return IDLE

    @wp.func
    def add_arrival(
        e: int,
        f: int,
        d: int,
        queue: wp.array3d(dtype=wp.int8),
        queue_len: wp.array2d(dtype=int),
        queue_up: wp.array2d(dtype=int),
        dropped: wp.array(dtype=int),
    ):
        n = queue_len[e, f]
        if n < Q:
            queue[e, f, n] = wp.int8(d)
            queue_len[e, f] = n + 1
            if d > f:
                queue_up[e, f] = queue_up[e, f] + 1
        else:
            dropped[e] = dropped[e] + 1

    @wp.func
    def destination_weight(
        f: int, g: int, m0: ft, m1: ft, m2: ft, destinations: wp.array3d(dtype=ft)
    ) -> ft:
        """Unnormalised chance that an arrival at floor f goes to floor g."""
        return m0 * destinations[0, f, g] + m1 * destinations[1, f, g] + m2 * destinations[2, f, g]

    @wp.func
    def draw_arrivals(
        e: int,
        fraction: ft,
        state: wp.uint32,
        origins: wp.array2d(dtype=ft),
        destinations: wp.array3d(dtype=ft),
        key_times: wp.array(dtype=ft),
        key_weights: wp.array2d(dtype=ft),
        queue: wp.array3d(dtype=wp.int8),
        queue_len: wp.array2d(dtype=int),
        queue_up: wp.array2d(dtype=int),
        dropped: wp.array(dtype=int),
    ) -> wp.uint32:
        if wp.static(UNIFORM):
            for f in range(F):
                if wp.randf(state) < P:
                    d = wp.randi(state, 0, F - 1)
                    if d >= f:
                        d = d + 1
                    add_arrival(e, f, d, queue, queue_len, queue_up, dropped)
            return state
        # The traffic mix at this point of the episode, then Poisson counts.
        t = wp.clamp(fraction, ZERO, ONE)
        w0 = key_weights[N_KEY - 1, 0]
        w1 = key_weights[N_KEY - 1, 1]
        w2 = key_weights[N_KEY - 1, 2]
        for k in range(N_KEY - 1):
            t0 = key_times[k]
            t1 = key_times[k + 1]
            if t <= t1 and t >= t0:
                a = ONE
                if t1 > t0:
                    a = (t - t0) / (t1 - t0)
                w0 = (ONE - a) * key_weights[k, 0] + a * key_weights[k + 1, 0]
                w1 = (ONE - a) * key_weights[k, 1] + a * key_weights[k + 1, 1]
                w2 = (ONE - a) * key_weights[k, 2] + a * key_weights[k + 1, 2]
                break
        for f in range(F):
            m0 = w0 * origins[0, f]
            m1 = w1 * origins[1, f]
            m2 = w2 * origins[2, f]
            rate = TOTAL * (m0 + m1 + m2)
            count = int(wp.poisson(state, float(rate)))
            total = ZERO
            for g in range(F):
                total += destination_weight(f, g, m0, m1, m2, destinations)
            for _ in range(count):
                d = F - 1
                if total > ZERO:
                    u = ft(wp.randf(state)) * total
                    acc = ZERO
                    for g in range(F):
                        acc += destination_weight(f, g, m0, m1, m2, destinations)
                        if u < acc:
                            d = g
                            break
                else:
                    d = wp.randi(state, 0, F)
                add_arrival(e, f, d, queue, queue_len, queue_up, dropped)
        return state

    @wp.func
    def serve(
        e: int,
        i: int,
        direction: int,
        position: wp.array2d(dtype=ft),
        passengers: wp.array3d(dtype=int),
        queue: wp.array3d(dtype=wp.int8),
        queue_len: wp.array2d(dtype=int),
        queue_up: wp.array2d(dtype=int),
    ) -> wp.vec2i:
        """BuildingSim._serve: drop off, then board in arrival order until full."""
        floor = round_even(position[e, i])
        delivered = passengers[e, i, floor]
        passengers[e, i, floor] = 0
        load = int(0)
        for g in range(F):
            load += passengers[e, i, g]
        space = CAP - load
        n = queue_len[e, floor]
        boarded = int(0)
        kept = int(0)
        up_left = int(0)
        for j in range(n):
            d = int(queue[e, floor, j])
            board = boarded < space
            if wp.static(HALL):
                board = board and (d - floor) * direction > 0
            if board:
                passengers[e, i, d] = passengers[e, i, d] + 1
                boarded += 1
                if d > floor:
                    up_left += 1
            else:
                if kept != j:  # move up to fill the gap
                    queue[e, floor, kept] = wp.int8(d)
                kept += 1
        queue_len[e, floor] = kept
        queue_up[e, floor] = queue_up[e, floor] - up_left
        return wp.vec2i(delivered, boarded)

    @wp.func
    def move(
        e: int,
        i: int,
        direction: int,
        position: wp.array2d(dtype=ft),
        velocity: wp.array2d(dtype=ft),
        target: wp.array2d(dtype=int),
        door_timer: wp.array2d(dtype=ft),
    ):
        floor = round_even(position[e, i])
        if wp.static(not KIN):
            new = wp.clamp(floor + direction, 0, TOP)
            position[e, i] = ft(new)
            target[e, i] = new
            return
        s = status(e, i, position, velocity, target, door_timer)
        if s == IDLE:
            target[e, i] = wp.clamp(floor + direction, 0, TOP)
        elif s == MOVING:
            # Carry on one floor further while the target is the next floor ahead.
            ahead = (ft(target[e, i]) - position[e, i]) * ft(direction)
            if ahead > ZERO and ahead <= ONE:
                target[e, i] = wp.clamp(target[e, i] + direction, 0, TOP)

    @wp.func
    def integrate(
        e: int,
        i: int,
        position: wp.array2d(dtype=ft),
        velocity: wp.array2d(dtype=ft),
        target: wp.array2d(dtype=int),
        door_timer: wp.array2d(dtype=ft),
    ):
        """BuildingSim._integrate: doors, then a bang-bang profile."""
        if door_timer[e, i] > ZERO:
            door_timer[e, i] = wp.max(door_timer[e, i] - DT, ZERO)
            return
        goal = ft(target[e, i])
        distance = goal - position[e, i]
        if distance == ZERO and velocity[e, i] == ZERO:
            return
        sign = ONE
        if distance <= ZERO:
            sign = -ONE
        distance = wp.abs(distance)
        if distance < ft(1e-9):
            position[e, i] = goal
            velocity[e, i] = ZERO
            return
        speed = velocity[e, i] * sign
        braking = speed * speed / (TWO * ACCEL)
        new_speed = ZERO
        if distance <= braking:
            deceleration = speed * speed / (TWO * distance)
            new_speed = wp.max(speed - deceleration * DT, ZERO)
        else:
            new_speed = wp.min(speed + ACCEL_DT, VMAX)
        travelled = (speed + new_speed) / TWO * DT
        if travelled >= distance or new_speed == ZERO:
            position[e, i] = goal
            velocity[e, i] = ZERO
        else:
            position[e, i] = position[e, i] + sign * travelled
            velocity[e, i] = sign * new_speed

    @wp.func
    def observe(
        e: int,
        row: int,
        out: wp.array2d(dtype=wp.float32),
        position: wp.array2d(dtype=ft),
        velocity: wp.array2d(dtype=ft),
        target: wp.array2d(dtype=int),
        door_timer: wp.array2d(dtype=ft),
        passengers: wp.array3d(dtype=int),
        queue_len: wp.array2d(dtype=int),
        queue_up: wp.array2d(dtype=int),
        direction: wp.array2d(dtype=int),
    ):
        """BuildingEnv's "relative" or "box" observation of env e, into out[row]."""
        top = ft(TOP)
        if wp.static(not RELATIVE):
            col = int(0)
            for i in range(L):
                out[row, col] = wp.float32(position[e, i] / top)
                col += 1
                if wp.static(KIN):
                    out[row, col] = wp.float32(ft(target[e, i]) / top)
                    out[row, col + 1] = wp.float32(velocity[e, i] / VMAX)
                    out[row, col + 2] = wp.float32(door_timer[e, i] / DOOR_TIME)
                    col += 3
            for f in range(F):
                up = queue_len[e, f]
                if wp.static(HALL):
                    up = queue_up[e, f]
                out[row, col + f] = wp.float32(wp.where(up > 0, 1.0, 0.0))
            col += F
            if wp.static(HALL):
                for f in range(F):
                    down = queue_len[e, f] - queue_up[e, f]
                    out[row, col + f] = wp.float32(wp.where(down > 0, 1.0, 0.0))
                col += F
            for i in range(L):
                for f in range(F):
                    out[row, col] = wp.float32(wp.where(passengers[e, i, f] > 0, 1.0, 0.0))
                    col += 1
            if wp.static(DIRECTION):
                for i in range(L):
                    out[row, col + i] = wp.float32(direction[e, i])
            return
        cap = wp.float32(CAP)
        for i in range(L):
            base = i * LIFT_BLOCK
            floor = round_even(position[e, i])
            load = int(0)
            for g in range(F):
                load += passengers[e, i, g]
            s = status(e, i, position, velocity, target, door_timer)
            out[row, base + 0] = wp.float32(position[e, i] / top)
            out[row, base + 1] = wp.float32(ft(load) / ft(CAP))
            out[row, base + 2] = wp.float32(direction[e, i])
            out[row, base + 3] = wp.float32(wp.where(s == IDLE, 1.0, 0.0))
            out[row, base + 4] = wp.float32(wp.where(s == MOVING, 1.0, 0.0))
            out[row, base + 5] = wp.float32(wp.where(s == DOORS, 1.0, 0.0))
            out[row, base + 6] = wp.float32((ft(target[e, i]) - position[e, i]) / top)
            out[row, base + 7] = wp.float32(velocity[e, i] / VMAX)
            if wp.static(KIN):
                out[row, base + 8] = wp.float32(door_timer[e, i] / DOOR_TIME)
            else:
                out[row, base + 8] = wp.float32(0.0)
            out[row, base + 9] = wp.float32(1.0)  # free: always, with step actions
            # Floor offsets -(F-1)..F-1 from this lift; the window's middle is its floor.
            for o in range(2 * F - 1):
                g = floor - (F - 1) + o
                col = base + RELATIVE_LIFT_FIELDS + o * RELATIVE_FLOOR_FIELDS
                if g < 0 or g > TOP:
                    for k in range(RELATIVE_FLOOR_FIELDS):
                        out[row, col + k] = wp.float32(0.0)
                    continue
                up = queue_len[e, g]
                down = up
                if wp.static(HALL):
                    up = queue_up[e, g]
                    down = queue_len[e, g] - up
                here = int(0)
                for j in range(L):
                    if j != i and round_even(position[e, j]) == g:
                        here += 1
                out[row, col + 0] = wp.float32(1.0)
                out[row, col + 1] = wp.min(wp.float32(passengers[e, i, g]) / cap, wp.float32(1.0))
                out[row, col + 2] = wp.min(wp.float32(up) / cap, wp.float32(1.0))
                out[row, col + 3] = wp.min(wp.float32(down) / cap, wp.float32(1.0))
                out[row, col + 4] = wp.float32(here) / wp.float32(OTHERS)

    @wp.func
    def potential(e: int, position: wp.array2d(dtype=ft), passengers: wp.array3d(dtype=int)) -> ft:
        remaining = ZERO
        for i in range(L):
            for g in range(F):
                remaining += ft(passengers[e, i, g]) * wp.abs(ft(g) - position[e, i])
        return -PROGRESS * remaining / ft(TOP)

    @wp.func
    def clear(
        e: int,
        position: wp.array2d(dtype=ft),
        velocity: wp.array2d(dtype=ft),
        target: wp.array2d(dtype=int),
        door_timer: wp.array2d(dtype=ft),
        passengers: wp.array3d(dtype=int),
        queue_len: wp.array2d(dtype=int),
        queue_up: wp.array2d(dtype=int),
        direction: wp.array2d(dtype=int),
        steps: wp.array(dtype=int),
        last_potential: wp.array(dtype=ft),
    ):
        for i in range(L):
            position[e, i] = ZERO
            velocity[e, i] = ZERO
            target[e, i] = 0
            door_timer[e, i] = ZERO
            direction[e, i] = 0
            for g in range(F):
                passengers[e, i, g] = 0
        for f in range(F):
            queue_len[e, f] = 0  # the slots past the length are never read
            queue_up[e, f] = 0
        steps[e] = 0
        last_potential[e] = ZERO  # nobody on board yet

    # Kernel arguments: inputs, then the state (in-out), then outputs.
    @wp.kernel(module="unique", module_options=opts)
    def reset_kernel(
        seeds: wp.array(dtype=int),
        warmup_destination: wp.array4d(dtype=int),
        warmup_count: wp.array3d(dtype=int),
        inject: int,
        origins: wp.array2d(dtype=ft),
        destinations: wp.array3d(dtype=ft),
        key_times: wp.array(dtype=ft),
        key_weights: wp.array2d(dtype=ft),
        position: wp.array2d(dtype=ft),
        velocity: wp.array2d(dtype=ft),
        target: wp.array2d(dtype=int),
        door_timer: wp.array2d(dtype=ft),
        passengers: wp.array3d(dtype=int),
        queue: wp.array3d(dtype=wp.int8),
        queue_len: wp.array2d(dtype=int),
        queue_up: wp.array2d(dtype=int),
        direction: wp.array2d(dtype=int),
        steps: wp.array(dtype=int),
        last_potential: wp.array(dtype=ft),
        dropped: wp.array(dtype=int),
        rng: wp.array(dtype=wp.uint32),
        episode_return: wp.array(dtype=ft),
        episode_length: wp.array(dtype=int),
        obs: wp.array2d(dtype=wp.float32),
    ):
        e = wp.tid()
        clear(
            e,
            position,
            velocity,
            target,
            door_timer,
            passengers,
            queue_len,
            queue_up,
            direction,
            steps,
            last_potential,
        )
        dropped[e] = 0
        episode_return[e] = ZERO
        episode_length[e] = 0
        state = wp.rand_init(seeds[e])
        for r in range(W):
            if inject != 0:
                for f in range(F):
                    for k in range(warmup_count[e, r, f]):
                        add_arrival(
                            e,
                            f,
                            warmup_destination[e, r, f, k],
                            queue,
                            queue_len,
                            queue_up,
                            dropped,
                        )
            else:
                state = draw_arrivals(
                    e,
                    ZERO,
                    state,
                    origins,
                    destinations,
                    key_times,
                    key_weights,
                    queue,
                    queue_len,
                    queue_up,
                    dropped,
                )
        rng[e] = state
        observe(
            e,
            e,
            obs,
            position,
            velocity,
            target,
            door_timer,
            passengers,
            queue_len,
            queue_up,
            direction,
        )

    @wp.kernel(module="unique", module_options=opts)
    def step_kernel(
        actions: wp.array2d(dtype=int),
        given_destination: wp.array3d(dtype=int),
        given_count: wp.array2d(dtype=int),
        inject: int,
        auto_reset: int,
        origins: wp.array2d(dtype=ft),
        destinations: wp.array3d(dtype=ft),
        key_times: wp.array(dtype=ft),
        key_weights: wp.array2d(dtype=ft),
        position: wp.array2d(dtype=ft),
        velocity: wp.array2d(dtype=ft),
        target: wp.array2d(dtype=int),
        door_timer: wp.array2d(dtype=ft),
        passengers: wp.array3d(dtype=int),
        queue: wp.array3d(dtype=wp.int8),
        queue_len: wp.array2d(dtype=int),
        queue_up: wp.array2d(dtype=int),
        direction: wp.array2d(dtype=int),
        steps: wp.array(dtype=int),
        last_potential: wp.array(dtype=ft),
        dropped: wp.array(dtype=int),
        rng: wp.array(dtype=wp.uint32),
        episode_return: wp.array(dtype=ft),
        episode_length: wp.array(dtype=int),
        obs: wp.array2d(dtype=wp.float32),
        reward_out: wp.array(dtype=ft),
        done_out: wp.array(dtype=int),
        terminal_obs: wp.array2d(dtype=wp.float32),
        final_return: wp.array(dtype=ft),
        final_length: wp.array(dtype=int),
        delivered_out: wp.array(dtype=int),
        boarded_out: wp.array(dtype=int),
    ):
        e = wp.tid()
        state = rng[e]
        if inject != 0:
            for f in range(F):
                for k in range(given_count[e, f]):
                    add_arrival(
                        e, f, given_destination[e, f, k], queue, queue_len, queue_up, dropped
                    )
        else:
            fraction = ft(steps[e]) / ft(MAX_STEPS)
            state = draw_arrivals(
                e,
                fraction,
                state,
                origins,
                destinations,
                key_times,
                key_weights,
                queue,
                queue_len,
                queue_up,
                dropped,
            )

        before = floats()
        for i in range(L):
            before[i] = position[e, i]
        delivered = int(0)
        boarded = int(0)
        empty = int(0)
        for i in range(L):  # in lift order: lower lifts board first
            a = actions[e, i]
            if a == UP or a == DOWN:
                d = 1
                if a == DOWN:
                    d = -1
                move(e, i, d, position, velocity, target, door_timer)
            else:
                can = True
                if wp.static(KIN):
                    can = status(e, i, position, velocity, target, door_timer) == IDLE
                if can:
                    d = 0
                    if wp.static(HALL):
                        d = -1
                        if a == SERVE_UP:
                            d = 1
                    db = serve(e, i, d, position, passengers, queue, queue_len, queue_up)
                    delivered += db[0]
                    boarded += db[1]
                    if db[0] == 0 and db[1] == 0:
                        empty += 1
                    if wp.static(KIN):
                        door_timer[e, i] = DOOR_TIME
        if wp.static(KIN):
            for _ in range(SUBSTEPS):
                for i in range(L):
                    integrate(e, i, position, velocity, target, door_timer)
        for i in range(L):
            if position[e, i] != before[i]:
                if position[e, i] > before[i]:
                    direction[e, i] = 1
                else:
                    direction[e, i] = -1

        reward = ft(delivered)
        if wp.static(HAS_PICKUP):
            reward += PICKUP * ft(boarded)
        if wp.static(HAS_EMPTY):
            reward -= EMPTY * ft(empty)
        if wp.static(HAS_PROGRESS):
            p = potential(e, position, passengers)
            reward += GAMMA * p - last_potential[e]
            last_potential[e] = p
        if wp.static(HAS_WAITING):
            waiting = int(0)
            for f in range(F):
                waiting += queue_len[e, f]
            reward -= WAITING * ft(waiting) / ft(F)

        steps[e] = steps[e] + 1
        done = steps[e] >= MAX_STEPS
        total = episode_return[e] + reward
        length = episode_length[e] + 1
        reward_out[e] = reward
        delivered_out[e] = delivered
        boarded_out[e] = boarded
        final_return[e] = total
        final_length[e] = length
        done_out[e] = wp.where(done, 1, 0)
        if done and auto_reset != 0:
            # The observation the episode ended on, then a fresh episode, all
            # in this thread: the others skip it.
            observe(
                e,
                e,
                terminal_obs,
                position,
                velocity,
                target,
                door_timer,
                passengers,
                queue_len,
                queue_up,
                direction,
            )
            clear(
                e,
                position,
                velocity,
                target,
                door_timer,
                passengers,
                queue_len,
                queue_up,
                direction,
                steps,
                last_potential,
            )
            for _ in range(W):
                state = draw_arrivals(
                    e,
                    ZERO,
                    state,
                    origins,
                    destinations,
                    key_times,
                    key_weights,
                    queue,
                    queue_len,
                    queue_up,
                    dropped,
                )
            total = ZERO
            length = 0
        episode_return[e] = total
        episode_length[e] = length
        rng[e] = state
        observe(
            e,
            e,
            obs,
            position,
            velocity,
            target,
            door_timer,
            passengers,
            queue_len,
            queue_up,
            direction,
        )

    return reset_kernel, step_kernel


class WarpBuildingEnv:
    """BuildingEnv on Warp kernels; the same arguments and attributes as JaxBuildingEnv."""

    def __init__(
        self,
        config=None,
        reward_shaping=False,
        obs_type="relative",
        max_steps=200,
        observe_direction=False,
    ):
        if isinstance(config, str):
            config = PRESETS[config]
        self.config = c = config if config is not None else BuildingConfig()
        if obs_type not in OBS_TYPES:
            raise ValueError(f"the Warp env supports obs_type {OBS_TYPES}, got {obs_type!r}")
        if c.n_floors > 127:
            raise ValueError("queues hold destinations as int8: at most 127 floors")
        if reward_shaping is True:
            reward_shaping = SHAPINGS["default"]
        elif isinstance(reward_shaping, str):
            reward_shaping = SHAPINGS[reward_shaping]
        self.reward_shaping = reward_shaping or None
        self.obs_type = obs_type
        self.max_steps = max_steps
        self.observe_direction = observe_direction
        self.n_actions = c.n_actions
        n = c.n_floors
        if obs_type == "relative":
            self.observation_size = c.n_lifts * (
                RELATIVE_LIFT_FIELDS + (2 * n - 1) * RELATIVE_FLOOR_FIELDS
            )
        else:
            lift_fields = 4 if c.kinematics else 1
            n_waiting = (2 if c.hall_calls else 1) * n
            n_dir = c.n_lifts if observe_direction else 0
            self.observation_size = c.n_lifts * lift_fields + n_waiting + c.n_lifts * n + n_dir
        self._kernels = {}

    def _float(self):
        return jnp.zeros(()).dtype

    def kernels(self):
        """(reset, step) as JAX functions, compiled for the current float type."""
        x64 = self._float() == jnp.float64
        if x64 not in self._kernels:
            ft = wp.float64 if x64 else wp.float32
            reset_kernel, step_kernel = build_kernels(self, ft)
            n_state = len(STATE_FIELDS)
            self._kernels[x64] = (
                wp.jax_kernel(reset_kernel, num_outputs=n_state + 1, in_out_argnames=STATE_FIELDS),
                wp.jax_kernel(step_kernel, num_outputs=n_state + 8, in_out_argnames=STATE_FIELDS),
            )
        return self._kernels[x64]

    def _constants(self):
        f = self._float()
        times, weights = _keyframes(self.config.traffic)
        origins, destinations = traffic_patterns(self.config.n_floors)
        return (
            jnp.asarray(origins, f),
            jnp.asarray(destinations, f),
            jnp.asarray(times, f),
            jnp.asarray(weights, f),
        )

    def empty_state(self, n_envs):
        c, f = self.config, self._float()
        e, lifts, floors = n_envs, c.n_lifts, c.n_floors
        shapes = {
            "position": ((e, lifts), f),
            "velocity": ((e, lifts), f),
            "target": ((e, lifts), jnp.int32),
            "door_timer": ((e, lifts), f),
            "passengers": ((e, lifts, floors), jnp.int32),
            "queue": ((e, floors, MAX_QUEUE), jnp.int8),
            "queue_len": ((e, floors), jnp.int32),
            "queue_up": ((e, floors), jnp.int32),
            "direction": ((e, lifts), jnp.int32),
            "steps": ((e,), jnp.int32),
            "last_potential": ((e,), f),
            "dropped": ((e,), jnp.int32),
            "rng": ((e,), jnp.uint32),
            "episode_return": ((e,), f),
            "episode_length": ((e,), jnp.int32),
        }
        return {name: jnp.zeros(*shapes[name]) for name in STATE_FIELDS}

    def reset(self, state, seeds, warmup=None):
        """Reset every env: (state, obs). `warmup` (Arrivals with leading axes
        (n_envs, WARMUP_ROUNDS)) replaces the draws, to replay Python."""
        reset_kernel, _ = self.kernels()
        n_envs = seeds.shape[0]
        if warmup is None:
            dest = jnp.zeros((1, 1, 1, 1), jnp.int32)
            count = jnp.zeros((1, 1, 1), jnp.int32)
            inject = 0
        else:
            dest, count, inject = warmup.destination, warmup.count, 1
        out = reset_kernel(
            seeds.astype(jnp.int32),
            dest.astype(jnp.int32),
            count.astype(jnp.int32),
            inject,
            *self._constants(),
            *(state[name] for name in STATE_FIELDS),
            launch_dims=(n_envs,),
            output_dims={"obs": (n_envs, self.observation_size)},
        )
        return dict(zip(STATE_FIELDS, out[:-1], strict=True)), out[-1]

    def step(self, state, actions, arrivals=None, auto_reset=True):
        """(state, obs, reward, done, info) for every env.

        With auto_reset, finished envs start a new episode in the same step
        and info["terminal_obs"] holds their last observation (only valid
        where done). `arrivals` (Arrivals with a leading n_envs axis)
        replaces this step's draws.
        """
        _, step_kernel = self.kernels()
        n_envs = actions.shape[0]
        if arrivals is None:
            dest = jnp.zeros((1, 1, 1), jnp.int32)
            count = jnp.zeros((1, 1), jnp.int32)
            inject = 0
        else:
            dest, count, inject = arrivals.destination, arrivals.count, 1
        size = self.observation_size
        out = step_kernel(
            actions.astype(jnp.int32),
            dest.astype(jnp.int32),
            count.astype(jnp.int32),
            inject,
            int(auto_reset),
            *self._constants(),
            *(state[name] for name in STATE_FIELDS),
            launch_dims=(n_envs,),
            output_dims={
                "obs": (n_envs, size),
                "reward_out": (n_envs,),
                "done_out": (n_envs,),
                "terminal_obs": (n_envs, size),
                "final_return": (n_envs,),
                "final_length": (n_envs,),
                "delivered_out": (n_envs,),
                "boarded_out": (n_envs,),
            },
        )
        n = len(STATE_FIELDS)
        state = dict(zip(STATE_FIELDS, out[:n], strict=True))
        obs, reward, done, terminal_obs, ret, length, delivered, boarded = out[n:]
        info = {
            "terminal_obs": terminal_obs,
            "episode_return": ret,
            "episode_length": length,
            "delivered": delivered,
            "boarded": boarded,
        }
        return state, obs, reward, done.astype(bool), info

    def make_vec_env(self, n_envs):
        """(reset, step) with auto-reset, as jax_env.make_vec_env, for jax_ppo."""

        def reset(key):
            seeds = jax.random.randint(key, (n_envs,), 0, 2**31 - 1)
            return self.reset(self.empty_state(n_envs), seeds)

        def step(state, actions):
            return self.step(state, actions)

        return reset, step


def stack_arrivals(per_env):
    """Arrivals with a leading env axis, from one Arrivals per env."""
    return Arrivals(*(jnp.stack(field) for field in zip(*per_env, strict=True)))
