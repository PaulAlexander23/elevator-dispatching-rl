"""The building as a multi-agent env: one agent per lift, in Mava's layout.

The API is Jumanji's, which Mava uses: `reset(key) -> (state, timestep)` and
`step(state, action) -> (state, timestep)`, pure functions with the PRNG key
in the state, plus `num_agents`, `time_limit` and `action_dim`. Observations
carry an agents axis:

- `agents_view` (lifts, block): each lift's block of the "relative"
  observation, its own state and the floors around it;
- `action_mask` (lifts, actions): all true; no action is ever invalid (in
  target mode a lift with a goal ignores its action, whatever it is);
- `global_state` (lifts, obs): the whole observation, repeated per lift, for
  a centralised critic;
- `step_count` (lifts,).

The reward is the team's, repeated per lift. Episodes end only at the time
limit, a truncation, so the discount stays 1 at the last step.

The layout follows Mava's `mava/wrappers/jumanji.py` without depending on
Mava or Jumanji (Mava pins JAX 0.5), so a Mava wrapper later is a few lines.
`MultiAgentEnv` wraps one `JaxBuildingEnv` and can be `jit`ted and `vmap`ped;
`make_vec_env` batches either the JAX or the Warp env, with auto-reset, for
our own MAPPO.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


class StepType:
    """As `jumanji.types.StepType` (dm_env's)."""

    FIRST = 0
    MID = 1
    LAST = 2


class Observation(NamedTuple):
    agents_view: jax.Array  # (lifts, block)
    action_mask: jax.Array  # (lifts, actions) bool
    step_count: jax.Array  # (lifts,) int


class ObservationGlobalState(NamedTuple):
    agents_view: jax.Array  # (lifts, block)
    action_mask: jax.Array  # (lifts, actions) bool
    global_state: jax.Array  # (lifts, obs)
    step_count: jax.Array  # (lifts,) int


class TimeStep(NamedTuple):
    """As `jumanji.types.TimeStep`, with a reward and discount per lift."""

    step_type: jax.Array  # () int, StepType
    reward: jax.Array  # (lifts,)
    discount: jax.Array  # (lifts,)
    observation: Any  # Observation or ObservationGlobalState
    extras: dict

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST


class ActionSpec(NamedTuple):
    """As a Jumanji `MultiDiscreteArray`: one discrete action per lift."""

    num_values: jax.Array  # (lifts,) int
    dtype: Any = jnp.int32

    @property
    def shape(self):
        return self.num_values.shape


def _check(env):
    if env.obs_type != "relative":
        raise ValueError(
            "the multi-agent env needs the per-lift blocks of the relative observation"
        )


class _Layout:
    """The agents-axis view of single-agent observations, shared by both wrappers."""

    def __init__(self, env, global_state=True):
        _check(env)
        self.env = env
        self.global_state = global_state
        self.num_agents = env.config.n_lifts
        self.action_dim = env.n_actions
        self.time_limit = env.max_steps
        self.agent_obs_size = env.observation_size // self.num_agents

    def observation(self, obs, steps):
        """obs (..., obs_size) and steps (...) -> observation with an agents axis."""
        lead = obs.shape[:-1]
        n = self.num_agents
        view = obs.reshape(*lead, n, self.agent_obs_size)
        mask = jnp.ones((*lead, n, self.action_dim), bool)
        step_count = jnp.broadcast_to(steps[..., None], (*lead, n)).astype(jnp.int32)
        if not self.global_state:
            return Observation(view, mask, step_count)
        state = jnp.broadcast_to(obs[..., None, :], (*lead, n, obs.shape[-1]))
        return ObservationGlobalState(view, mask, state, step_count)

    def per_agent(self, x):
        """A team value (...) repeated per lift (..., lifts)."""
        return jnp.broadcast_to(x[..., None], (*x.shape, self.num_agents))

    def observation_spec(self):
        """The observation's shapes and dtypes, as jax.ShapeDtypeStruct."""
        f = jnp.float32  # the observation is float32 even under jax_enable_x64
        n = self.num_agents
        fields = {
            "agents_view": jax.ShapeDtypeStruct((n, self.agent_obs_size), f),
            "action_mask": jax.ShapeDtypeStruct((n, self.action_dim), bool),
            "step_count": jax.ShapeDtypeStruct((n,), jnp.int32),
        }
        if not self.global_state:
            return Observation(**fields)
        state = jax.ShapeDtypeStruct((n, self.env.observation_size), f)
        return ObservationGlobalState(global_state=state, **fields)

    def action_spec(self):
        return ActionSpec(jnp.full(self.num_agents, self.action_dim, jnp.int32))


class MultiAgentEnv(_Layout):
    """One `JaxBuildingEnv` as a Jumanji-style multi-agent env (no auto-reset).

    `step` takes one action per lift, (lifts,) int, as the single-agent env.
    `extras` holds the single-agent env's info ("delivered", "boarded").
    """

    def reset(self, key):
        state, obs = self.env.reset(key)
        reward = jnp.zeros(self.num_agents, state.position.dtype)  # the env's float
        timestep = TimeStep(
            jnp.asarray(StepType.FIRST, jnp.int32),
            reward,
            jnp.ones_like(reward),
            self.observation(obs, state.steps),
            {"delivered": jnp.zeros((), jnp.int32), "boarded": jnp.zeros((), jnp.int32)},
        )
        return state, timestep

    def step(self, state, action):
        state, obs, reward, truncated, info = self.env.step(state, action)
        step_type = jnp.where(truncated, StepType.LAST, StepType.MID).astype(jnp.int32)
        reward = self.per_agent(reward)
        info = {k: v.astype(jnp.int32) for k, v in info.items()}
        observation = self.observation(obs, state.steps)
        return state, TimeStep(step_type, reward, jnp.ones_like(reward), observation, info)


def _steps(vec_state):
    """Steps into the episode per env, from a JAX VecState or the Warp env's state."""
    if isinstance(vec_state, dict):
        return vec_state["steps"]
    return vec_state.env.steps


class VecMultiAgentEnv(_Layout):
    """Batched (reset, step) with auto-reset over the JAX or Warp env.

    Every array gains a leading env axis. Where an episode ends, the timestep
    is LAST but its observation is already the next episode's first, as in
    the envs' own `make_vec_env`; extras["terminal_observation"] holds the
    observation it ended on, for bootstrapping at the time limit.
    """

    def __init__(self, env, n_envs, global_state=True):
        super().__init__(env, global_state)
        self.n_envs = n_envs
        self._reset, self._step = env.make_vec_env(n_envs)

    def reset(self, key):
        vec_state, obs = self._reset(key)
        reward = jnp.zeros((self.n_envs, self.num_agents))
        timestep = TimeStep(
            jnp.full(self.n_envs, StepType.FIRST, jnp.int32),
            reward,
            jnp.ones_like(reward),
            self.observation(obs, _steps(vec_state)),
            {},
        )
        return vec_state, timestep

    def step(self, vec_state, action):
        """action (n_envs, lifts) int."""
        vec_state, obs, reward, done, info = self._step(vec_state, action)
        step_type = jnp.where(done, StepType.LAST, StepType.MID).astype(jnp.int32)
        reward = self.per_agent(reward)
        steps = _steps(vec_state)
        extras = {
            "terminal_observation": self.observation(
                info["terminal_obs"], jnp.full_like(steps, self.time_limit)
            ),
            "episode_return": info["episode_return"],
            "episode_length": info["episode_length"],
        }
        observation = self.observation(obs, steps)
        return vec_state, TimeStep(step_type, reward, jnp.ones_like(reward), observation, extras)


def make_vec_env(env, n_envs, global_state=True):
    return VecMultiAgentEnv(env, n_envs, global_state)
