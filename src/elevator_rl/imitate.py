"""Warm-start a PPO policy by imitating the collective-control heuristic.

PPO from scratch struggles to discover dispatching on the 30-floor presets:
a delivery takes dozens of correct steps in a row, so random exploration
rarely finds one. Behaviour cloning copies `CollectivePolicy` into the PPO
network (and fits the value head to the heuristic's discounted returns), so
PPO starts from a working policy and can improve on it.
"""

import numpy as np
import torch

from elevator_rl.baselines import CollectivePolicy


def collect(
    preset,
    n_samples,
    reward_shaping="default",
    gamma=0.99,
    seed=0,
    obs_type="custom",
    observe_direction=False,
):
    """Run the heuristic and return observations, actions and discounted returns.

    The heuristic reads the "custom" observation; `obs_type` and
    `observe_direction` choose what is recorded for the policy being trained.
    """
    from elevator_rl.building_env import BuildingEnv

    env = BuildingEnv(preset, reward_shaping=reward_shaping, observe_direction=observe_direction)
    policy = CollectivePolicy(preset)

    def record():
        env.obs_type = obs_type
        obs = env._observation()
        env.obs_type = "custom"
        return obs

    observations, actions, returns = [], [], []
    episode = 0
    while len(observations) < n_samples:
        obs, _ = env.reset(seed=seed + episode)
        policy.reset()
        rewards, truncated = [], False
        while not truncated:
            (action,), _ = policy.predict([obs])
            observations.append(record())
            actions.append(action)
            obs, reward, _, truncated, _ = env.step(action)
            rewards.append(reward)
        g, episode_returns = 0.0, []
        for reward in reversed(rewards):
            g = reward + gamma * g
            episode_returns.append(g)
        returns.extend(reversed(episode_returns))
        episode += 1
    n = n_samples
    return np.array(observations[:n]), np.array(actions[:n]), np.array(returns[:n], np.float32)


def pretrain(model, observations, actions, returns, epochs=10, batch_size=256, lr=1e-3, seed=0):
    """Fit the policy to the actions (cross-entropy) and the value head to the returns.

    Returns the final epoch's mean action log-likelihood and value loss.
    """
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    act_t = torch.as_tensor(actions, device=policy.device)
    ret_t = torch.as_tensor(returns, device=policy.device)
    log_likelihood = value_loss = float("nan")
    for _ in range(epochs):
        order = rng.permutation(len(actions))
        totals = []
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            idx = torch.as_tensor(batch, device=policy.device)
            obs = policy.obs_to_tensor(observations[batch])[0]
            values, log_prob, _ = policy.evaluate_actions(obs, act_t[idx])
            v_loss = torch.nn.functional.mse_loss(values.flatten(), ret_t[idx])
            loss = -log_prob.mean() + 0.5 * v_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals.append((log_prob.mean().item(), v_loss.item()))
        log_likelihood, value_loss = np.mean(totals, axis=0)
    return float(log_likelihood), float(value_loss)


def warm_start(
    model,
    preset,
    n_samples=100_000,
    reward_shaping="default",
    epochs=15,
    seed=0,
    obs_type="custom",
    observe_direction=False,
):
    observations, actions, returns = collect(
        preset, n_samples, reward_shaping, model.gamma, seed, obs_type, observe_direction
    )
    return pretrain(model, observations, actions, returns, epochs=epochs, seed=seed)
