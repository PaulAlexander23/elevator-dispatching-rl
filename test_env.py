from gymnasium import Env
from gymnasium.utils.env_checker import check_env
from env import LiftEnv
from sim import LiftState


def test_env_is_gym():
    env = LiftEnv(multi_discrete=True)

    assert isinstance(env, Env)
    check_env(env)


def test_env_random_agent():
    env = LiftEnv(reward_shaping=True)
    env.render_mode = "human"
    action_space = env.action_space
    env.reset()
    for n in range(10):
        # action = action_space.sample()
        action = (n % 2)*2
        print(action)
        env.step(action)
        env.render()


def test_env_random_agent_reward():
    env = LiftEnv()
    action_space = env.action_space
    env.reset()
    total_reward = 0
    terminated = False
    while not terminated:
        action = action_space.sample()
        _, reward, terminated, _, _ = env.step(action)
        total_reward += reward

    print(total_reward)


def test_obs():
    env = LiftEnv()
    env.render_mode = "human"
    env.reset()
    env.render()
    action_space = env.action_space
    print(env._map_state_to_obs(env.sim.state()))
    for n in range(10):
        action = action_space.sample()
        env.step(action)
    env.render()
    print(env._map_state_to_obs(env.sim.state()))


def test_obs_mb():
    env = LiftEnv(multi_binary=True)
    env.render_mode = "human"
    env.reset()
    env.render()
    action_space = env.action_space
    print(env._map_state_to_obs(env.sim.state()))
    for n in range(10):
        action = action_space.sample()
        env.step(action)
    env.render()
    print(env._map_state_to_obs(env.sim.state()))


def test_obs_md():
    env = LiftEnv(multi_discrete=True)
    env.render_mode = "human"
    env.reset()
    env.render()
    action_space = env.action_space
    print(env._map_state_to_obs(env.sim.state()))
    for n in range(10):
        action = action_space.sample()
        env.step(action)
    env.render()
    print(env._map_state_to_obs(env.sim.state()))


if __name__ == "__main__":
    test_env_is_gym()
    # test_env_random_agent()
    # test_env_random_agent_reward()
    # test_obs()
    # test_obs_mb()
    test_obs_md()
