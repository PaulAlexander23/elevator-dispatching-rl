from gymnasium import Env
from gymnasium.utils.env_checker import check_env
from env import LiftEnv


def test_env_is_gym():
    env = LiftEnv()

    assert isinstance(env, Env)
    check_env(env)


def test_env_random_agent():
    env = LiftEnv()
    env.render_mode = "human"
    action_space = env.action_space
    env.reset()
    for n in range(10):
        action = action_space.sample()
        env.step(action)
        env.render()


def test_env_random_agent():
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


if __name__ == "__main__":
    test_env_random_agent()
