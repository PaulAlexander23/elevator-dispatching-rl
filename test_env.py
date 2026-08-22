from gymnasium import Env
from gymnasium.utils.env_checker import check_env
from env import LiftEnv


def test_env_is_gym():
    env = LiftEnv()

    assert isinstance(env, Env)
    check_env(env)


if __name__ == "__main__":
    test_env_is_gym()
