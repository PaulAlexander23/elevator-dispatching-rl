import time
from env import LiftEnv
import os
import psutil

process = psutil.Process(os.getpid())


env = LiftEnv(reward_shaping=True, multi_discrete=True)
obs, info = env.reset(seed=0)

for chunk in range(20):
    t0 = time.perf_counter()

    for _ in range(100_000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()
    dt = time.perf_counter() - t0

    print(
        f"steps={(chunk + 1) * 100_000:7d} "
        f"FPS={100_000 / dt:8.1f} "
        f"RAM: {process.memory_info().rss / 1024**3:.2f} GB "
    )
