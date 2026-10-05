import time
from env import LiftEnv
import os
import psutil
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


process = psutil.Process(os.getpid())

envs = DummyVecEnv(
    [lambda: LiftEnv(obs_type="multi_discrete")])

model = PPO("MlpPolicy", envs, device="cpu", verbose=1)
for chunk in range(20):
    t0 = time.perf_counter()

    model.learn(total_timesteps=100_000, log_interval=10,
                reset_num_timesteps=False)

    dt = time.perf_counter() - t0

    print(
        f"steps={(chunk + 1) * 100_000:7d} "
        f"FPS={100_000 / dt:8.1f} "
        f"RAM: {process.memory_info().rss / 1024**3:.2f} GB "
    )
