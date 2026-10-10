"""Parity between the Python env and the C++ port in cpp/.

Skipped unless the C++ tools are built:

    cmake -S cpp -B cpp/build && cmake --build cpp/build

(or point ELEVATOR_CPP_BUILD at another build directory).
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from elevator_rl import trace
from elevator_rl.building import PRESETS
from elevator_rl.building_env import BuildingEnv

BUILD = Path(os.environ.get("ELEVATOR_CPP_BUILD", Path(__file__).parents[1] / "cpp" / "build"))
REPLAY = BUILD / "elevator_replay"
BENCH = BUILD / "elevator_bench"

pytestmark = pytest.mark.skipif(
    not (REPLAY.exists() and BENCH.exists()), reason=f"C++ tools not built in {BUILD}"
)


def run(*args):
    result = subprocess.run([str(a) for a in args], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def bench(preset, **options):
    args = [BENCH, "--preset", preset]
    for key, value in options.items():
        args += [f"--{key.replace('_', '-')}", value]
    pairs = (line.split("=", 1) for line in run(*args).splitlines())
    return {key: value for key, value in pairs}


@pytest.mark.parametrize("preset", PRESETS)
def test_cpp_replays_python_traces_exactly(preset, tmp_path):
    path = tmp_path / f"{preset}.trace"
    path.write_text("\n".join(trace.record(PRESETS[preset], n_steps=1500, seed=3)) + "\n")
    assert "1500 steps match" in run(REPLAY, path)


def test_replay_reports_a_mismatch(tmp_path):
    lines = trace.record(PRESETS["kinematic"], n_steps=50)
    i = next(i for i, line in enumerate(lines) if line.startswith("result"))
    lines[i] = "result 99.0 0"
    path = tmp_path / "bad.trace"
    path.write_text("\n".join(lines) + "\n")
    result = subprocess.run([REPLAY, path], capture_output=True, text=True)
    assert result.returncode == 1
    assert "result: python 99.0" in result.stderr


@pytest.mark.parametrize("preset", PRESETS)
def test_cpp_presets_match_python(preset):
    output = run(BENCH, "--preset", preset, "--print-config").strip()
    assert output == trace.config_line(PRESETS[preset])


def python_random_rewards(preset, episodes, seed=0):
    env = BuildingEnv(preset)
    env.action_space.seed(seed)
    totals = []
    for episode in range(episodes):
        env.reset(seed=seed + episode)
        total, truncated = 0.0, False
        while not truncated:
            _, reward, _, truncated, _ = env.step(env.action_space.sample())
            total += reward
        totals.append(total)
    return np.array(totals)


@pytest.mark.parametrize("preset", ["original", "multi", "full"])
def test_cpp_random_policy_statistics_match_python(preset):
    """Different random numbers, same distributions: compare the means."""
    python = python_random_rewards(preset, episodes=80)
    cpp = bench(preset, steps=400_000)
    assert int(cpp["dropped"]) == 0

    # The C++ mean is over 2,000 episodes, so its error is small next to Python's.
    standard_error = python.std(ddof=1) / np.sqrt(len(python))
    assert abs(float(cpp["mean_episode_reward"]) - python.mean()) < 4 * standard_error

    config = PRESETS[preset]
    expected = config.arrival_probability * config.n_floors
    assert float(cpp["arrivals_per_step"]) == pytest.approx(expected, rel=0.02)
