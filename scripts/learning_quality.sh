#!/usr/bin/env bash
# Learning quality at the fast settings (docs/acceleration-plan.md, overnight
# backlog run): the full preset after the imitation warm start, lr 3e-5,
# 3 seeds per setting, on the Warp env. First 5M steps each (equal samples),
# then each fast setting for about the recipe's 2.7 minutes (equal time).
#   uv sync --group warp --group hydra && scripts/learning_quality.sh
#   uv run python -m elevator_rl.tune_report runs/lq/* runs/lq-time/*
set -e
common="preset=full backend=warp warm_start.samples=100000 ppo.learning_rate=3e-5 seed=0,1,2"
run() { out=$1; shift; uv run --no-sync python -m elevator_rl.tune -m $common "$@" hydra.sweep.dir=runs/$out; }
steps="timesteps=5000000 eval_freq=250000"
run lq/e64-mb64 ppo.n_envs=64 ppo.n_steps=32 ppo.batch_size=64 $steps
run lq/e64-mb512 ppo.n_envs=64 ppo.n_steps=32 ppo.batch_size=512 $steps
run lq/e64-mb2048 ppo.n_envs=64 ppo.n_steps=32 ppo.batch_size=2048 $steps
run lq/e1024-mb2048 ppo.n_envs=1024 ppo.n_steps=8 ppo.batch_size=2048 $steps
run lq/e16384-mb16384 ppo.n_envs=16384 ppo.n_steps=8 ppo.batch_size=16384 $steps
run lq-time/e64-mb512 ppo.n_envs=64 ppo.n_steps=32 ppo.batch_size=512 timesteps=10000000 eval_freq=500000
run lq-time/e64-mb2048 ppo.n_envs=64 ppo.n_steps=32 ppo.batch_size=2048 timesteps=12000000 eval_freq=600000
run lq-time/e1024-mb2048 ppo.n_envs=1024 ppo.n_steps=8 ppo.batch_size=2048 timesteps=40000000 eval_freq=2000000
run lq-time/e16384-mb16384 ppo.n_envs=16384 ppo.n_steps=8 ppo.batch_size=16384 timesteps=57000000 eval_freq=2800000
