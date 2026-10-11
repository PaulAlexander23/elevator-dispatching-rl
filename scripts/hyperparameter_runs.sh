#!/usr/bin/env bash
# The overnight hyperparameter search and longer runs on full
# (docs/acceleration-plan.md: "Hyperparameter search" and "Longer runs").
# All after the imitation warm start, on the Warp env. About 5 hours on a GTX 1080.
#   uv sync --group warp --group hydra && scripts/hyperparameter_runs.sh
#   uv run python -m elevator_rl.tune_report runs/tune/search/fast-warm-20m --importance
set -e
PY="uv run --no-sync python"
warm="preset=full backend=warp warm_start.samples=100000"
# Optuna at 1,024 envs x 8 steps (20M steps a trial), then at the recipe's 64 x 32 (5M).
$PY -m elevator_rl.tune_search --trials 30 --name fast-warm-20m --space src/elevator_rl/conf/search_fast.yaml \
  --startup 8 --warmup 4000000 $warm ppo.n_envs=1024 ppo.n_steps=8 timesteps=20000000 eval_freq=1000000
$PY -m elevator_rl.tune_search --trials 25 --name full-warm-5m --space src/elevator_rl/conf/search_warm.yaml \
  --startup 8 --warmup 1500000 $warm timesteps=5000000 eval_freq=250000
# The fast search's best settings and the untuned ones, 50M steps x 3 seeds.
fast="$warm ppo.n_envs=1024 ppo.n_steps=8 seed=0,1,2"
best="ppo.learning_rate=5.21e-05 ppo.n_epochs=10 ppo.batch_size=2048 ppo.ent_coef=2.77e-05 ppo.clip_range=0.1 ppo.gae_lambda=0.95 ppo.gamma=0.99 ppo.vf_coef=0.25"
$PY -m elevator_rl.tune -m $fast $best timesteps=50000000 eval_freq=2500000 hydra.sweep.dir=runs/long/best-50m
$PY -m elevator_rl.tune -m $fast ppo.learning_rate=3e-5 ppo.batch_size=2048 timesteps=50000000 eval_freq=2500000 hydra.sweep.dir=runs/long/base-50m
# Network size with the best settings, 20M steps.
for net in 64,64 128,128 512,512; do
  $PY -m elevator_rl.tune -m $fast $best "ppo.net_arch=[$net]" timesteps=20000000 eval_freq=1000000 hydra.sweep.dir=runs/long/net-${net/,/x}
done
