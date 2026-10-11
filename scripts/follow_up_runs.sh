#!/usr/bin/env bash
# The GPU runs of docs/follow-up-plan.md, items 2, 3 and 5, one job at a time
# (each JAX process takes 75% of the GPU's memory). Pass the stages to run:
#   uv sync --group warp --group hydra
#   scripts/follow_up_runs.sh profile arrivals encoder mappo transfer
# then tabulate a sweep with
#   uv run python -m elevator_rl.tune_report runs/follow-up/<stage>/*
# Stages, with rough times on a GTX 1080:
#   profile   env and PPO profiles before choosing encoder widths (minutes)
#   arrivals  tuned PPO on the JAX env, 20M steps x 3 seeds: learning with
#             the new arrival draw, against 73.6 ± 2.0 on Warp (about 30 min)
#   encoder   shared per-lift encoder widths at 20M steps x 3 seeds (about 40 min)
#   mappo     MAPPO with and without lift ids at 20M steps x 3 seeds (1-2 h: its
#             actor runs once per lift, 3.6x slower than PPO on the CPU)
#   transfer  MAPPO trained with 4 lifts on 2, 3, 6, 8 lifts, 3 seeds (about 2 h)
set -e
out=runs/follow-up
mkdir -p $out
tune() { dir=$1; shift; uv run --no-sync python -m elevator_rl.tune -m --config-name tuned seed=0,1,2 "$@" hydra.sweep.dir=$out/$dir; }
at20m="timesteps=20000000"
for stage in "$@"; do
  case $stage in
    profile)
      uv run --no-sync python scripts/profile_jax_env.py | tee $out/profile_jax_env.md
      uv run --no-sync python scripts/time_jax_env.py --backends jax warp --devices gpu | tee $out/time_jax_env.md
      for cfg in "0 256 256" "128 256" "128 256 256" "64 256 256" "32 256 256" "32 256"; do
        set -- $cfg; enc=$1; shift
        echo "lift_encoder $enc, net_arch $*" | tee -a $out/profile_jax_ppo.md
        uv run --no-sync python scripts/profile_jax_ppo.py --backends warp --settings 1024x8:2048 \
          --n-epochs 10 --lift-encoder $enc --net-arch "$@" | tee -a $out/profile_jax_ppo.md
      done
      for arch in "256 256" "128 128"; do
        echo "mappo, net_arch $arch" | tee -a $out/profile_jax_ppo.md
        uv run --no-sync python scripts/profile_jax_ppo.py --backends warp --settings 1024x8:2048 \
          --n-epochs 10 --algo mappo --net-arch $arch | tee -a $out/profile_jax_ppo.md
      done
      ;;
    arrivals) tune arrivals backend=jax $at20m ;;
    encoder)
      tune encoder-32 ppo.lift_encoder=32 $at20m
      tune encoder-64 ppo.lift_encoder=64 $at20m
      tune encoder-128 ppo.lift_encoder=128 'ppo.net_arch=[256]' $at20m
      ;;
    mappo)
      tune mappo ppo.algo=mappo $at20m
      tune mappo-no-id ppo.algo=mappo ppo.lift_id=false $at20m
      ;;
    transfer)
      uv run --no-sync python scripts/mappo_transfer.py --timesteps 20000000 --finetune 5000000 \
        --seeds 0 1 2 --out $out/transfer | tee $out/transfer.md
      ;;
    *) echo "unknown stage $stage" >&2; exit 1 ;;
  esac
done
