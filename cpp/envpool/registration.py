"""Register Elevator-v0: pass preset= (or building=, a config line) to make."""

from envpool.registration import register

register(
    task_id="Elevator-v0",
    import_path="envpool.elevator",
    spec_cls="ElevatorEnvSpec",
    dm_cls="ElevatorDMEnvPool",
    gymnasium_cls="ElevatorGymnasiumEnvPool",
    max_episode_steps=200,
)
