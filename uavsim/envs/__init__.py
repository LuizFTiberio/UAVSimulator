"""Gymnasium-compatible environments."""

try:
    import gymnasium

    gymnasium.register(
        id="uavsim/Hover-v0",
        entry_point="uavsim.envs.hover_env:HoverEnv",
    )
except ImportError:
    pass
