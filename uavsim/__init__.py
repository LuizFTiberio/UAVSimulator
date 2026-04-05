"""uavsim — JAX-first UAV simulator with MuJoCo physics."""

import os as _os

# Auto-configure GPU on import unless user already set JAX_PLATFORMS
# or explicitly opted out with UAVSIM_GPU=cpu.
if _os.environ.get("UAVSIM_GPU", "auto").lower() != "cpu":
    from uavsim.core.gpu import configure_gpu as _configure_gpu
    _configure_gpu()
