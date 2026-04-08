"""MuJoCo passive viewer with drone tracking."""

import logging
import time as _time

import mujoco
import mujoco.viewer
import numpy as np

from uavsim.sim.mujoco_sim import MuJoCoSimulator

logger = logging.getLogger(__name__)


class SimulationVisualizer:
    """Real-time passive viewer that keeps the drone centred in the frame.

    Usage
    -----
    vis = SimulationVisualizer(sim)
    vis.launch()
    while vis.is_running:
        sim.step(cmd)
        vis.sync()
    vis.close()
    """

    def __init__(
        self,
        sim: MuJoCoSimulator,
        cam_distance: float = 4.0,
        cam_elevation: float = -25.0,
        cam_azimuth: float = 135.0,
        track: bool = True,
        render_fps: float = 60.0,
    ):
        self.sim = sim
        self.cam_distance = cam_distance
        self.cam_elevation = cam_elevation
        self.cam_azimuth = cam_azimuth
        self.track = track
        self.render_fps = render_fps
        self.viewer = None
        self._last_render_time: float = 0.0
        self._wall_start: float = 0.0
        self._sim_start: float = 0.0

    # ── public API ───────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        if self.viewer is None:
            return False
        try:
            return self.viewer.is_running()
        except Exception:
            return False

    def launch(self) -> "SimulationVisualizer":
        self.viewer = mujoco.viewer.launch_passive(
            self.sim.model, self.sim.data)
        self._configure_scene()
        self._set_initial_camera()
        self._log_renderer_info()
        self._wall_start = _time.monotonic()
        self._sim_start = self.sim.current_time
        self._last_render_time = 0.0
        return self

    def draw_gates(self, gates: list[dict]) -> None:
        """Draw square gate frames using custom scene geoms.

        Parameters
        ----------
        gates : list of dict
            Each dict must contain:
            - 'corners': list of 4 array-like (3,) corner positions
            - 'rgba': array-like (4,) RGBA colour in [0, 1]
        """
        if self.viewer is None:
            return
        try:
            scn = self.viewer.user_scn
        except AttributeError:
            return
        with self.viewer.lock():
            scn.ngeom = 0
            beam_width = 0.06
            for gate in gates:
                corners = [np.asarray(c, dtype=float) for c in gate["corners"]]
                rgba = np.asarray(gate["rgba"], dtype=np.float32)
                for i1, i2 in ((0, 1), (1, 2), (2, 3), (3, 0)):
                    if scn.ngeom >= scn.maxgeom:
                        return
                    g = scn.geoms[scn.ngeom]
                    p1, p2 = corners[i1], corners[i2]
                    mujoco.mjv_connector(
                        g,
                        mujoco.mjtGeom.mjGEOM_CAPSULE,
                        beam_width,
                        p1.reshape(3, 1),
                        p2.reshape(3, 1),
                    )
                    g.rgba[:] = rgba
                    scn.ngeom += 1

    def sync(self, real_time_factor: float = 1.0) -> None:
        """Render one frame if enough time has passed, with wall-clock sync.

        Only renders at ``render_fps`` (default 60 Hz).  Between renders
        the call returns immediately, so physics can run at full speed.

        Parameters
        ----------
        real_time_factor : float
            1.0 = real-time, 2.0 = 2× speed, 0 = no throttling (fast-forward).
        """
        if self.viewer is None:
            return

        # Check if it's time to render a new frame
        sim_elapsed = self.sim.current_time - self._sim_start
        render_interval = 1.0 / self.render_fps
        if sim_elapsed - self._last_render_time < render_interval:
            return  # skip — not time yet

        self._last_render_time = sim_elapsed

        if self.track:
            self._follow_drone()
        self.viewer.sync()

        # Wall-clock sync: sleep only if physics is ahead of desired pace
        if real_time_factor > 0.0:
            wall_elapsed = _time.monotonic() - self._wall_start
            desired_wall = sim_elapsed / real_time_factor
            ahead = desired_wall - wall_elapsed
            if ahead > 0.001:
                _time.sleep(ahead)

    def close(self) -> None:
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None

    # ── private ──────────────────────────────────────────────────────────

    def _configure_scene(self) -> None:
        with self.viewer.lock():
            scn = self.viewer.opt
            scn.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
            scn.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = False
            scn.flags[mujoco.mjtVisFlag.mjVIS_COM] = False
            scn.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = False
            scn.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False

    def _set_initial_camera(self) -> None:
        with self.viewer.lock():
            cam = self.viewer.cam
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.distance = self.cam_distance
            cam.elevation = self.cam_elevation
            cam.azimuth = self.cam_azimuth
            cam.lookat[:] = self.sim.data.qpos[0:3]

    def _follow_drone(self) -> None:
        with self.viewer.lock():
            pos = self.sim.data.qpos[0:3].copy()
            alpha = 0.15
            self.viewer.cam.lookat[:] = (
                (1.0 - alpha) * self.viewer.cam.lookat + alpha * pos
            )

    def _log_renderer_info(self) -> None:
        """Log the OpenGL renderer being used by MuJoCo."""
        try:
            ctx = self.sim.model.vis.global_
            gl_renderer = mujoco.mj_getPluginConfig(self.sim.model, -1, "renderer") if hasattr(mujoco, 'mj_getPluginConfig') else ""
        except Exception:
            gl_renderer = ""
        from uavsim.core.gpu import gpu_info
        info = gpu_info()
        logger.info(
            "Viewer launched — rendering on %s (%s)",
            info.device_name or "default GPU",
            info.vendor.value,
        )
