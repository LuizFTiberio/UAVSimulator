"""MuJoCo passive viewer with drone tracking."""

import time as _time

import mujoco
import mujoco.viewer
import numpy as np

from uavsim.sim.mujoco_sim import MuJoCoSimulator


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
    ):
        self.sim = sim
        self.cam_distance = cam_distance
        self.cam_elevation = cam_elevation
        self.cam_azimuth = cam_azimuth
        self.track = track
        self.viewer = None

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
        """Render one frame, optionally tracking the drone."""
        if self.viewer is None:
            return
        if self.track:
            self._follow_drone()
        self.viewer.sync()
        if real_time_factor > 0.0:
            _time.sleep(self.sim.dt * real_time_factor)

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
