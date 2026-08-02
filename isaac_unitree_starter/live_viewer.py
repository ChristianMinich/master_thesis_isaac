"""Live 2D viewer for the captured sensor data: RGB | depth | top-down LiDAR,
side by side in one Kit UI window.

Deliberately NOT a 3D-scene overlay (like RTX Lidar's "draw-point-cloud" debug
writer): that draws into the shared Hydra scene, so it shows up in every
render product looking at that scene -- including the robot's own RGB camera
sensor, not just the main viewport. This viewer instead rasterizes its own
panels (via cv2's image-processing functions -- resize/colormap/drawing, which
don't need a GUI backend) from the same numpy arrays SensorManager/DataLogger
already read, with no connection to any 3D camera, so it can never contaminate
a camera's image.

Displayed via omni.ui.ByteImageProvider rather than cv2.imshow: Isaac Sim's
bundled OpenCV is a headless build (no GTK/Qt/Cocoa), so cv2's own window
functions raise "function not implemented" regardless of a working X display --
this is a build-time limitation of that package, not a runtime/display issue.
ByteImageProvider is Kit's native mechanism for a widget whose pixel data is
updated every frame (the same pattern isaacsim.replicator.mobility_gen.ui uses
for its occupancy-map preview).
"""

from __future__ import annotations

import cv2
import numpy as np
import omni.ui as ui

_WINDOW_TITLE = "Sensor Feed (RGB | Depth | LiDAR top-down)"

PANEL_WIDTH = 400
PANEL_HEIGHT = 300
DEPTH_MAX_METERS = 10.0  # depth values beyond this clip to the colormap's max
LIDAR_PANEL_SIZE = 400  # square, pixels
LIDAR_RANGE_METERS = 8.0  # half-width of the top-down view, in meters


class LiveViewer:
    """Owns the Kit UI window and renders one combined frame per update() call."""

    def __init__(self) -> None:
        self._panel_w = PANEL_WIDTH
        self._panel_h = PANEL_HEIGHT
        self._lidar_size = LIDAR_PANEL_SIZE

        combined_w = 2 * self._panel_w + self._panel_h
        combined_h = self._panel_h
        self._image_provider = ui.ByteImageProvider()
        self._window = ui.Window(_WINDOW_TITLE, width=combined_w, height=combined_h + 24)
        with self._window.frame:
            with ui.VStack():
                ui.ImageWithProvider(self._image_provider, width=combined_w, height=combined_h)

    def _rgb_panel(self, rgb: np.ndarray | None) -> np.ndarray:
        if rgb is None:
            panel = np.zeros((self._panel_h, self._panel_w, 3), dtype=np.uint8)
            cv2.putText(panel, "no RGB", (10, self._panel_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            return panel
        bgr = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR)
        return cv2.resize(bgr, (self._panel_w, self._panel_h))

    def _depth_panel(self, depth: np.ndarray | None) -> np.ndarray:
        if depth is None:
            panel = np.zeros((self._panel_h, self._panel_w, 3), dtype=np.uint8)
            cv2.putText(panel, "no depth", (10, self._panel_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            return panel
        depth_2d = depth[:, :, 0] if depth.ndim == 3 else depth
        clipped = np.clip(depth_2d, 0.0, DEPTH_MAX_METERS)
        normalized = (clipped / DEPTH_MAX_METERS * 255.0).astype(np.uint8)
        colorized = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        return cv2.resize(colorized, (self._panel_w, self._panel_h))

    def _lidar_panel(self, points: np.ndarray) -> np.ndarray:
        size = self._lidar_size
        panel = np.zeros((size, size, 3), dtype=np.uint8)
        center = size // 2
        scale = center / LIDAR_RANGE_METERS  # pixels per meter

        cv2.circle(panel, (center, center), 4, (0, 0, 255), -1)  # robot marker
        cv2.line(panel, (center, center), (center, center - 15), (0, 0, 255), 1)  # forward tick

        if points.shape[0] > 0:
            # Sensor-local frame: x=forward, y=left. Image: right=+x_forward drawn up,
            # so plot (col, row) = (center - y*scale, center - x*scale) for an
            # intuitive "robot facing up the screen" top-down layout.
            cols = (center - points[:, 1] * scale).astype(np.int32)
            rows = (center - points[:, 0] * scale).astype(np.int32)
            in_bounds = (cols >= 0) & (cols < size) & (rows >= 0) & (rows < size)
            panel[rows[in_bounds], cols[in_bounds]] = (0, 255, 0)

        return panel

    def update(self, camera_reading: dict, lidar_reading: dict) -> None:
        """Render one combined frame from the latest camera + LiDAR readings."""
        rgb_panel = self._rgb_panel(camera_reading.get("rgb"))
        depth_panel = self._depth_panel(camera_reading.get("depth"))
        lidar_panel = cv2.resize(
            self._lidar_panel(lidar_reading["points"]), (self._panel_h, self._panel_h)
        )
        combined_bgr = np.hstack([rgb_panel, depth_panel, lidar_panel])
        combined_rgba = cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGBA)
        height, width = combined_rgba.shape[:2]
        # list(bytes) matches the confirmed-working usage in
        # isaacsim.replicator.mobility_gen.ui (set_bytes_data's binding is untyped,
        # so following the one real example rather than guessing raw bytes/ndarray support).
        self._image_provider.set_bytes_data(list(combined_rgba.tobytes()), [width, height])

    def close(self) -> None:
        self._window.visible = False
