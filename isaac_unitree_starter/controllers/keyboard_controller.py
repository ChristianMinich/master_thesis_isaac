"""Continuous arrow-key(+ZC) keyboard control for the robot's base velocity command.

Tracks the set of currently-held mapped keys (press adds, release removes)
rather than accumulating +=/-= deltas, so a missed release event (e.g. from a
focus change) can't leave a stale velocity baked into the command, and OS key
auto-repeat can't double-count a press.

Deliberately not WASD: Isaac Sim's viewport toolbar (omni.kit.widget.toolbar)
globally binds W/E/R/Q -> Move/Rotate/Scale/Select TOOL switching and T ->
toggle selection mode, independent of any app-level keyboard subscription and
with no focus/hover guard -- those keys never reach this controller at all.
Arrow keys + Z/C/L/Space/Escape avoid every one of those reserved letters
(confirmed against omni.kit.widget.toolbar's hotkey registrations).
"""

from __future__ import annotations

import carb
import omni.appwindow
from isaacsim.core.deprecation_manager import import_module

LINEAR_SPEED = 1.0  # m/s
STRAFE_SPEED = 1.0  # m/s
YAW_SPEED = 1.0  # rad/s

# [v_x, v_y, w_z] contribution while each key is held. Values are additive and
# applied continuously via press/release tracking.
KEY_COMMAND_MAP = {
    "UP": [LINEAR_SPEED, 0.0, 0.0],
    "DOWN": [-LINEAR_SPEED, 0.0, 0.0],
    "LEFT": [0.0, 0.0, YAW_SPEED],
    "RIGHT": [0.0, 0.0, -YAW_SPEED],
    "Z": [0.0, STRAFE_SPEED, 0.0],
    "C": [0.0, -STRAFE_SPEED, 0.0],
}
RECORD_TOGGLE_KEY = "L"
DEFAULT_COMMAND_DEVICE = "cuda"


class KeyboardController:
    """Subscribes to keyboard events and exposes the current velocity command."""

    def __init__(self, command_device: str = DEFAULT_COMMAND_DEVICE, on_toggle_recording=None) -> None:
        self._device = command_device
        self._on_toggle_recording = on_toggle_recording
        self._held_keys: set[str] = set()
        self.exit_requested = False

        self._appwindow = None
        self._input = None
        self._keyboard = None
        self._sub = None

    def start(self) -> None:
        """Subscribe to keyboard events. Call once after the viewport/app window exists."""
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)

    def stop(self) -> None:
        """Unsubscribe from keyboard events."""
        if self._sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
            self._sub = None

    def _on_keyboard_event(self, event, *_args, **_kwargs) -> bool:
        # KeyboardEventType also includes KEY_REPEAT and CHAR (fired for regular text
        # input); for CHAR, event.input is a plain string with no .name attribute. Only
        # PRESS/RELEASE carry the KeyboardInput enum our held-key logic needs.
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            name = event.input.name
            if name == "ESCAPE":
                self.exit_requested = True
            elif name == "SPACE":
                self._held_keys.clear()
            elif name == RECORD_TOGGLE_KEY:
                if self._on_toggle_recording is not None:
                    self._on_toggle_recording()
            elif name in KEY_COMMAND_MAP:
                self._held_keys.add(name)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._held_keys.discard(event.input.name)
        return True

    def update(self, dt: float = 0.0):
        """Return the current [v_x, v_y, w_z] command tensor from currently-held keys.

        `dt` is accepted (and unused) so this satisfies the same CommandSource
        interface as RandomWalkController -- callers can call `.update(dt)`
        polymorphically without branching on which driver is active.
        """
        torch = import_module("torch")
        v_x = v_y = w_z = 0.0
        for key in self._held_keys:
            dx, dy, dw = KEY_COMMAND_MAP[key]
            v_x += dx
            v_y += dy
            w_z += dw
        return torch.tensor([v_x, v_y, w_z], device=self._device)
