"""Headless synthetic test for keyboard_controller.py's held-key command logic.

Boots a minimal (headless) SimulationApp so `carb` is fully initialized, then
feeds synthetic press/release events directly into KeyboardController's event
handler -- no real X11/window keypress needed. Validates: arrow-key/ZC
accumulation, multi-key holds, Space-to-stop, and Escape-to-exit.
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import carb

from controllers.keyboard_controller import KeyboardController


class FakeInput:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeEvent:
    def __init__(self, event_type, name: str) -> None:
        self.type = event_type
        self.input = FakeInput(name)


class FakeCharEvent:
    """Mirrors the real carb CHAR event shape: .input is a plain str, no .name."""

    def __init__(self, char: str) -> None:
        self.type = carb.input.KeyboardEventType.CHAR
        self.input = char


def press(key: str) -> FakeEvent:
    return FakeEvent(carb.input.KeyboardEventType.KEY_PRESS, key)


def release(key: str) -> FakeEvent:
    return FakeEvent(carb.input.KeyboardEventType.KEY_RELEASE, key)


def assert_command(kb: KeyboardController, expected: list, label: str) -> None:
    got = kb.update(0.0).tolist()
    ok = all(abs(g - e) < 1e-6 for g, e in zip(got, expected))
    status = "PASS" if ok else "FAIL"
    print(f">>> [test] {status}: {label} -> got={got} expected={expected}", flush=True)
    if not ok:
        raise AssertionError(f"{label}: got={got} expected={expected}")


def main() -> None:
    kb = KeyboardController(command_device="cpu")

    # Real carb keyboard subscriptions also deliver CHAR events (event.input is a plain
    # str, not an enum with .name) alongside PRESS/RELEASE for printable keys. This must
    # not crash -- regression test for a real bug hit in windowed/live use.
    kb._on_keyboard_event(FakeCharEvent("w"))
    print(">>> [test] PASS: CHAR event (plain-str .input) does not crash", flush=True)

    kb._on_keyboard_event(press("UP"))
    assert_command(kb, [1.0, 0.0, 0.0], "UP held -> forward")

    kb._on_keyboard_event(press("RIGHT"))
    assert_command(kb, [1.0, 0.0, -1.0], "UP+RIGHT held -> forward + turn right")

    kb._on_keyboard_event(release("UP"))
    assert_command(kb, [0.0, 0.0, -1.0], "release UP, RIGHT still held -> turn right only")

    kb._on_keyboard_event(press("Z"))
    assert_command(kb, [0.0, 1.0, -1.0], "RIGHT+Z held -> turn right + strafe left")

    kb._on_keyboard_event(press("SPACE"))
    assert_command(kb, [0.0, 0.0, 0.0], "Space -> stop clears all held keys")

    # Repeated press events (OS auto-repeat) must not double-accumulate.
    kb._on_keyboard_event(press("DOWN"))
    kb._on_keyboard_event(press("DOWN"))
    kb._on_keyboard_event(press("DOWN"))
    assert_command(kb, [-1.0, 0.0, 0.0], "repeated DOWN presses -> single backward command, no double count")

    kb._on_keyboard_event(release("DOWN"))
    assert_command(kb, [0.0, 0.0, 0.0], "release DOWN -> zero command")

    assert not kb.exit_requested, "exit_requested should be False before Escape"
    kb._on_keyboard_event(press("ESCAPE"))
    assert kb.exit_requested, "Escape press should set exit_requested"
    print(">>> [test] PASS: Escape sets exit_requested", flush=True)

    print(">>> [test] ALL KEYBOARD CONTROLLER TESTS PASSED", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        simulation_app.close()
        raise
