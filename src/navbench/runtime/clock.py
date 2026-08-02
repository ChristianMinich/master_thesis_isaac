"""Common simulation clock and multi-rate triggers.

Specification reference: section 7 ("Recommended timing and synchronization").

Every recorded sample carries a ``timestamp_ns`` taken from **one** clock.
Sensor-specific timestamps are never reconstructed from file order, so the
clock is the single authority for time in the whole pipeline.

Rates used by the pipeline (spec section 7):

======================  ========  ============================================
Process                 Rate      Note
======================  ========  ============================================
Physics simulation      200 Hz    integration step, not fully recorded
Low-level locomotion     50 Hz    frozen Go2 velocity-tracking controller
Proprioception           50 Hz    joint/base/contact state
IMU                     100 Hz    raw measurement + bias/noise parameters
High-level decision       5 Hz    requested/executed action
RGB, depth, LiDAR        10 Hz    synchronized observation frames
Annotations              10 Hz    aligned with camera frames
Inspection burst         30 Hz    2-5 s around each inspection trigger
======================  ========  ============================================

All rates are integer divisors of the 200 Hz physics rate, therefore every
sample lands exactly on a physics tick and integer nanosecond arithmetic is
exact (no floating point drift over an episode).
"""

from __future__ import annotations

from dataclasses import dataclass, field

NS_PER_S: int = 1_000_000_000
"""Nanoseconds per second. Timestamps are integers to avoid float drift."""


@dataclass(frozen=True)
class RateSpec:
    """A named process rate that must divide the physics rate exactly."""

    name: str
    hz: int

    def period_ns(self, physics_hz: int) -> int:
        """Return the sample period in nanoseconds.

        Raises:
            ValueError: if ``hz`` does not divide ``physics_hz`` exactly, which
                would make the process unsynchronizable with the physics tick.
        """
        if self.hz <= 0:
            raise ValueError(f"rate '{self.name}' must be positive, got {self.hz}")
        if physics_hz % self.hz != 0:
            raise ValueError(
                f"rate '{self.name}' ({self.hz} Hz) must divide the physics rate "
                f"({physics_hz} Hz) exactly"
            )
        return NS_PER_S // self.hz


class SimClock:
    """Monotonic integer-nanosecond simulation clock driven by physics ticks.

    The clock starts at ``start_ns`` and advances by exactly one physics period
    per :meth:`tick`. It never derives time from wall-clock time, so an episode
    is bit-reproducible for a fixed seed.

    Args:
        physics_hz: Physics integration rate in Hz (spec: 200 Hz).
        start_ns: Episode start timestamp on the common clock.
    """

    def __init__(self, physics_hz: int = 200, start_ns: int = 0) -> None:
        if physics_hz <= 0:
            raise ValueError("physics_hz must be positive")
        if NS_PER_S % physics_hz != 0:
            raise ValueError("physics_hz must divide 1e9 ns exactly")
        self._physics_hz = physics_hz
        self._dt_ns = NS_PER_S // physics_hz
        self._start_ns = int(start_ns)
        self._step = 0

    @property
    def physics_hz(self) -> int:
        """Physics integration rate in Hz."""
        return self._physics_hz

    @property
    def dt_ns(self) -> int:
        """Physics period in nanoseconds."""
        return self._dt_ns

    @property
    def dt(self) -> float:
        """Physics period in seconds."""
        return self._dt_ns / NS_PER_S

    @property
    def start_ns(self) -> int:
        """Episode start timestamp (``episode_start_ns`` in the spec)."""
        return self._start_ns

    @property
    def step_index(self) -> int:
        """Number of physics steps executed since reset."""
        return self._step

    @property
    def now_ns(self) -> int:
        """Current timestamp on the common simulation clock."""
        return self._start_ns + self._step * self._dt_ns

    @property
    def elapsed_s(self) -> float:
        """Seconds elapsed since the episode start."""
        return (self._step * self._dt_ns) / NS_PER_S

    def tick(self) -> int:
        """Advance the clock by one physics step and return the new timestamp."""
        self._step += 1
        return self.now_ns

    def reset(self, start_ns: int | None = None) -> None:
        """Reset the step counter and optionally move the episode start."""
        self._step = 0
        if start_ns is not None:
            self._start_ns = int(start_ns)


@dataclass
class RateTrigger:
    """Fires on physics steps that belong to a slower process rate.

    A trigger fires on step 0 and then every ``divisor`` steps, so the first
    observation frame is always aligned with ``episode_start_ns``.

    Args:
        name: Process name, used in error messages and logs.
        hz: Process rate in Hz.
        physics_hz: Physics rate the trigger is derived from.
    """

    name: str
    hz: int
    physics_hz: int = 200
    _divisor: int = field(init=False, repr=False)
    _count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        spec = RateSpec(self.name, self.hz)
        spec.period_ns(self.physics_hz)  # validates divisibility
        self._divisor = self.physics_hz // self.hz

    @property
    def divisor(self) -> int:
        """Number of physics steps between two samples of this process."""
        return self._divisor

    @property
    def period_ns(self) -> int:
        """Sample period in nanoseconds."""
        return NS_PER_S // self.hz

    @property
    def period_s(self) -> float:
        """Sample period in seconds."""
        return 1.0 / self.hz

    @property
    def count(self) -> int:
        """Number of samples emitted so far (the next ``frame_index``)."""
        return self._count

    def due(self, step_index: int) -> bool:
        """Return whether ``step_index`` is a sampling step for this process."""
        return step_index % self._divisor == 0

    def fire(self, step_index: int) -> bool:
        """Return :meth:`due` and increment the emitted-sample counter if due."""
        if self.due(step_index):
            self._count += 1
            return True
        return False

    def reset(self) -> None:
        """Reset the emitted-sample counter."""
        self._count = 0


def frames_for_duration(hz: int, duration_s: float) -> int:
    """Return the number of samples of a process at ``hz`` over ``duration_s``.

    Sample 0 is emitted at t=0, hence the ``+ 1``.
    """
    return int(round(duration_s * hz)) + 1