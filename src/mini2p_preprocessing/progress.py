"""Progress lines with a time estimate for long loops (the line itself is pure)."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime, timedelta

PROGRESS_EVERY_S = 30.0  # long loops report at most this often, plus first and last


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def progress_line(done: int, total: int, durations_s: list[float], now: datetime) -> str:
    """'[done/total] last …, about … left, done around HH:MM Day', from the mean step time.

    ``durations_s`` are the steps that count towards the estimate (not failed ones).
    The finish time is local clock time, including any clock change before then.
    """
    line = f"[{done}/{total}]"
    if not durations_s:
        return line
    line += f" last {format_duration(durations_s[-1])}"
    remaining = total - done
    if remaining <= 0:
        return f"{line}, all done"
    left = remaining * sum(durations_s) / len(durations_s)
    finish = (now + timedelta(seconds=left)).astimezone()  # local offset at that moment
    return f"{line}, about {format_duration(left)} left, done around {finish:%H:%M %a}"


class ProgressLog:
    """Times each step of a loop and reports a progress line with a time estimate.

    Reports the first and last steps, and otherwise at most once per ``min_interval_s``.
    """

    def __init__(
        self,
        label: str,
        total: int,
        report: Callable[[str], None],
        min_interval_s: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], datetime] = lambda: datetime.now().astimezone(),
    ):
        self.label, self.total, self.report = label, total, report
        self.min_interval_s, self.clock, self.wall = min_interval_s, clock, wall
        self.durations: list[float] = []
        self.done = 0
        self.last_tick = self.last_report = clock()

    def step_done(self, succeeded: bool = True) -> None:
        """Count a step; a failed one is left out of the time per step."""
        now = self.clock()
        if succeeded:
            self.durations.append(now - self.last_tick)
        self.last_tick = now
        self.done += 1
        done = self.done
        due = now - self.last_report >= self.min_interval_s
        if done == 1 or done >= self.total or due:
            self.report(
                f"{self.label}: {progress_line(done, self.total, self.durations, self.wall())}"
            )
            self.last_report = now
