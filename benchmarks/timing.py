"""Sequential request-stage timing using one local monotonic clock."""

from contextlib import contextmanager
from time import perf_counter_ns


def nonnegative_integer(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def nanoseconds(value, label):
    return nonnegative_integer(value, f"{label} (nanoseconds)")


def identifier(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def validate_trace(trace):
    """Reject overlapping/out-of-order stages rather than double-count them."""
    identifier(trace.get("request_id"), "trace.request_id")
    duration = nanoseconds(trace.get("duration_ns"), "trace.duration_ns")
    stages = trace.get("stages")
    if not isinstance(stages, list):
        raise ValueError("trace.stages must be a list")
    previous_end = 0
    for stage in stages:
        if not isinstance(stage, dict):
            raise ValueError("each trace stage must be an object")
        identifier(stage.get("name"), "stage.name")
        start = nanoseconds(stage.get("start_ns"), "stage.start_ns")
        end = nanoseconds(stage.get("end_ns"), "stage.end_ns")
        if not previous_end <= start <= end <= duration:
            raise ValueError("stages must be ordered, nonoverlapping and inside the trace")
        previous_end = end


class Trace:
    """Time one sequential path. This object is not shared between requests/tasks.

    Offsets are relative to this trace's origin, not client timestamps. Nested
    stages are rejected. Exceptions propagate while the failed stage is retained.
    """

    def __init__(self, request_id, *, clock=perf_counter_ns):
        self.request_id = identifier(request_id, "request_id")
        self._clock = clock
        self._origin = nanoseconds(clock(), "clock")
        self._last = 0
        self._active = False
        self._closed = False
        self._stages = []

    def _offset(self):
        offset = nanoseconds(self._clock(), "clock") - self._origin
        if offset < self._last:
            raise ValueError("clock moved backwards")
        self._last = offset
        return offset

    @contextmanager
    def stage(self, name):
        identifier(name, "stage name")
        if self._closed or self._active:
            raise ValueError("stages require an open trace with no active stage")
        start = self._offset()
        self._active = True
        try:
            yield
        finally:
            try:
                end = self._offset()
                self._stages.append({"name": name, "start_ns": start, "end_ns": end})
            finally:
                self._active = False

    def finish(self):
        if self._closed or self._active:
            raise ValueError("finish requires an open trace with no active stage")
        trace = {
            "request_id": self.request_id,
            "duration_ns": self._offset(),
            "stages": [dict(stage) for stage in self._stages],
        }
        validate_trace(trace)
        self._closed = True
        return trace
