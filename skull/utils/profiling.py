from __future__ import annotations

import time
from contextlib import ContextDecorator


class Timer(ContextDecorator):
    def __init__(self):
        self.start_time = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.start_time
        return False
