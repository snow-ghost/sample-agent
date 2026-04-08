from __future__ import annotations

from typing import Callable, Iterable


HandlerFn = Callable[[], bool]


def run_fastpath_handlers(
    handlers: Iterable[HandlerFn],
) -> bool:
    for handler in handlers:
        if handler():
            return True
    return False
