"""Optional progress events emitted by completed work, without UI dependencies."""
from collections.abc import Callable


ProgressCallback = Callable[[float, str], None]


def emit_progress(callback: ProgressCallback | None, fraction: float, message: str) -> None:
    if callback is not None:
        callback(fraction, message)
