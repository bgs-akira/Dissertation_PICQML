"""Small CLI/IO helpers shared by ``scripts/`` entry points.

Currently houses one thing: ``setup_logging``, a stdout/stderr tee so every
runner and sweep script writes a sidecar log file next to its JSON output
without each script reinventing the wiring.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import IO


class _Tee:
    """Forward writes to multiple file-like objects.

    Used to mirror stdout/stderr into a log file while keeping the
    console output live. Each ``write`` flushes immediately so a tailing
    ``Get-Content -Wait`` (or ``tail -f``) sees output in real time.
    """

    def __init__(self, *streams: IO[str]) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        n = 0
        for s in self._streams:
            n = s.write(data)
            s.flush()
        return n

    def flush(self) -> None:
        for s in self._streams:
            s.flush()

    def isatty(self) -> bool:
        return False


def setup_logging(log_path: Path) -> None:
    """Tee ``sys.stdout`` and ``sys.stderr`` into ``log_path``.

    Idempotent across calls: the file is truncated and a fresh tee is
    installed each time. The original ``sys.__stdout__`` and
    ``sys.__stderr__`` are preserved as the first leg of the tee, so
    the console keeps seeing live output.

    The parent directory is created if missing.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
