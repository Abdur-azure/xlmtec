"""
xlmtec.utils.telemetry
~~~~~~~~~~~~~~~~~~~~~~~
Structured JSON-Lines session logging for app insights.

Every xlmtec invocation writes a session file:
    ~/.xlmtec/logs/session_<YYYYMMDD_HHMMSS>_<cmd>.jsonl

Each line is one JSON event. The full session file can be attached to a
GitHub issue as a crash report. No network calls are ever made. All data
stays on disk in ~/.xlmtec/logs/.

Privacy rules (enforced in log_invocation):
  - Path values are reduced to basename only
  - Keys containing "key", "token", "secret", "password", "api" are redacted
  - No model weight values, dataset contents, or config file contents are logged

Usage
-----
AppLogger is a module-level singleton — one instance per process.

    from xlmtec.utils.telemetry import AppLogger, track

    # In cli/main.py @app.callback():
    AppLogger.start(cmd="train")

    # Log CLI flags:
    AppLogger.log_invocation("train", {"method": "lora", "epochs": 3})

    # Log any error (called automatically from FineTuneError.__init__):
    AppLogger.log_error(exc)

    # Time and log any function with a decorator:
    @track("data.load")
    def load(self, path): ...

    # At exit:
    AppLogger.finalize(exit_code=0)
"""

from __future__ import annotations

import functools
import json
import os
import platform
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOG_DIR = Path.home() / ".xlmtec" / "logs"
_MAX_LOG_FILES = 50          # rotate oldest when exceeded
_SENSITIVE_KEY_FRAGMENTS = {"key", "token", "secret", "password", "api", "auth"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _short_id() -> str:
    return uuid.uuid4().hex[:8]


def _safe_basename(value: Any) -> Any:
    """Reduce a path string to its basename. Leaves non-strings untouched."""
    if isinstance(value, str) and (os.sep in value or "/" in value):
        return Path(value).name
    if isinstance(value, Path):
        return value.name
    return value


def _sanitize_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Strip sensitive keys and reduce paths to basenames."""
    out: Dict[str, Any] = {}
    for k, v in args.items():
        k_lower = k.lower()
        if any(frag in k_lower for frag in _SENSITIVE_KEY_FRAGMENTS):
            out[k] = "<redacted>"
        else:
            out[k] = _safe_basename(v)
    return out


def _xlmtec_version() -> str:
    try:
        from importlib.metadata import version
        return version("xlmtec")
    except Exception:
        return "unknown"


def _gpu_info() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return f"{name} ({vram:.1f} GB VRAM)"
        return "CPU only"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class Session:
    """Represents one CLI invocation. Owns the JSONL log file handle."""

    def __init__(self, cmd: str, log_dir: Path = _LOG_DIR) -> None:
        self.session_id: str = _short_id()
        self.cmd: str = cmd or "unknown"
        self.start_time: float = time.monotonic()
        self.start_wall: str = _now_iso()
        self._log_dir = log_dir
        self._file: Optional[Any] = None
        self._events: List[Dict[str, Any]] = []   # in-memory buffer for tests
        self._path: Optional[Path] = None

    # ------------------------------------------------------------------

    def open(self) -> "Session":
        """Create the log directory and open the session file."""
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_cmd = self.cmd.replace(" ", "_")[:20]
            filename = f"session_{ts}_{safe_cmd}.jsonl"
            self._path = self._log_dir / filename
            self._file = self._path.open("a", encoding="utf-8", buffering=1)
            self._rotate()
        except Exception:
            # Telemetry must never crash the app
            pass
        return self

    def write(self, event: Dict[str, Any]) -> None:
        """Write one event to the JSONL file and keep an in-memory copy."""
        event.setdefault("ts", _now_iso())
        event.setdefault("session", self.session_id)
        self._events.append(event)
        if self._file:
            try:
                self._file.write(json.dumps(event, default=str) + "\n")
            except Exception:
                pass

    def close(self) -> None:
        if self._file:
            try:
                self._file.flush()
                self._file.close()
            except Exception:
                pass
            self._file = None

    @property
    def path(self) -> Optional[Path]:
        return self._path

    @property
    def events(self) -> List[Dict[str, Any]]:
        return list(self._events)

    # ------------------------------------------------------------------

    def _rotate(self) -> None:
        """Delete oldest log files if we exceed _MAX_LOG_FILES."""
        try:
            files = sorted(self._log_dir.glob("session_*.jsonl"))
            for old in files[:-_MAX_LOG_FILES]:
                old.unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# AppLogger — module-level singleton
# ---------------------------------------------------------------------------


class _AppLogger:
    """Module-level singleton. One session per process."""

    def __init__(self) -> None:
        self._session: Optional[Session] = None

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start(self, cmd: str = "unknown", log_dir: Path = _LOG_DIR) -> Session:
        """Create and open a new session. Call from cli/main.py @app.callback()."""
        self._session = Session(cmd=cmd, log_dir=log_dir).open()
        self._session.write({
            "event": "session.start",
            "cmd": cmd,
            "xlmtec_version": _xlmtec_version(),
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "os": platform.platform(terse=True),
            "gpu": _gpu_info(),
        })
        return self._session

    def finalize(self, exit_code: int = 0) -> None:
        """Write session.end and close the file. Call at process exit."""
        if not self._session:
            return
        elapsed = round(time.monotonic() - self._session.start_time, 3)
        self._session.write({
            "event": "session.end",
            "exit_code": exit_code,
            "elapsed_s": elapsed,
        })
        self._session.close()

    # ------------------------------------------------------------------
    # Event logging
    # ------------------------------------------------------------------

    def log(self, event: str, **kwargs: Any) -> None:
        """Write a single event. Safe to call even before start()."""
        if not self._session:
            return
        self._session.write({"event": event, **kwargs})

    def log_invocation(self, cmd: str, args: Dict[str, Any]) -> None:
        """Log CLI flags — strips paths to basename and redacts sensitive keys."""
        self.log("cli.invocation", cmd=cmd, args=_sanitize_args(args))

    def log_error(self, exc: BaseException) -> None:
        """Log an exception with its traceback. Called from FineTuneError.__init__."""
        if not self._session:
            return
        tb = traceback.format_exc()
        self._session.write({
            "event": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": tb,
        })

    def log_stage(self, stage: str, **kwargs: Any) -> None:
        """Convenience: log a named pipeline stage event."""
        self.log(f"stage.{stage}", **kwargs)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def session(self) -> Optional[Session]:
        return self._session

    @property
    def active(self) -> bool:
        return self._session is not None


# Module-level singleton
AppLogger = _AppLogger()


# ---------------------------------------------------------------------------
# @track decorator
# ---------------------------------------------------------------------------


def track(event: str) -> Callable[[F], F]:
    """Decorator that logs start, completion (with elapsed), and failures.

    Usage:
        @track("trainer.train")
        def train(self, dataset): ...

    Emits:
        {"event": "trainer.train.start"}
        {"event": "trainer.train.complete", "elapsed_s": 14.2}
        — or on exception —
        {"event": "trainer.train.failed", "error": "...", "elapsed_s": 0.3}
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            AppLogger.log(f"{event}.start")
            t0 = time.monotonic()
            try:
                result = fn(*args, **kwargs)
                AppLogger.log(
                    f"{event}.complete",
                    elapsed_s=round(time.monotonic() - t0, 3),
                )
                return result
            except Exception as exc:
                AppLogger.log(
                    f"{event}.failed",
                    error=str(exc),
                    error_type=type(exc).__name__,
                    elapsed_s=round(time.monotonic() - t0, 3),
                )
                raise
        return wrapper  # type: ignore[return-value]
    return decorator