"""
xlmtec.utils.crash_report
~~~~~~~~~~~~~~~~~~~~~~~~~~
Write and read human-readable crash report files.

Crash files live at:
    ~/.xlmtec/logs/crash_<YYYYMMDD_HHMMSS>.txt

A crash file is intentionally human-readable so a user can attach it
to a GitHub issue without any processing. It contains:
  - xlmtec version, Python, OS, GPU
  - The CLI command that was running
  - The exception type and message
  - The full Python traceback
  - The last 10 events from the session log

Usage
-----
    from xlmtec.utils.crash_report import CrashReporter

    # Write on unhandled exception:
    CrashReporter.write(session, exc)

    # Find latest crash file:
    path = CrashReporter.latest()      # Path | None

    # List recent:
    files = CrashReporter.list_recent(5)  # list[CrashFile]
"""

from __future__ import annotations

import platform
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

_LOG_DIR = Path.home() / ".xlmtec" / "logs"
_MAX_CRASH_FILES = 20


@dataclass
class CrashFile:
    path: Path
    timestamp: str       # YYYYMMDD_HHMMSS from filename
    cmd: str             # command that crashed


class CrashReporter:
    """Static utility for writing and locating crash reports."""

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    @staticmethod
    def write(
        session: Optional[object],   # utils.telemetry.Session (typed loosely to avoid circular)
        exc: BaseException,
        log_dir: Path = _LOG_DIR,
    ) -> Path:
        """Write a crash report and return its path.

        Never raises — crash reporting must not cause a second crash.
        """
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = log_dir / f"crash_{ts}.txt"
            content = CrashReporter._format(session, exc)
            path.write_text(content, encoding="utf-8")
            CrashReporter._rotate(log_dir)
            return path
        except Exception:
            # Absolute last resort
            fallback = Path.home() / f"xlmtec_crash_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            try:
                fallback.write_text(traceback.format_exc(), encoding="utf-8")
            except Exception:
                pass
            return fallback

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @staticmethod
    def latest(log_dir: Path = _LOG_DIR) -> Optional[Path]:
        """Return the path to the most recent crash file, or None."""
        files = sorted(log_dir.glob("crash_*.txt"))
        return files[-1] if files else None

    @staticmethod
    def list_recent(n: int = 10, log_dir: Path = _LOG_DIR) -> List[CrashFile]:
        """Return the n most recent crash files, newest first."""
        files = sorted(log_dir.glob("crash_*.txt"), reverse=True)[:n]
        result = []
        for p in files:
            parts = p.stem.split("_", 1)           # ["crash", "20260311_142201"]
            ts = parts[1] if len(parts) == 2 else "unknown"
            # Try to extract cmd from matching session file
            cmd = CrashReporter._cmd_from_session(p.parent, ts)
            result.append(CrashFile(path=p, timestamp=ts, cmd=cmd))
        return result

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format(session: Optional[object], exc: BaseException) -> str:
        from xlmtec.utils.telemetry import _xlmtec_version, _gpu_info

        tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
        tb_str = "".join(tb_lines).strip()

        # Pull last 10 events from session
        events_section = ""
        if session is not None:
            try:
                events = list(getattr(session, "events", []))[-10:]
                if events:
                    import json
                    lines = []
                    for ev in events:
                        ts = ev.get("ts", "")[-12:]   # just time portion
                        name = ev.get("event", "?")
                        extra = {k: v for k, v in ev.items()
                                 if k not in ("ts", "session", "event")}
                        line = f"  [{ts}]  {name}"
                        if extra:
                            line += "  " + "  ".join(f"{k}={v}" for k, v in extra.items())
                        lines.append(line)
                    events_section = "\n\nLast events\n-----------\n" + "\n".join(lines)
            except Exception:
                pass

        cmd = "unknown"
        session_id = "unknown"
        if session is not None:
            cmd = getattr(session, "cmd", "unknown")
            session_id = getattr(session, "session_id", "unknown")

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sep = "=" * 60

        return (
            f"xlmtec crash report — {now}\n"
            f"{sep}\n\n"
            f"Session ID      : {session_id}\n"
            f"xlmtec version  : {_xlmtec_version()}\n"
            f"Python          : {sys.version.split()[0]} ({sys.platform})\n"
            f"OS              : {platform.platform(terse=True)}\n"
            f"GPU             : {_gpu_info()}\n"
            f"Command         : xlmtec {cmd}\n\n"
            f"Error\n"
            f"-----\n"
            f"{type(exc).__name__}: {exc}\n\n"
            f"Traceback\n"
            f"---------\n"
            f"{tb_str}"
            f"{events_section}\n\n"
            f"{sep}\n"
            f"Attach this file to your GitHub issue:\n"
            f"https://github.com/Abdur-azure/xlmtec/issues/new?template=bug_report.yml\n"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _rotate(log_dir: Path) -> None:
        files = sorted(log_dir.glob("crash_*.txt"))
        for old in files[:-_MAX_CRASH_FILES]:
            try:
                old.unlink(missing_ok=True)
            except Exception:
                pass

    @staticmethod
    def _cmd_from_session(log_dir: Path, ts_prefix: str) -> str:
        """Try to infer cmd from a session file with a matching timestamp."""
        # ts_prefix like "20260311_142201" — session files are "session_20260311_142201_<cmd>.jsonl"
        try:
            date = ts_prefix[:8]
            matches = list(log_dir.glob(f"session_{date}_*.jsonl"))
            if matches:
                # Parse first event to find cmd
                import json
                first_line = matches[-1].read_text(encoding="utf-8").splitlines()[0]
                ev = json.loads(first_line)
                return str(ev.get("cmd", "unknown"))
        except Exception:
            pass
        return "unknown"