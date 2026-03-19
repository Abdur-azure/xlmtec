"""
tests/test_telemetry.py
~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for xlmtec.utils.telemetry and xlmtec.utils.crash_report.
All tests use tmp_path — nothing is written to ~/.xlmtec.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from xlmtec.utils.telemetry import (
    Session,
    _AppLogger,
    _safe_basename,
    _sanitize_args,
    track,
)

# ============================================================================
# Helpers
# ============================================================================


def _fresh_logger(tmp_path: Path) -> _AppLogger:
    """Return a brand-new _AppLogger instance (not the module singleton)."""
    return _AppLogger()


def _start_session(logger: _AppLogger, tmp_path: Path, cmd: str = "train") -> Session:
    return logger.start(cmd=cmd, log_dir=tmp_path)


def _read_events(session: Session) -> list[dict]:
    session.close()
    if session.path and session.path.exists():
        lines = session.path.read_text(encoding="utf-8").splitlines()
        return [json.loads(l) for l in lines if l.strip()]
    return []


# ============================================================================
# _safe_basename
# ============================================================================


class TestSafeBasename:
    def test_strips_unix_path(self):
        assert _safe_basename("/home/user/data/train.jsonl") == "train.jsonl"

    def test_strips_windows_path(self):
        assert _safe_basename("C:\\Users\\user\\data\\train.jsonl") == "train.jsonl"

    def test_leaves_plain_string(self):
        assert _safe_basename("lora") == "lora"

    def test_leaves_int(self):
        assert _safe_basename(3) == 3

    def test_strips_pathlib_path(self):
        assert _safe_basename(Path("/tmp/data.jsonl")) == "data.jsonl"


# ============================================================================
# _sanitize_args
# ============================================================================


class TestSanitizeArgs:
    def test_redacts_api_key(self):
        result = _sanitize_args({"api_key": "sk-abc123"})
        assert result["api_key"] == "<redacted>"

    def test_redacts_token(self):
        result = _sanitize_args({"hf_token": "hf_xyz"})
        assert result["hf_token"] == "<redacted>"

    def test_redacts_secret(self):
        result = _sanitize_args({"webhook_secret": "abc"})
        assert result["webhook_secret"] == "<redacted>"

    def test_keeps_safe_keys(self):
        result = _sanitize_args({"method": "lora", "epochs": 3})
        assert result["method"] == "lora"
        assert result["epochs"] == 3

    def test_strips_path_values(self):
        result = _sanitize_args({"dataset": "/home/user/data/train.jsonl"})
        assert result["dataset"] == "train.jsonl"


# ============================================================================
# Session
# ============================================================================


class TestSession:
    def test_open_creates_file(self, tmp_path):
        s = Session(cmd="train", log_dir=tmp_path).open()
        assert s.path is not None
        assert s.path.exists()
        s.close()

    def test_write_produces_valid_json_lines(self, tmp_path):
        s = Session(cmd="train", log_dir=tmp_path).open()
        s.write({"event": "test.event", "value": 42})
        s.close()
        lines = s.path.read_text(encoding="utf-8").splitlines()
        events = [json.loads(l) for l in lines]
        assert any(e["event"] == "test.event" for e in events)

    def test_write_adds_ts_and_session_id(self, tmp_path):
        s = Session(cmd="train", log_dir=tmp_path).open()
        s.write({"event": "x"})
        s.close()
        ev = json.loads(s.path.read_text().splitlines()[0])
        assert "ts" in ev
        assert "session" in ev

    def test_events_in_memory(self, tmp_path):
        s = Session(cmd="train", log_dir=tmp_path).open()
        s.write({"event": "a"})
        s.write({"event": "b"})
        s.close()
        assert len(s.events) == 2

    def test_write_does_not_raise_if_file_closed(self, tmp_path):
        s = Session(cmd="train", log_dir=tmp_path).open()
        s.close()
        s.write({"event": "late"})  # should not raise


# ============================================================================
# _AppLogger
# ============================================================================


class TestAppLogger:
    def test_start_creates_session(self, tmp_path):
        logger = _fresh_logger(tmp_path)
        session = _start_session(logger, tmp_path)
        assert logger.active
        assert logger.session is not None
        session.close()

    def test_start_writes_session_start_event(self, tmp_path):
        logger = _fresh_logger(tmp_path)
        session = _start_session(logger, tmp_path)
        events = _read_events(session)
        assert any(e["event"] == "session.start" for e in events)

    def test_session_start_includes_version_and_python(self, tmp_path):
        logger = _fresh_logger(tmp_path)
        session = _start_session(logger, tmp_path)
        events = _read_events(session)
        start = next(e for e in events if e["event"] == "session.start")
        assert "xlmtec_version" in start
        assert "python" in start

    def test_log_writes_event(self, tmp_path):
        logger = _fresh_logger(tmp_path)
        session = _start_session(logger, tmp_path)
        logger.log("stage.load", source="local_file")
        events = _read_events(session)
        assert any(e["event"] == "stage.load" for e in events)

    def test_log_invocation_strips_sensitive_keys(self, tmp_path):
        logger = _fresh_logger(tmp_path)
        session = _start_session(logger, tmp_path)
        logger.log_invocation("train", {"method": "lora", "api_key": "sk-secret"})
        events = _read_events(session)
        inv = next(e for e in events if e["event"] == "cli.invocation")
        assert inv["args"]["api_key"] == "<redacted>"
        assert inv["args"]["method"] == "lora"

    def test_log_error_writes_error_event(self, tmp_path):
        logger = _fresh_logger(tmp_path)
        session = _start_session(logger, tmp_path)
        try:
            raise ValueError("bad config")
        except ValueError as exc:
            logger.log_error(exc)
        events = _read_events(session)
        err = next(e for e in events if e["event"] == "error")
        assert err["error_type"] == "ValueError"
        assert "bad config" in err["message"]

    def test_finalize_writes_session_end(self, tmp_path):
        logger = _fresh_logger(tmp_path)
        session = _start_session(logger, tmp_path)
        logger.finalize(exit_code=0)
        events = [json.loads(l) for l in session.path.read_text().splitlines() if l]
        assert any(e["event"] == "session.end" for e in events)
        end = next(e for e in events if e["event"] == "session.end")
        assert end["exit_code"] == 0
        assert "elapsed_s" in end

    def test_log_before_start_does_not_raise(self):
        logger = _AppLogger()  # no start() called
        logger.log("orphan.event")  # should not raise
        logger.log_error(ValueError("x"))


# ============================================================================
# @track decorator
# ============================================================================


class TestTrackDecorator:
    def test_complete_event_written(self, tmp_path):
        logger = _fresh_logger(tmp_path)
        session = _start_session(logger, tmp_path)

        @track("trainer.train")
        def dummy_train():
            return 42

        with patch("xlmtec.utils.telemetry.AppLogger", logger):
            result = dummy_train()

        assert result == 42
        events = _read_events(session)
        names = [e["event"] for e in events]
        assert "trainer.train.start" in names
        assert "trainer.train.complete" in names

    def test_failed_event_written_on_exception(self, tmp_path):
        logger = _fresh_logger(tmp_path)
        session = _start_session(logger, tmp_path)

        @track("data.load")
        def bad_load():
            raise RuntimeError("file missing")

        with patch("xlmtec.utils.telemetry.AppLogger", logger):
            with pytest.raises(RuntimeError):
                bad_load()

        events = _read_events(session)
        names = [e["event"] for e in events]
        assert "data.load.failed" in names
        failed = next(e for e in events if e["event"] == "data.load.failed")
        assert "file missing" in failed["error"]

    def test_elapsed_is_present_and_non_negative(self, tmp_path):
        logger = _fresh_logger(tmp_path)
        session = _start_session(logger, tmp_path)

        @track("sweep.run")
        def fast():
            time.sleep(0.01)

        with patch("xlmtec.utils.telemetry.AppLogger", logger):
            fast()

        events = _read_events(session)
        complete = next(e for e in events if e["event"] == "sweep.run.complete")
        assert complete["elapsed_s"] >= 0


# ============================================================================
# CrashReporter
# ============================================================================


class TestCrashReporter:
    def _make_session(self, tmp_path: Path, cmd: str = "train") -> Session:
        s = Session(cmd=cmd, log_dir=tmp_path).open()
        s.write({"event": "cli.invocation", "cmd": cmd})
        s.write({"event": "stage.load"})
        return s

    def test_write_creates_file(self, tmp_path):
        from xlmtec.utils.crash_report import CrashReporter

        session = self._make_session(tmp_path)
        try:
            raise ValueError("test crash")
        except ValueError as exc:
            path = CrashReporter.write(session, exc, log_dir=tmp_path)
        assert path.exists()
        session.close()

    def test_crash_file_contains_error_type(self, tmp_path):
        from xlmtec.utils.crash_report import CrashReporter

        session = self._make_session(tmp_path)
        try:
            raise RuntimeError("dataset not found")
        except RuntimeError as exc:
            path = CrashReporter.write(session, exc, log_dir=tmp_path)
        content = path.read_text(encoding="utf-8")
        assert "RuntimeError" in content
        assert "dataset not found" in content
        session.close()

    def test_crash_file_contains_session_id(self, tmp_path):
        from xlmtec.utils.crash_report import CrashReporter

        session = self._make_session(tmp_path)
        sid = session.session_id
        try:
            raise ValueError("x")
        except ValueError as exc:
            path = CrashReporter.write(session, exc, log_dir=tmp_path)
        assert sid in path.read_text(encoding="utf-8")
        session.close()

    def test_latest_returns_most_recent(self, tmp_path):
        from xlmtec.utils.crash_report import CrashReporter

        session = self._make_session(tmp_path)
        try:
            raise OSError("disk full")
        except OSError as exc:
            CrashReporter.write(session, exc, log_dir=tmp_path)
        try:
            raise KeyError("missing")
        except KeyError as exc:
            p2 = CrashReporter.write(session, exc, log_dir=tmp_path)
        latest = CrashReporter.latest(log_dir=tmp_path)
        assert latest == p2
        session.close()

    def test_latest_returns_none_when_no_crashes(self, tmp_path):
        from xlmtec.utils.crash_report import CrashReporter

        assert CrashReporter.latest(log_dir=tmp_path) is None

    def test_list_recent_returns_correct_count(self, tmp_path):
        from xlmtec.utils.crash_report import CrashReporter

        session = self._make_session(tmp_path)
        for _ in range(3):
            try:
                raise ValueError("x")
            except ValueError as exc:
                CrashReporter.write(session, exc, log_dir=tmp_path)
        result = CrashReporter.list_recent(n=2, log_dir=tmp_path)
        assert len(result) == 2
        session.close()

    def test_write_does_not_raise_on_none_session(self, tmp_path):
        from xlmtec.utils.crash_report import CrashReporter

        try:
            raise ValueError("no session")
        except ValueError as exc:
            path = CrashReporter.write(None, exc, log_dir=tmp_path)
        assert path.exists()

    def test_crash_file_contains_github_link(self, tmp_path):
        from xlmtec.utils.crash_report import CrashReporter

        session = self._make_session(tmp_path)
        try:
            raise ValueError("x")
        except ValueError as exc:
            path = CrashReporter.write(session, exc, log_dir=tmp_path)
        assert "github.com/Abdur-azure/xlmtec" in path.read_text(encoding="utf-8")
        session.close()


# ============================================================================
# run_report CLI logic
# ============================================================================


class TestRunReport:
    def test_no_log_dir_returns_0(self, tmp_path):
        from xlmtec.cli.commands.report import run_report

        code = run_report(log_dir=tmp_path / "nonexistent")
        assert code == 0

    def test_no_crashes_returns_0(self, tmp_path):
        from xlmtec.cli.commands.report import run_report

        tmp_path.mkdir(exist_ok=True)
        code = run_report(log_dir=tmp_path)
        assert code == 0

    def test_with_crash_returns_0(self, tmp_path):
        from xlmtec.cli.commands.report import run_report
        from xlmtec.utils.crash_report import CrashReporter
        from xlmtec.utils.telemetry import Session

        s = Session(cmd="train", log_dir=tmp_path).open()
        try:
            raise ValueError("test")
        except ValueError as exc:
            CrashReporter.write(s, exc, log_dir=tmp_path)
        s.close()
        code = run_report(log_dir=tmp_path)
        assert code == 0

    def test_sessions_flag_lists_session_files(self, tmp_path):
        from xlmtec.cli.commands.report import run_report
        from xlmtec.utils.telemetry import Session

        s = Session(cmd="sweep", log_dir=tmp_path).open()
        s.close()
        code = run_report(sessions=True, last=5, log_dir=tmp_path)
        assert code == 0
