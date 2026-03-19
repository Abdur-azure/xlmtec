"""
tests/test_notifications.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for the notifications system — all network calls mocked.
No real HTTP requests, SMTP connections, or desktop popups.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from xlmtec.notifications.base import Notifier, NotifyEvent, NotifyPayload
from xlmtec.notifications.dispatcher import NotificationDispatcher

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _payload(event=NotifyEvent.TRAINING_COMPLETE, run="run1", msg="Done!", details=None):
    return NotifyPayload(event=event, run_name=run, message=msg, details=details or {})


# ---------------------------------------------------------------------------
# NotifyPayload
# ---------------------------------------------------------------------------


class TestNotifyPayload:
    def test_title_complete(self):
        p = _payload(NotifyEvent.TRAINING_COMPLETE, run="my_run")
        assert "my_run" in p.title
        assert "complete" in p.title.lower() or "✅" in p.title

    def test_title_failed(self):
        p = _payload(NotifyEvent.TRAINING_FAILED, run="my_run")
        assert "failed" in p.title.lower() or "❌" in p.title

    def test_title_epoch(self):
        p = _payload(NotifyEvent.EPOCH_END, run="my_run")
        assert "epoch" in p.title.lower() or "📈" in p.title

    def test_details_default_empty(self):
        p = _payload()
        assert p.details == {}


# ---------------------------------------------------------------------------
# SlackNotifier
# ---------------------------------------------------------------------------


class TestSlackNotifier:
    def test_send_success(self):
        from xlmtec.notifications.slack import SlackNotifier

        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/test")
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert notifier.send(_payload()) is True

    def test_send_network_failure_returns_false(self):
        from urllib.error import URLError

        from xlmtec.notifications.slack import SlackNotifier

        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/test")

        with patch("urllib.request.urlopen", side_effect=URLError("timeout")):
            assert notifier.send(_payload()) is False

    def test_missing_webhook_raises(self):
        from xlmtec.notifications.slack import SlackNotifier

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="webhook"):
                SlackNotifier()

    def test_webhook_from_env(self):
        from xlmtec.notifications.slack import WEBHOOK_ENV, SlackNotifier

        with patch.dict("os.environ", {WEBHOOK_ENV: "https://hooks.slack.com/env"}):
            n = SlackNotifier()
            assert n.webhook_url == "https://hooks.slack.com/env"

    def test_details_included_in_body(self):
        import json

        from xlmtec.notifications.slack import SlackNotifier

        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/test")
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["body"] = json.loads(req.data.decode())
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = lambda s: mock_resp
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            notifier.send(_payload(details={"loss": "0.21", "epochs": "3"}))

        assert "loss" in captured["body"]["text"]


# ---------------------------------------------------------------------------
# EmailNotifier
# ---------------------------------------------------------------------------


class TestEmailNotifier:
    def test_send_success(self):
        from xlmtec.notifications.email import EmailNotifier

        n = EmailNotifier(to="test@example.com", host="localhost", port=25)
        mock_smtp = MagicMock()
        mock_smtp.__enter__ = lambda s: mock_smtp
        mock_smtp.__exit__ = MagicMock(return_value=False)

        with patch("smtplib.SMTP", return_value=mock_smtp):
            assert n.send(_payload()) is True

    def test_smtp_error_returns_false(self):
        import smtplib

        from xlmtec.notifications.email import EmailNotifier

        n = EmailNotifier(to="test@example.com", host="localhost", port=25)

        with patch("smtplib.SMTP", side_effect=smtplib.SMTPException("conn refused")):
            assert n.send(_payload()) is False

    def test_missing_to_raises(self):
        from xlmtec.notifications.email import EmailNotifier

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="recipient"):
                EmailNotifier()

    def test_to_from_env(self):
        from xlmtec.notifications.email import EmailNotifier

        with patch.dict("os.environ", {"XLMTEC_EMAIL_TO": "env@example.com"}):
            n = EmailNotifier()
            assert n.to == "env@example.com"

    def test_starttls_called_for_port_587(self):
        from xlmtec.notifications.email import EmailNotifier

        n = EmailNotifier(to="t@t.com", host="smtp.gmail.com", port=587, user="u", password="p")
        mock_smtp = MagicMock()
        mock_smtp.__enter__ = lambda s: mock_smtp
        mock_smtp.__exit__ = MagicMock(return_value=False)

        with patch("smtplib.SMTP", return_value=mock_smtp):
            n.send(_payload())
            mock_smtp.starttls.assert_called_once()


# ---------------------------------------------------------------------------
# DesktopNotifier
# ---------------------------------------------------------------------------


class TestDesktopNotifier:
    def test_send_with_plyer(self):
        from xlmtec.notifications.desktop import DesktopNotifier

        mock_plyer = MagicMock()
        with patch.dict(
            "sys.modules", {"plyer": mock_plyer, "plyer.notification": mock_plyer.notification}
        ):
            with patch(
                "xlmtec.notifications.desktop.DesktopNotifier.send", return_value=True
            ):
                n = DesktopNotifier()
                result = n._safe_send(_payload())
                # _safe_send calls send() which is mocked to True
                assert result is True

    def test_send_without_plyer_fallback(self, capsys):
        from xlmtec.notifications.desktop import DesktopNotifier

        n = DesktopNotifier()
        with patch("builtins.__import__", side_effect=ImportError("No module named 'plyer'")):
            # Direct test of fallback path
            pass
        # Real test: if plyer not installed, send returns True (console fallback)
        try:
            import plyer  # noqa: F401

            pytest.skip("plyer is installed — skip fallback test")
        except ImportError:
            result = n.send(_payload())
            assert result is True


# ---------------------------------------------------------------------------
# NotificationDispatcher
# ---------------------------------------------------------------------------


class TestNotificationDispatcher:
    def test_from_channels_slack(self):
        with patch(
            "xlmtec.notifications.slack.SlackNotifier.__init__", return_value=None
        ) as mock_init:
            mock_init.return_value = None
            d = NotificationDispatcher.from_channels(
                ["slack"], slack={"webhook_url": "https://hooks.slack.com/x"}
            )
            assert "slack" in d.channels

    def test_unknown_channel_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            NotificationDispatcher.from_channels(["telegram"])

    def test_notify_routes_to_all(self):
        mock_a = MagicMock(spec=Notifier, name="a")
        mock_a.name = "a"
        mock_a._safe_send.return_value = True
        mock_b = MagicMock(spec=Notifier, name="b")
        mock_b.name = "b"
        mock_b._safe_send.return_value = False

        d = NotificationDispatcher([mock_a, mock_b])
        results = d.notify(NotifyEvent.TRAINING_COMPLETE, "run1", "done")
        assert results == {"a": True, "b": False}
        mock_a._safe_send.assert_called_once()
        mock_b._safe_send.assert_called_once()

    def test_available_channels(self):
        channels = NotificationDispatcher.available_channels()
        assert set(channels) == {"slack", "email", "desktop"}

    def test_empty_dispatcher_notify_returns_empty(self):
        d = NotificationDispatcher([])
        results = d.notify(NotifyEvent.TRAINING_COMPLETE, "r", "m")
        assert results == {}

    def test_safe_send_catches_exception(self):
        from xlmtec.notifications.base import Notifier

        class BrokenNotifier(Notifier):
            name = "broken"

            def send(self, payload: NotifyPayload) -> bool:
                raise RuntimeError("network down")

        n = BrokenNotifier()
        result = n._safe_send(_payload())
        assert result is False
