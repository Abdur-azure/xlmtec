"""
xlmtec.notifications.dispatcher
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
NotificationDispatcher — builds notifiers from a list of channel names
and routes NotifyPayload events to all of them.

Usage:
    from xlmtec.notifications.dispatcher import NotificationDispatcher
    from xlmtec.notifications.base import NotifyEvent, NotifyPayload

    d = NotificationDispatcher.from_channels(["slack", "desktop"])
    d.notify(NotifyEvent.TRAINING_COMPLETE, run_name="run1",
             message="3 epochs done.", details={"loss": 0.21})
"""

from __future__ import annotations

from xlmtec.notifications.base import Notifier, NotifyEvent, NotifyPayload

# Registry mapping channel name → notifier class
_NOTIFIERS: dict[str, type[Notifier]] = {}


def _register() -> None:
    """Lazily populate the registry the first time it's needed."""
    if _NOTIFIERS:
        return
    from xlmtec.notifications.desktop import DesktopNotifier
    from xlmtec.notifications.email import EmailNotifier
    from xlmtec.notifications.slack import SlackNotifier

    _NOTIFIERS["slack"] = SlackNotifier
    _NOTIFIERS["email"] = EmailNotifier
    _NOTIFIERS["desktop"] = DesktopNotifier


class NotificationDispatcher:
    """Routes notification events to one or more notifier backends."""

    def __init__(self, notifiers: list[Notifier]) -> None:
        self._notifiers = notifiers

    @classmethod
    def from_channels(cls, channels: list[str], **kwargs) -> "NotificationDispatcher":
        """Build a dispatcher from a list of channel names.

        Args:
            channels: e.g. ["slack", "desktop"]
            **kwargs: passed to each notifier constructor

        Raises:
            ValueError: If a channel name is unknown.
        """
        _register()
        notifiers: list[Notifier] = []
        for ch in channels:
            ch = ch.lower().strip()
            if ch not in _NOTIFIERS:
                known = ", ".join(sorted(_NOTIFIERS))
                raise ValueError(f"Unknown notification channel: {ch!r}. Known: {known}")
            notifiers.append(_NOTIFIERS[ch](**kwargs.get(ch, {})))
        return cls(notifiers)

    def notify(
        self,
        event: NotifyEvent,
        run_name: str,
        message: str,
        details: dict | None = None,
    ) -> dict[str, bool]:
        """Send an event to all notifiers.

        Returns:
            dict mapping notifier name → success bool
        """
        payload = NotifyPayload(
            event=event,
            run_name=run_name,
            message=message,
            details=details or {},
        )
        results: dict[str, bool] = {}
        for n in self._notifiers:
            results[n.name] = n._safe_send(payload)
        return results

    @property
    def channels(self) -> list[str]:
        return [n.name for n in self._notifiers]

    @staticmethod
    def available_channels() -> list[str]:
        _register()
        return sorted(_NOTIFIERS.keys())
