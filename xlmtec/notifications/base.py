"""
xlmtec.notifications.base
~~~~~~~~~~~~~~~~~~~~~~~~~~
Notifier ABC and shared event types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NotifyEvent(str, Enum):
    TRAINING_COMPLETE = "training_complete"
    TRAINING_FAILED = "training_failed"
    EPOCH_END = "epoch_end"


@dataclass
class NotifyPayload:
    event: NotifyEvent
    run_name: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    # Convenience helpers
    @property
    def title(self) -> str:
        titles = {
            NotifyEvent.TRAINING_COMPLETE: f"✅ Training complete — {self.run_name}",
            NotifyEvent.TRAINING_FAILED: f"❌ Training failed — {self.run_name}",
            NotifyEvent.EPOCH_END: f"📈 Epoch done — {self.run_name}",
        }
        return titles.get(self.event, self.run_name)


class Notifier(ABC):
    """Base class for all notification backends."""

    name: str = "base"

    @abstractmethod
    def send(self, payload: NotifyPayload) -> bool:
        """Send a notification.

        Returns:
            True on success, False on failure (never raises).
        """

    def _safe_send(self, payload: NotifyPayload) -> bool:
        """Wrapper that catches all exceptions and returns False."""
        try:
            return self.send(payload)
        except Exception:
            return False
