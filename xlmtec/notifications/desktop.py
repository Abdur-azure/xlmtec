"""
xlmtec.notifications.desktop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Desktop OS notification via plyer (optional dep).

Install:
    pip install xlmtec[notify]

Falls back to a console print if plyer is not installed.
"""

from __future__ import annotations

from xlmtec.notifications.base import Notifier, NotifyPayload

APP_NAME = "xlmtec"
TIMEOUT = 10  # seconds


class DesktopNotifier(Notifier):
    name = "desktop"

    def send(self, payload: NotifyPayload) -> bool:
        try:
            from plyer import notification  # type: ignore[import]

            notification.notify(
                title=payload.title,
                message=payload.message,
                app_name=APP_NAME,
                timeout=TIMEOUT,
            )
            return True
        except ImportError:
            # Graceful fallback — print to console if plyer not installed
            print(f"\n[xlmtec notify] {payload.title}\n{payload.message}\n")
            return True
        except Exception:
            return False
