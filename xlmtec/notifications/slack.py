"""
xlmtec.notifications.slack
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Slack webhook notifier — uses stdlib urllib, no extra deps.

Usage:
    export XLMTEC_SLACK_WEBHOOK=https://hooks.slack.com/services/...
    xlmtec train --config config.yaml --notify slack
"""

from __future__ import annotations

import json
import os
import urllib.request
from urllib.error import URLError

from xlmtec.notifications.base import Notifier, NotifyPayload

WEBHOOK_ENV = "XLMTEC_SLACK_WEBHOOK"


class SlackNotifier(Notifier):
    name = "slack"

    def __init__(self, webhook_url: str | None = None) -> None:
        self.webhook_url = webhook_url or os.environ.get(WEBHOOK_ENV)
        if not self.webhook_url:
            raise ValueError(
                f"Slack webhook URL not set. Export {WEBHOOK_ENV} or pass webhook_url= explicitly."
            )

    def send(self, payload: NotifyPayload) -> bool:
        lines = [f"*{payload.title}*", payload.message]
        if payload.details:
            for k, v in payload.details.items():
                lines.append(f"• {k}: `{v}`")

        body = json.dumps({"text": "\n".join(lines)}).encode()
        url: str = self.webhook_url or ""   # ← narrow to str first
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except URLError:
            return False
