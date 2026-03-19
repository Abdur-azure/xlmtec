"""
xlmtec.notifications.email
~~~~~~~~~~~~~~~~~~~~~~~~~~~
SMTP email notifier — uses stdlib smtplib, no extra deps.

Usage:
    export XLMTEC_EMAIL_TO=you@example.com
    export XLMTEC_EMAIL_FROM=xlmtec@example.com
    export XLMTEC_SMTP_HOST=smtp.gmail.com
    export XLMTEC_SMTP_PORT=587
    export XLMTEC_SMTP_USER=you@gmail.com
    export XLMTEC_SMTP_PASSWORD=app_password
    xlmtec train --config config.yaml --notify email
"""

from __future__ import annotations

import os
import smtplib
from email.mime.text import MIMEText

from xlmtec.notifications.base import Notifier, NotifyPayload


class EmailNotifier(Notifier):
    name = "email"

    ENV_DEFAULTS = {
        "to":       "XLMTEC_EMAIL_TO",
        "from_":    "XLMTEC_EMAIL_FROM",
        "host":     "XLMTEC_SMTP_HOST",
        "port":     "XLMTEC_SMTP_PORT",
        "user":     "XLMTEC_SMTP_USER",
        "password": "XLMTEC_SMTP_PASSWORD",
    }

    def __init__(
        self,
        to: str | None = None,
        from_: str | None = None,
        host: str | None = None,
        port: int | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        self.to       = to       or os.environ.get(self.ENV_DEFAULTS["to"])
        self.from_    = from_    or os.environ.get(self.ENV_DEFAULTS["from_"], "xlmtec@localhost")
        self.host     = host     or os.environ.get(self.ENV_DEFAULTS["host"], "localhost")
        self.port     = port     or int(os.environ.get(self.ENV_DEFAULTS["port"], "587"))
        self.user     = user     or os.environ.get(self.ENV_DEFAULTS["user"])
        self.password = password or os.environ.get(self.ENV_DEFAULTS["password"])

        if not self.to:
            raise ValueError(
                f"Email recipient not set. Export {self.ENV_DEFAULTS['to']} or pass to= explicitly."
            )

    def send(self, payload: NotifyPayload) -> bool:
        lines = [payload.message, ""]
        for k, v in payload.details.items():
            lines.append(f"{k}: {v}")

        msg = MIMEText("\n".join(lines))
        msg["Subject"] = payload.title
        # FIX line 66: narrow str|None → str (from_ has a default so never None)
        msg["From"]    = self.from_ or "xlmtec@localhost"
        # self.to is guaranteed non-None by __init__ validator above
        msg["To"]      = self.to or ""

        try:
            with smtplib.SMTP(self.host or "localhost", self.port or 587, timeout=10) as server:
                server.ehlo()
                if (self.port or 587) == 587:
                    server.starttls()
                if self.user and self.password:
                    server.login(self.user, self.password)
                # FIX line 75: narrow both str|None → str before passing to sendmail
                from_addr: str = self.from_ or "xlmtec@localhost"
                to_addr: str   = self.to or ""
                server.sendmail(from_addr, [to_addr], msg.as_string())
            return True
        except (smtplib.SMTPException, OSError):
            return False