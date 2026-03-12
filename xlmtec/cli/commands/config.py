"""
xlmtec.cli.commands.config
~~~~~~~~~~~~~~~~~~~~~~~~~~~
DEPRECATED — superseded by config_validate.py (Sprint 35).

This file is kept as a tombstone only. The active config commands live in
``xlmtec/cli/commands/config_validate.py`` and are registered in main.py as:

    app.add_typer(config_validate.app, name="config")

Do not add logic here. Delete this file once you are certain nothing in your
environment imports it directly.
"""

import warnings

warnings.warn(
    "xlmtec.cli.commands.config is deprecated and will be removed. "
    "Use xlmtec.cli.commands.config_validate instead.",
    DeprecationWarning,
    stacklevel=2,
)
