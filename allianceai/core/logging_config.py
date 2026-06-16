"""
Centralised logging configuration for AllianceAI.

Every module should call `get_logger(__name__)` rather than building its own
handler chain.  Logs go to both the console (INFO+) and a rotating file
(DEBUG+) so nothing is lost when things go wrong in production.
"""

import logging
import logging.handlers
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_FILE_HANDLER = logging.handlers.RotatingFileHandler(
    _LOG_DIR / "allianceai.log",
    maxBytes=10 * 1024 * 1024,  # 10 MB per file
    backupCount=5,
    encoding="utf-8",
)
_FILE_HANDLER.setLevel(logging.DEBUG)
_FILE_HANDLER.setFormatter(
    logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
)

_CONSOLE_HANDLER = logging.StreamHandler()
_CONSOLE_HANDLER.setLevel(logging.INFO)
_CONSOLE_HANDLER.setFormatter(
    logging.Formatter("%(levelname)-8s %(name)s — %(message)s")
)

# Root logger: capture everything; individual loggers inherit this setup.
_root = logging.getLogger("allianceai")
_root.setLevel(logging.DEBUG)
if not _root.handlers:
    _root.addHandler(_FILE_HANDLER)
    _root.addHandler(_CONSOLE_HANDLER)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'allianceai' hierarchy."""
    return logging.getLogger(f"allianceai.{name}" if not name.startswith("allianceai") else name)
