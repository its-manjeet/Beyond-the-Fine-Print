"""Structured logging with PII masking.

Every record passes through ``PIIMaskingFilter`` before it is formatted, so Aadhaar,
PAN, phone and account numbers never reach a log sink, whether they appear in the
message, its %-args, or structured ``extra`` fields.
"""

import json
import logging
import re
import sys
from datetime import UTC, datetime
from typing import Any

# Order matters: the most specific patterns run first so a 12-digit Aadhaar is not
# swallowed by the generic account-number pattern.
_PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Aadhaar: 12 digits, first digit 2-9, optionally grouped 4-4-4 by space or hyphen.
    (re.compile(r"(?<![\w-])[2-9]\d{3}([ -]?)\d{4}\1\d{4}(?![\w-])"), "[AADHAAR]"),
    # PAN: 5 letters, 4 digits, 1 letter (4th letter is the holder type).
    (re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{3}[ABCFGHLJPTKabcfghljptk][A-Za-z]\d{4}[A-Za-z](?![A-Za-z0-9])"), "[PAN]"),
    # Indian mobile: optional +91 / 91 / 0 prefix, then 10 digits starting 6-9.
    (re.compile(r"(?<![\w+])(?:\+91[ -]?|91[ -]?|0)?[6-9]\d{4}[ -]?\d{5}(?![\w-])"), "[PHONE]"),
    # Bank account numbers: 9-18 consecutive digits.
    (re.compile(r"(?<![\w-])\d{9,18}(?![\w-])"), "[ACCOUNT]"),
]

# LogRecord attributes that are not user-supplied ``extra`` fields.
_RESERVED_ATTRS = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", None, None)).keys() | {"message", "asctime", "taskName", "color_message"}
)


def mask_pii(text: str) -> str:
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _mask_value(value: Any) -> Any:
    if isinstance(value, str):
        return mask_pii(value)
    if isinstance(value, dict):
        return {k: _mask_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_mask_value(v) for v in value)
    if isinstance(value, int) and not isinstance(value, bool) and len(str(abs(value))) >= 9:
        return mask_pii(str(value))
    return value


class PIIMaskingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = mask_pii(record.getMessage())
        record.args = None
        for key, value in list(vars(record).items()):
            if key not in _RESERVED_ATTRS:
                setattr(record, key, _mask_value(value))
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = mask_pii(record.exc_text)
        return True


def _extra_fields(record: logging.LogRecord) -> dict[str, Any]:
    return {k: v for k, v in vars(record).items() if k not in _RESERVED_ATTRS}


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            **_extra_fields(record),
        }
        if record.exc_text:
            payload["exc"] = record.exc_text
        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        extras = " ".join(f"{k}={v}" for k, v in _extra_fields(record).items())
        line = f"{record.levelname:<7} {record.name}: {record.getMessage()}"
        if extras:
            line = f"{line} | {extras}"
        if record.exc_text:
            line = f"{line}\n{record.exc_text}"
        return line


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Install a single masked handler on the root logger (idempotent)."""
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(PIIMaskingFilter())
    handler.setFormatter(JSONFormatter() if fmt == "json" else ConsoleFormatter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        if getattr(existing, "_bfp_handler", False):
            root.removeHandler(existing)
    handler._bfp_handler = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Route uvicorn's loggers through the masked root handler.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers.clear()
        uv_logger.propagate = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
