import json
import logging

import pytest

from app.core.logging import JSONFormatter, PIIMaskingFilter, mask_pii


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Aadhaar 2345 6789 0123 on file", "Aadhaar [AADHAAR] on file"),
        ("aadhaar=2345-6789-0123", "aadhaar=[AADHAAR]"),
        ("id 234567890123", "id [AADHAAR]"),
        ("PAN ABCPE1234F", "PAN [PAN]"),
        ("pan: abcpe1234f.", "pan: [PAN]."),
        ("call +91 98765 43210", "call [PHONE]"),
        ("call 09876543210 now", "call [PHONE] now"),
        ("mobile 9876543210", "mobile [PHONE]"),
        ("A/c no. 50100123456789 debited", "A/c no. [ACCOUNT] debited"),
        ("account 123456789", "account [ACCOUNT]"),
    ],
)
def test_masks_pii(raw: str, expected: str) -> None:
    assert mask_pii(raw) == expected


@pytest.mark.parametrize(
    "safe",
    [
        "Rent of Rs 15000 due on 2026-10-05",
        "Section 106 of the Transfer of Property Act, 1882",
        "Pincode 400001, clause c-012, 11 months",
        "latency_ms=1234.5 tokens=4096",
    ],
)
def test_leaves_non_pii_alone(safe: str) -> None:
    assert mask_pii(safe) == safe


def test_filter_masks_message_args_and_extra_fields() -> None:
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "user phone %s", ("9876543210",), None)
    record.account = "50100123456789"
    record.details = {"pan": "ABCPE1234F", "n": 234567890123}

    PIIMaskingFilter().filter(record)
    payload = json.loads(JSONFormatter().format(record))

    assert payload["msg"] == "user phone [PHONE]"
    assert payload["account"] == "[ACCOUNT]"
    assert payload["details"] == {"pan": "[PAN]", "n": "[AADHAAR]"}
    assert "9876543210" not in json.dumps(payload)
