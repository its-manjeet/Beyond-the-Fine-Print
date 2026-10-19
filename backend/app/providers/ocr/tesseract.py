"""Offline OCR fallback via the Tesseract command-line tool, used only if it is installed."""

import asyncio
import csv
import io
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class OcrLine:
    text: str
    bbox: list[float]  # [x0, y0, x1, y1] as fractions of the image size
    confidence: float  # mean word confidence, 0-100


def tesseract_available(cmd: str) -> bool:
    return shutil.which(cmd) is not None


async def tesseract_lines(cmd: str, image_png: bytes, width: int, height: int, lang: str = "eng") -> list[OcrLine]:
    """OCR a PNG image and return its lines in reading order."""
    result = await asyncio.to_thread(
        subprocess.run,
        [cmd, "stdin", "stdout", "-l", lang, "tsv"],
        input=image_png,
        capture_output=True,
        timeout=120,
        check=True,
    )
    return parse_tsv(result.stdout.decode("utf-8", errors="replace"), width, height)


def parse_tsv(tsv: str, width: int, height: int) -> list[OcrLine]:
    """Group Tesseract's word-level TSV rows into lines."""
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
        if row.get("level") != "5" or not (row.get("text") or "").strip():
            continue
        key = (row["page_num"], row["block_num"], row["par_num"], row["line_num"])
        groups.setdefault(key, []).append(row)

    lines: list[OcrLine] = []
    for words in groups.values():
        x0 = min(int(w["left"]) for w in words)
        y0 = min(int(w["top"]) for w in words)
        x1 = max(int(w["left"]) + int(w["width"]) for w in words)
        y1 = max(int(w["top"]) + int(w["height"]) for w in words)
        confidences = [float(w["conf"]) for w in words if float(w["conf"]) >= 0]
        lines.append(
            OcrLine(
                text=" ".join(w["text"].strip() for w in words),
                bbox=[round(x0 / width, 4), round(y0 / height, 4), round(x1 / width, 4), round(y1 / height, 4)],
                confidence=sum(confidences) / len(confidences) if confidences else 0.0,
            )
        )
    lines.sort(key=lambda line: (line.bbox[1], line.bbox[0]))
    return lines
