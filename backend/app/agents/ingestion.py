"""F1 Upload & ingest: validate the upload and extract numbered lines with positions.

- Digital PDF pages: PyMuPDF words are grouped into lines by vertical position.
- Pages without a usable text layer, and photos: the page image goes to the provider's
  vision model, which transcribes it line by line. Tesseract is a fallback, used only if
  it is installed.
- Pages with very little text are flagged as low confidence.
"""

import io
from dataclasses import dataclass
from typing import Literal

import pymupdf
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

from app.core.config import Settings
from app.core.llm import ImageInput, LLMError, LLMProvider
from app.core.logging import get_logger
from app.prompts import ocr as ocr_prompt
from app.providers.ocr.tesseract import tesseract_available, tesseract_lines

logger = get_logger(__name__)

FileKind = Literal["pdf", "png", "jpeg"]
CONTENT_TYPES: dict[FileKind, str] = {"pdf": "application/pdf", "png": "image/png", "jpeg": "image/jpeg"}
TESSERACT_MIN_CONFIDENCE = 60.0


class UploadError(Exception):
    """The upload is not acceptable; ``status_code`` is the HTTP status to return."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


@dataclass(frozen=True)
class UploadInfo:
    kind: FileKind
    content_type: str
    page_count: int


class ExtractedLine(BaseModel):
    line_number: int
    text: str
    bbox: list[float] | None = None


class ExtractedPage(BaseModel):
    page_number: int
    width: float
    height: float
    source: Literal["text_layer", "vision", "tesseract", "none"]
    low_confidence: bool
    lines: list[ExtractedLine]


class IngestResult(BaseModel):
    pages: list[ExtractedPage]
    warnings: list[str] = Field(default_factory=list)

    @property
    def line_count(self) -> int:
        return sum(len(page.lines) for page in self.pages)


class OcrOutput(BaseModel):
    """What the vision model returns for one page."""

    lines: list[str] = Field(description="The page's text, one item per printed line, top to bottom")
    legible: bool = Field(description="False if most of the page cannot be read")


# --- validation ------------------------------------------------------------------------


def sniff_kind(data: bytes) -> FileKind | None:
    if data.startswith(b"%PDF-"):
        return "pdf"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    return None


def validate_upload(data: bytes, settings: Settings) -> UploadInfo:
    """Check type, size and page count from the bytes themselves (not the filename)."""
    if not data:
        raise UploadError(400, "The file is empty.")
    limit = settings.max_upload_mb * 1024 * 1024
    if len(data) > limit:
        raise UploadError(
            413, f"The file is {len(data) / 1024 / 1024:.1f} MB. The limit is {settings.max_upload_mb} MB."
        )
    kind = sniff_kind(data)
    if kind is None:
        raise UploadError(400, "Unsupported file type. Upload a PDF, JPG or PNG file.")

    if kind == "pdf":
        try:
            with pymupdf.open(stream=data, filetype="pdf") as doc:
                if doc.needs_pass:
                    raise UploadError(400, "This PDF is password-protected. Remove the password and upload it again.")
                pages = doc.page_count
        except UploadError:
            raise
        except Exception as exc:  # PyMuPDF raises several types for damaged files
            raise UploadError(400, "This PDF could not be opened. It may be damaged.") from exc
        if pages == 0:
            raise UploadError(400, "This PDF has no pages.")
        if pages > settings.max_pages:
            raise UploadError(400, f"The PDF has {pages} pages. The limit is {settings.max_pages} pages.")
        return UploadInfo(kind, CONTENT_TYPES[kind], pages)

    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise UploadError(400, "This image could not be read. It may be damaged.") from exc
    return UploadInfo(kind, CONTENT_TYPES[kind], 1)


# --- extraction ------------------------------------------------------------------------


async def ingest(data: bytes, info: UploadInfo, llm: LLMProvider | None, settings: Settings) -> IngestResult:
    """Extract every page's lines. ``llm`` is used for vision OCR; None disables it."""
    pages: list[ExtractedPage] = []
    warnings: list[str] = []
    if info.kind == "pdf":
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            for index, page in enumerate(doc, start=1):
                lines = text_layer_lines(page)
                if _char_count(lines) >= settings.low_text_chars:
                    pages.append(_page(index, page.rect.width, page.rect.height, "text_layer", lines, settings))
                    continue
                image = _render(page, settings.ocr_max_image_side)
                page_result = await _ocr_page(index, image, llm, settings, warnings, fallback_lines=lines)
                # Report PDF pages in points like text-layer pages; bboxes are fractions anyway.
                page_result.width, page_result.height = round(page.rect.width, 2), round(page.rect.height, 2)
                pages.append(page_result)
    else:
        image = _load_photo(data)
        pages.append(await _ocr_page(1, _fit(image, settings.ocr_max_image_side), llm, settings, warnings))

    for page in pages:
        if page.low_confidence:
            warnings.append(f"Page {page.page_number} has very little readable text; results for it may be incomplete.")
    return IngestResult(pages=pages, warnings=warnings)


def text_layer_lines(page: pymupdf.Page) -> list[ExtractedLine]:
    """Group the page's words into lines by vertical position, numbered from 1."""
    width, height = page.rect.width, page.rect.height
    words = [w for w in page.get_text("words") if w[4].strip()]
    words.sort(key=lambda w: ((w[1] + w[3]) / 2, w[0]))

    rows: list[list[tuple]] = []
    row_center = row_height = 0.0
    for word in words:
        center, word_height = (word[1] + word[3]) / 2, word[3] - word[1]
        if rows and abs(center - row_center) <= 0.5 * max(word_height, row_height):
            rows[-1].append(word)
            count = len(rows[-1])
            row_center += (center - row_center) / count
            row_height = max(row_height, word_height)
        else:
            rows.append([word])
            row_center, row_height = center, word_height

    lines: list[ExtractedLine] = []
    for number, row in enumerate(rows, start=1):
        row.sort(key=lambda w: w[0])
        x0, y0 = min(w[0] for w in row), min(w[1] for w in row)
        x1, y1 = max(w[2] for w in row), max(w[3] for w in row)
        lines.append(
            ExtractedLine(
                line_number=number,
                text=" ".join(w[4] for w in row),
                bbox=[round(x0 / width, 4), round(y0 / height, 4), round(x1 / width, 4), round(y1 / height, 4)],
            )
        )
    return lines


async def _ocr_page(
    number: int,
    image: Image.Image,
    llm: LLMProvider | None,
    settings: Settings,
    warnings: list[str],
    fallback_lines: list[ExtractedLine] | None = None,
) -> ExtractedPage:
    width, height = image.size
    if llm is not None:
        try:
            jpeg = _to_jpeg(image)
            result = await llm.generate_structured(
                ocr_prompt.PROMPT, OcrOutput, images=[ImageInput(jpeg, "image/jpeg")], max_tokens=4096
            )
            lines = [
                ExtractedLine(line_number=i, text=text.strip())
                for i, text in enumerate((t for t in result.lines if t.strip()), start=1)
            ]
            page = _page(number, width, height, "vision", lines, settings)
            if not result.legible:
                page.low_confidence = True
            return page
        except LLMError as exc:
            logger.warning("ingest.vision_failed", extra={"page": number, "error": type(exc).__name__})
            warnings.append(f"Page {number}: the vision model could not read it ({type(exc).__name__}).")

    if tesseract_available(settings.tesseract_cmd):
        try:
            ocr_lines = await tesseract_lines(settings.tesseract_cmd, _to_png(image), width, height)
        except Exception as exc:  # a broken install must not fail the whole document
            logger.warning("ingest.tesseract_failed", extra={"page": number, "error": type(exc).__name__})
        else:
            lines = [ExtractedLine(line_number=i, text=l.text, bbox=l.bbox) for i, l in enumerate(ocr_lines, start=1)]
            page = _page(number, width, height, "tesseract", lines, settings)
            confidences = [l.confidence for l in ocr_lines]
            if confidences and sum(confidences) / len(confidences) < TESSERACT_MIN_CONFIDENCE:
                page.low_confidence = True
            return page

    # Nothing could read the image: keep whatever little text layer there was.
    lines = fallback_lines or []
    return _page(number, width, height, "text_layer" if lines else "none", lines, settings, force_low=True)


def _page(
    number: int,
    width: float,
    height: float,
    source: str,
    lines: list[ExtractedLine],
    settings: Settings,
    force_low: bool = False,
) -> ExtractedPage:
    return ExtractedPage(
        page_number=number,
        width=round(width, 2),
        height=round(height, 2),
        source=source,  # type: ignore[arg-type]
        low_confidence=force_low or _char_count(lines) < settings.low_text_chars,
        lines=lines,
    )


def _char_count(lines: list[ExtractedLine]) -> int:
    return sum(len("".join(line.text.split())) for line in lines)


def _render(page: pymupdf.Page, max_side: int) -> Image.Image:
    zoom = min(max_side / max(page.rect.width, page.rect.height), 300 / 72)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _load_photo(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as image:
        # Phone photos are often stored sideways with an EXIF rotation flag.
        return ImageOps.exif_transpose(image).convert("RGB")


def _fit(image: Image.Image, max_side: int) -> Image.Image:
    if max(image.size) <= max_side:
        return image
    image = image.copy()
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def _to_jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def _to_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
