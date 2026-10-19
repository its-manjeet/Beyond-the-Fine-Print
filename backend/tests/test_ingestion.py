"""F1: upload validation and line extraction. The LLM is mocked; no network."""

import io

import pymupdf
import pytest
from PIL import Image

from app.agents import ingestion
from app.agents.ingestion import UploadError, ingest, validate_upload
from app.core.config import Settings
from app.core.llm import LLMRequestError
from tests.mocks import MockProvider, sample_pdf_bytes


@pytest.fixture
def no_tesseract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ingestion, "tesseract_available", lambda cmd: False)


def pdf_with_pages(count: int, text: str = "Hello world, this page has enough text to be read.") -> bytes:
    doc = pymupdf.open()
    for _ in range(count):
        doc.new_page().insert_text((72, 72), text)
    return doc.tobytes()


def image_only_pdf() -> bytes:
    """A 'scanned' PDF: one page that is just a picture, with no text layer."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(page.rect, stream=png_bytes())
    return doc.tobytes()


def png_bytes(size: tuple[int, int] = (400, 300)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, "white").save(buffer, format="PNG")
    return buffer.getvalue()


def jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (300, 400), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


# --- sample PDF line numbering ---------------------------------------------------------


async def test_sample_pdf_lines_are_numbered_per_page_with_bboxes(settings: Settings) -> None:
    data = sample_pdf_bytes()
    result = await ingest(data, validate_upload(data, settings), None, settings)

    assert [len(p.lines) for p in result.pages] == [35, 33, 5]
    assert all(p.source == "text_layer" and not p.low_confidence for p in result.pages)
    page1, page2 = result.pages[0], result.pages[1]
    assert page1.lines[0].text == "LEAVE AND LICENCE AGREEMENT"
    assert page1.lines[17].line_number == 18 and page1.lines[17].text == "3. Licence Fee"
    assert "Rs. 18,000/-" in page1.lines[18].text
    assert page1.lines[18].text.startswith("The Licensee shall pay to the Licensor a monthly licence fee")
    assert page2.lines[0].line_number == 1 and page2.lines[0].text == "7. Termination"
    assert page2.lines[1].text.startswith("The Licensor may terminate this licence at any time by giving seven (7)")
    for page in result.pages:
        assert [line.line_number for line in page.lines] == list(range(1, len(page.lines) + 1))
        tops = [line.bbox[1] for line in page.lines]
        assert tops == sorted(tops)
        assert all(0 <= v <= 1 for line in page.lines for v in line.bbox)
    assert result.warnings == []


async def test_words_on_one_baseline_form_one_line_in_x_order(settings: Settings) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((300, 100), "right side")
    page.insert_text((72, 100), "Left side and")
    page.insert_text((72, 130), "Second line of text here for the threshold")
    lines = ingestion.text_layer_lines(page)

    assert [line.text for line in lines] == ["Left side and right side", "Second line of text here for the threshold"]


# --- validation ------------------------------------------------------------------------


def test_accepts_pdf_png_and_jpeg(settings: Settings) -> None:
    assert validate_upload(sample_pdf_bytes(), settings).page_count == 3
    assert validate_upload(png_bytes(), settings).kind == "png"
    assert validate_upload(jpeg_bytes(), settings).content_type == "image/jpeg"


@pytest.mark.parametrize(
    ("data", "status", "message"),
    [
        (b"", 400, "empty"),
        (b"PK\x03\x04 this is a zip file", 400, "Upload a PDF, JPG or PNG"),
        (b"%PDF-1.7 this is not really a pdf", 400, "could not be opened"),
        (b"\x89PNG\r\n\x1a\n not an image", 400, "image could not be read"),
    ],
)
def test_rejects_bad_files(settings: Settings, data: bytes, status: int, message: str) -> None:
    with pytest.raises(UploadError, match=message) as info:
        validate_upload(data, settings)
    assert info.value.status_code == status


def test_rejects_file_over_size_limit_with_413(settings: Settings) -> None:
    settings.max_upload_mb = 1
    with pytest.raises(UploadError, match="limit is 1 MB") as info:
        validate_upload(b"%PDF-" + b"0" * (1024 * 1024), settings)
    assert info.value.status_code == 413


def test_rejects_pdf_over_page_limit(settings: Settings) -> None:
    assert validate_upload(pdf_with_pages(20), settings).page_count == 20
    with pytest.raises(UploadError, match="21 pages. The limit is 20") as info:
        validate_upload(pdf_with_pages(21), settings)
    assert info.value.status_code == 400


def test_rejects_password_protected_pdf(settings: Settings) -> None:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret")
    encrypted = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")
    with pytest.raises(UploadError, match="password-protected"):
        validate_upload(encrypted, settings)


# --- OCR paths -------------------------------------------------------------------------


async def test_photo_is_read_by_vision_model(settings: Settings, no_tesseract: None) -> None:
    llm = MockProvider({"OcrOutput": {"lines": ["LEAVE AND LICENCE AGREEMENT", " ", "1. Premises: Flat 304, Baner"], "legible": True}})
    data = png_bytes((2400, 3200))

    result = await ingest(data, validate_upload(data, settings), llm, settings)

    page = result.pages[0]
    assert page.source == "vision" and not page.low_confidence
    assert [(l.line_number, l.text, l.bbox) for l in page.lines] == [
        (1, "LEAVE AND LICENCE AGREEMENT", None),
        (2, "1. Premises: Flat 304, Baner", None),
    ]
    image = llm.calls[0]["images"][0]
    assert image.media_type == "image/jpeg"
    with Image.open(io.BytesIO(image.data)) as sent:
        assert max(sent.size) == settings.ocr_max_image_side  # downscaled before sending


async def test_scanned_pdf_page_goes_to_vision(settings: Settings, no_tesseract: None) -> None:
    llm = MockProvider({"OcrOutput": {"lines": ["The licensee shall pay rent of Rs. 18,000 by the 5th."], "legible": True}})
    data = image_only_pdf()

    result = await ingest(data, validate_upload(data, settings), llm, settings)

    assert result.pages[0].source == "vision"
    assert len(llm.calls) == 1


async def test_little_text_is_flagged_low_confidence(settings: Settings, no_tesseract: None) -> None:
    llm = MockProvider({"OcrOutput": {"lines": ["Page 2"], "legible": True}})
    data = png_bytes()

    result = await ingest(data, validate_upload(data, settings), llm, settings)

    assert result.pages[0].low_confidence
    assert "very little readable text" in result.warnings[0]


async def test_illegible_page_is_flagged_low_confidence(settings: Settings, no_tesseract: None) -> None:
    llm = MockProvider({"OcrOutput": {"lines": ["x" * 80], "legible": False}})
    data = png_bytes()

    result = await ingest(data, validate_upload(data, settings), llm, settings)

    assert result.pages[0].low_confidence


async def test_vision_failure_without_tesseract_gives_empty_low_confidence_page(
    settings: Settings, no_tesseract: None
) -> None:
    llm = MockProvider({"OcrOutput": LLMRequestError("GROQ_VISION_MODEL is not set")})
    data = png_bytes()

    result = await ingest(data, validate_upload(data, settings), llm, settings)

    page = result.pages[0]
    assert (page.source, page.lines, page.low_confidence) == ("none", [], True)
    assert any("vision model could not read it" in w for w in result.warnings)


async def test_tesseract_is_used_as_fallback_when_installed(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.providers.ocr.tesseract import parse_tsv

    tsv = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t10\t20\t50\t10\t91\tRent\n"
        "5\t1\t1\t1\t1\t2\t70\t20\t40\t10\t89\tis\n"
        "5\t1\t1\t1\t2\t1\t10\t40\t90\t10\t95\tRs.18,000\n"
    )

    async def fake_tesseract(cmd, image_png, width, height, lang="eng"):
        return parse_tsv(tsv, width, height)

    monkeypatch.setattr(ingestion, "tesseract_available", lambda cmd: True)
    monkeypatch.setattr(ingestion, "tesseract_lines", fake_tesseract)
    llm = MockProvider({"OcrOutput": LLMRequestError("no vision model")})
    data = png_bytes((200, 100))

    result = await ingest(data, validate_upload(data, settings), llm, settings)

    page = result.pages[0]
    assert page.source == "tesseract"
    assert [(l.text, l.bbox) for l in page.lines] == [
        ("Rent is", [0.05, 0.2, 0.55, 0.3]),
        ("Rs.18,000", [0.05, 0.4, 0.5, 0.5]),
    ]
    assert page.low_confidence  # only 15 characters
