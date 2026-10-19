"""Generate the fictitious rental sample used by tests and demos.

Run from backend/:  .venv\\Scripts\\python -m scripts.make_rental_sample

Writes data/samples/rental_agreement_sample.pdf (a digital PDF with a real text layer)
and data/samples/rental_page1.png (page 1 rendered to look like a phone photo).
All names, addresses and numbers are invented.
"""

import random
import sys
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw, ImageFilter

SAMPLES_DIR = Path(__file__).resolve().parents[2] / "data" / "samples"
PDF_PATH = SAMPLES_DIR / "rental_agreement_sample.pdf"
PNG_PATH = SAMPLES_DIR / "rental_page1.png"

PAGE_W, PAGE_H = 595, 842  # A4 in points
MARGIN_X, TOP, BOTTOM = 64, 70, 70
FONT, BOLD = "helv", "hebo"
SIZE, LEADING = 10.5, 15.5

TITLE = "LEAVE AND LICENCE AGREEMENT"
PREAMBLE = [
    "This Leave and Licence Agreement is made and executed at Pune on this 20th day of October 2026",
    "BETWEEN Mr. Suresh Vasant Kulkarni, aged about 58 years, residing at 12 Gulmohar Lane, Kothrud, "
    "Pune 411038, hereinafter called the \"LICENSOR\" (which expression shall include his heirs and assigns) "
    "of the ONE PART,",
    "AND Ms. Ananya Prakash Rao, aged about 27 years, presently residing at 7 Lake View Hostel, Hingne, "
    "Pune 411051, hereinafter called the \"LICENSEE\" of the OTHER PART.",
    "WHEREAS the Licensor is the owner of the flat described below and has agreed to allow the Licensee to "
    "use it on leave and licence basis on the following terms and conditions:",
]

CLAUSES: list[tuple[str, str]] = [
    (
        "Premises",
        "The Licensor hereby grants to the Licensee a licence to use Flat No. 304, third floor, Sai Kiran "
        "Residency, Baner Road, Baner, Pune 411045, admeasuring about 650 sq. ft. carpet area, together with "
        "the fittings and fixtures listed in Annexure A (the \"Premises\").",
    ),
    (
        "Period of Licence",
        "This licence is for a period of eleven (11) months commencing from 1st November 2026 and ending "
        "on 30th September 2027, unless terminated earlier as provided in this Agreement.",
    ),
    (
        "Licence Fee",
        "The Licensee shall pay to the Licensor a monthly licence fee of Rs. 18,000/- (Rupees Eighteen "
        "Thousand only) in advance, on or before the 5th day of each English calendar month, by bank "
        "transfer to the account notified by the Licensor in writing.",
    ),
    (
        "Late Payment",
        "If the licence fee is not received by the 5th day of the month, the Licensee shall pay a late fee "
        "of Rs. 500/- for every day of delay. Delay beyond 15 days shall be treated as a breach of this "
        "Agreement.",
    ),
    (
        "Security Deposit",
        "The Licensee has paid an interest-free refundable security deposit of Rs. 1,00,000/- (Rupees One "
        "Lakh only). The deposit shall be refunded within 30 days of the Licensee handing over vacant "
        "possession, after deducting any unpaid dues and the cost of repairing damage beyond normal wear "
        "and tear.",
    ),
    (
        "Lock-in Period",
        "The parties agree to a lock-in period of eleven (11) months from the commencement date. If the "
        "Licensee vacates the Premises before the end of the lock-in period for any reason, the Licensee "
        "shall pay the Licensor compensation equal to two months' licence fee, that is Rs. 36,000/-, which "
        "the Licensor may deduct from the security deposit.",
    ),
    (
        "Termination",
        "The Licensor may terminate this licence at any time by giving seven (7) days' written notice to "
        "the Licensee, without assigning any reason. The Licensee may terminate this licence only after "
        "the lock-in period by giving one (1) month's written notice to the Licensor.",
    ),
    (
        "Increase in Licence Fee",
        "The licence fee shall stand increased by ten per cent (10%) every twelve months, including on any "
        "renewal of this Agreement, over the licence fee last paid.",
    ),
    (
        "Maintenance and Charges",
        "Society maintenance charges and property tax shall be paid by the Licensor. Electricity, water and "
        "piped gas charges as per actual bills shall be paid by the Licensee. Minor repairs costing up to "
        "Rs. 2,000/- each shall be done by the Licensee at its own cost; structural and major repairs shall "
        "be done by the Licensor.",
    ),
    (
        "Use of Premises",
        "The Premises shall be used only for the residential purpose of the Licensee and her immediate "
        "family. The Licensee shall not sub-let, assign or part with possession of the Premises, keep pets "
        "without written consent, or make any structural alteration.",
    ),
    (
        "Inspection",
        "The Licensor or his authorised representative may inspect the Premises at a reasonable time after "
        "giving at least 24 hours' notice to the Licensee.",
    ),
    (
        "Renewal",
        "This Agreement may be renewed for a further period only by mutual agreement of the parties in "
        "writing, on such terms as may then be agreed, subject to the increase in licence fee under "
        "Clause 8.",
    ),
    (
        "Handing Over Possession",
        "On expiry or earlier termination of this licence, the Licensee shall hand over vacant and "
        "peaceful possession of the Premises to the Licensor. If the Licensee fails to do so, the Licensee "
        "shall pay Rs. 1,500/- per day as compensation for the period of unauthorised occupation.",
    ),
    (
        "Dispute Resolution",
        "Any dispute arising out of this Agreement shall first be resolved by mutual discussion. Failing "
        "that, it shall be referred to a sole arbitrator appointed by the Licensor, and the seat of "
        "arbitration shall be Pune. Courts at Pune alone shall have jurisdiction.",
    ),
]

CLOSING = [
    "IN WITNESS WHEREOF the parties have signed this Agreement on the day and year first written above.",
]
SIGNATURES = [
    "LICENSOR: Suresh Vasant Kulkarni            ____________________",
    "LICENSEE: Ananya Prakash Rao                ____________________",
    "Witness 1: Mahesh Dattatray Joshi            ____________________",
    "Witness 2: Priya Sunil Deshmukh              ____________________",
]


class Writer:
    """Lays out wrapped lines one at a time, so every printed line is a real text line."""

    def __init__(self) -> None:
        self.doc = pymupdf.open()
        self.page: pymupdf.Page | None = None
        self.y = 0.0
        self.width = PAGE_W - 2 * MARGIN_X
        self._new_page()

    def _new_page(self) -> None:
        self.page = self.doc.new_page(width=PAGE_W, height=PAGE_H)
        self.y = TOP

    def _ensure(self, lines: int) -> None:
        if self.y + lines * LEADING > PAGE_H - BOTTOM:
            self._new_page()

    def wrap(self, text: str, font: str = FONT, indent: float = 0) -> list[str]:
        words, lines, current = text.split(), [], ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if pymupdf.get_text_length(candidate, fontname=font, fontsize=SIZE) <= self.width - indent:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    def paragraph(self, text: str, font: str = FONT, indent: float = 0, keep_together: bool = True) -> None:
        lines = self.wrap(text, font, indent)
        if keep_together:
            self._ensure(len(lines))
        for line in lines:
            self._ensure(1)
            self.page.insert_text((MARGIN_X + indent, self.y), line, fontname=font, fontsize=SIZE)
            self.y += LEADING

    def heading(self, text: str, size: float = 14) -> None:
        width = pymupdf.get_text_length(text, fontname=BOLD, fontsize=size)
        self.page.insert_text(((PAGE_W - width) / 2, self.y), text, fontname=BOLD, fontsize=size)
        self.y += LEADING * 1.8

    def gap(self, factor: float = 0.6) -> None:
        self.y += LEADING * factor

    def finish(self) -> pymupdf.Document:
        for number, page in enumerate(self.doc, start=1):
            footer = f"Page {number} of {self.doc.page_count}"
            width = pymupdf.get_text_length(footer, fontname=FONT, fontsize=8)
            page.insert_text(((PAGE_W - width) / 2, PAGE_H - 36), footer, fontname=FONT, fontsize=8)
        self.doc.set_metadata({"title": "Leave and Licence Agreement (fictitious sample)", "author": "sample"})
        return self.doc


def build_pdf() -> bytes:
    writer = Writer()
    writer.heading(TITLE)
    for text in PREAMBLE:
        writer.paragraph(text)
        writer.gap()
    writer.paragraph("NOW THIS AGREEMENT WITNESSETH AS FOLLOWS:", font=BOLD)
    writer.gap()
    for number, (title, body) in enumerate(CLAUSES, start=1):
        # The heading and the body are kept on one page so the clause reads naturally.
        body_lines = writer.wrap(body)
        writer._ensure(1 + len(body_lines))
        writer.paragraph(f"{number}. {title}", font=BOLD)
        writer.paragraph(body)
        writer.gap()
    for text in CLOSING:
        writer.paragraph(text)
        writer.gap()
    writer._ensure(len(SIGNATURES) * 2)
    for text in SIGNATURES:
        writer.gap(1.0)
        writer.paragraph(text)
    return writer.finish().tobytes(garbage=3, deflate=True)


def photo_of_page1(pdf_bytes: bytes) -> Image.Image:
    """Render page 1 and make it look like a phone photo of paper on a table."""
    rng = random.Random(7)
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        pix = doc[0].get_pixmap(dpi=130)
        page = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    # Warm, slightly uneven paper with soft lighting from the top left.
    tint = Image.new("RGB", page.size, (246, 238, 222))
    page = Image.blend(page, tint, 0.18)
    shade = Image.new("L", page.size, 0)
    draw = ImageDraw.Draw(shade)
    for i in range(0, page.width + page.height, 8):
        draw.line([(i, 0), (0, i)], fill=min(70, i // 40), width=8)
    page = Image.composite(Image.new("RGB", page.size, (120, 110, 95)), page, shade.point(lambda v: v // 3))

    # Place on a dark wooden-ish table, rotate a little, blur slightly, add sensor noise.
    table = Image.new("RGB", (page.width + 220, page.height + 260), (74, 52, 38))
    table.paste(page, (110, 120))
    table = table.rotate(-1.6, resample=Image.Resampling.BICUBIC, fillcolor=(74, 52, 38))
    table = table.filter(ImageFilter.GaussianBlur(0.7))
    noise = Image.effect_noise(table.size, 18).convert("RGB")
    table = Image.blend(table, noise, 0.06)
    left, top = rng.randint(30, 60), rng.randint(40, 70)
    return table.crop((left, top, table.width - left // 2, table.height - top // 2))


def main() -> int:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_bytes = build_pdf()
    PDF_PATH.write_bytes(pdf_bytes)
    photo_of_page1(pdf_bytes).save(PNG_PATH, optimize=True)
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        pages = doc.page_count
    print(f"Wrote {PDF_PATH} ({pages} pages) and {PNG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
