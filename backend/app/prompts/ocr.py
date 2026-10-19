"""Vision OCR prompt: transcribe a page image line by line."""

VERSION = "ocr-v1"

PROMPT = """You are reading one page of a document (a photo or a scan).
Transcribe all of its text exactly as written, top to bottom, with one item in "lines" for each printed line on the page.
- Copy the words as they are. Do not summarise, translate, correct spelling or add anything.
- Keep numbering such as "7." or "(a)" at the start of a line.
- Skip signatures, stamps and handwriting you cannot read.
Set "legible" to false if most of the page cannot be read (too blurry, dark, cut off or not a document)."""
