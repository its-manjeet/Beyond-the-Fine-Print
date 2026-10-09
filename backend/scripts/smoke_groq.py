"""Live smoke test for the Groq provider. Makes 4 small real API calls.

Run from backend/:  .venv\\Scripts\\python -m scripts.smoke_groq

1. text, strict structured output (GROQ_TEXT_MODEL)
2. a second text model, to compare x-ratelimit-* headers across models
3. the first model again, to see whether its budget moved during call 2
4. image input with strict structured output (GROQ_VISION_MODEL)
"""

import asyncio
import struct
import sys
import zlib
from typing import Literal

from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.llm import ImageInput
from app.core.llm.groq_provider import GroqProvider, _shared_limiter
from app.core.logging import configure_logging
from app.core.schemas import Claim

SAMPLE = (
    "Clause 7 (Notice): Either party may terminate this leave and licence agreement by giving "
    "one month's written notice. Clause 8 (Lock-in): The licensee shall not vacate during the "
    "first 6 months; if they do, the security deposit of Rs 50,000 is forfeited."
)


class ImageFacts(BaseModel):
    dominant_color: Literal["red", "green", "blue", "other"]
    shape: str = Field(description="One word, e.g. square, circle")
    has_text: bool


def red_square_png(size: int = 64) -> bytes:
    """A solid red square, built without image libraries."""
    row = b"\x00" + b"\xff\x00\x00" * size
    raw = row * size

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


async def main() -> int:
    settings = get_settings()
    configure_logging("INFO", "console")
    limiter = _shared_limiter(settings.groq_requests_per_minute, settings.groq_max_limiter_wait_seconds)

    model_a = settings.groq_text_model
    model_b = "openai/gpt-oss-20b" if model_a != "openai/gpt-oss-20b" else "openai/gpt-oss-120b"
    settings.groq_agent_models = {**settings.groq_agent_models, "smoke_b": model_b}
    llm_a = GroqProvider(settings)
    llm_b = GroqProvider(settings, agent="smoke_b")
    prompt = "Explain clause 8 in one plain sentence for a tenant. Cite its clause id as 'c-8'."

    print(f"\n== 1. text / {model_a}")
    claim = await llm_a.generate_structured(prompt, Claim, context=SAMPLE, max_tokens=512)
    print("   ->", claim.model_dump())
    a1 = limiter.snapshot(model_a)
    print("   limits:", a1)

    print(f"\n== 2. text / {model_b}")
    claim_b = await llm_b.generate_structured(prompt, Claim, context=SAMPLE, max_tokens=512)
    print("   ->", claim_b.model_dump())
    b1 = limiter.snapshot(model_b)
    print("   limits:", b1)

    print(f"\n== 3. text / {model_a} again")
    await llm_a.generate_structured(prompt, Claim, context=SAMPLE, max_tokens=512)
    a2 = limiter.snapshot(model_a)
    print("   limits:", a2)

    print("\n== Rate-limit scope")
    print(f"   {model_a}: requests_remaining {a1['requests_remaining']} -> {a2['requests_remaining']}")
    print(f"   {model_b}: requests_remaining {b1['requests_remaining']}")
    if a1["requests_remaining"] is not None and a2["requests_remaining"] is not None:
        drop = a1["requests_remaining"] - a2["requests_remaining"]
        verdict = "SHARED across models" if drop >= 2 else "PER MODEL"
        print(f"   {model_a} daily budget dropped by {drop} across calls 2-3 -> {verdict}")

    vision = settings.groq_vision_model
    if not vision:
        print("\n== 4. vision skipped: GROQ_VISION_MODEL not set")
        return 0
    print(f"\n== 4. image / {vision}")
    facts = await llm_a.generate_structured(
        "Describe this image.",
        ImageFacts,
        images=[ImageInput(data=red_square_png(), media_type="image/png")],
        max_tokens=512,
    )
    print("   ->", facts.model_dump())
    print("   limits:", limiter.snapshot(vision))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
