"""Each model SDK may be imported only by its own provider module; everything else uses LLMProvider."""

import ast
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
LLM_DIR = APP_DIR / "core" / "llm"

# SDK top-level module -> the only file allowed to import it.
SDK_OWNERS: dict[str, Path] = {
    "anthropic": LLM_DIR / "anthropic_provider.py",
    "openai": LLM_DIR / "groq_provider.py",
    "groq": LLM_DIR / "groq_provider.py",
}


def _imported_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("sdk", sorted(SDK_OWNERS))
def test_sdk_imported_only_by_its_provider(sdk: str) -> None:
    offenders = [
        str(path.relative_to(APP_DIR))
        for path in APP_DIR.rglob("*.py")
        if path != SDK_OWNERS[sdk] and sdk in _imported_roots(path)
    ]
    assert offenders == []
