"""Selects the LLM provider named by ``LLM_PROVIDER``.

To add a provider: implement ``LLMProvider`` in a new module in this package and add
one entry to ``_PROVIDERS``. Agent code does not change.
"""

import inspect
from functools import lru_cache
from importlib import import_module

from app.core.config import Settings, get_settings
from app.core.llm.base import LLMError, LLMProvider

# name -> "module:Class". Imported lazily so an unused provider's SDK need not be installed.
_PROVIDERS: dict[str, str] = {
    "anthropic": "app.core.llm.anthropic_provider:AnthropicProvider",
    "groq": "app.core.llm.groq_provider:GroqProvider",
}


def available_providers() -> list[str]:
    return sorted(_PROVIDERS)


def create_provider(settings: Settings, agent: str | None = None) -> LLMProvider:
    """Build the configured provider.

    ``agent`` lets providers that support per-agent model overrides (e.g.
    ``GROQ_AGENT_MODELS``) pick that agent's model; other providers ignore it.
    """
    key = settings.llm_provider.strip().lower()
    target = _PROVIDERS.get(key)
    if target is None:
        raise LLMError(f"Unknown LLM_PROVIDER '{settings.llm_provider}'. Available: {', '.join(available_providers())}")
    module_name, class_name = target.split(":")
    provider_cls = getattr(import_module(module_name), class_name)
    if agent is not None and "agent" in inspect.signature(provider_cls).parameters:
        return provider_cls(settings, agent=agent)
    return provider_cls(settings)


@lru_cache
def get_llm_provider(agent: str | None = None) -> LLMProvider:
    """Process-wide provider instance per agent (use as a FastAPI dependency or call directly).

    Agents pass their own name, e.g. ``get_llm_provider("classifier")``.
    """
    return create_provider(get_settings(), agent)
