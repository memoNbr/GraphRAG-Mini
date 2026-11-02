"""
One place to choose the LLM, shared by kg_builder.py and agent.py.

Why this exists: Mistral's free tier was exhausted mid-build (429, code
1300), and no amount of throttling fixes an empty daily pool. The provider
has to be switchable without editing two scripts.

Set LLM_PROVIDER in .env:

    groq     verified working, free, generous limits. Recommended.
    mistral  the original. Free tier is easy to exhaust.
    ollama   fully local, truly unlimited, no key. Needs `ollama serve`.

The embedding model is NOT switched here. Embeddings always run locally via
sentence-transformers, so they never consume API quota regardless of provider.
"""

import os

from dotenv import load_dotenv


def _load_env():
    """Find .env from this file's location, not the current directory."""
    from pathlib import Path

    here = Path(__file__).resolve().parent
    for folder in [here, *here.parents]:
        candidate = folder / ".env"
        if candidate.exists():
            load_dotenv(candidate)
            return candidate
    load_dotenv()
    return None


_load_env()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Stronger model for extraction, which needs reliable JSON over long chunks.
DEFAULT_BUILD_MODEL = "openai/gpt-oss-120b"
DEFAULT_CHAT_MODEL = "openai/gpt-oss-20b"


def provider() -> str:
    return (os.getenv("LLM_PROVIDER") or "groq").strip().lower()


def min_interval_seconds() -> float:
    """Seconds between LLM calls, per provider.

    Override with LLM_MIN_INTERVAL in .env. 0 disables throttling (only
    sensible for ollama, which is local).

      groq    8000 TOKENS/minute (measured, see below) -> ~30s apart
      mistral free tier is a global 1 request/second -> ~1.05s apart
      ollama  local, no limit -> 0

    Groq's limit is tokens, not requests. Read from x-ratelimit-limit-tokens
    on the live API: every usable chat model on this key is capped at 8000
    TPM (gpt-oss-120b, gpt-oss-20b, qwen3.8-27b). The request limit is 1000/min
    and never binds.

    Measured on real CoALA chunks with gpt-oss-120b:
        prompt ~1500 tokens, completion ~400-1500, total ~1900-3500 per chunk.
        Reasoning tokens count against TPM, so a model that thinks before it
        answers costs more than its visible output suggests.

    The binding constraint is the rolling 60s window: if requests are T
    seconds apart, a window holds 60/T of them, so we need
        (60 / T) * per_chunk <= 8000
    With a worst case of ~3500 tokens/chunk that gives T >= 26s. 30s leaves
    margin for unusually dense chunks. The earlier value of 2.0s was derived
    from a requests/minute assumption and was ~15x too fast, which is what
    produced the wall of 429s.

    At 30s, CoALA (31 chunks) takes ~16 minutes.
    """
    raw = os.getenv("LLM_MIN_INTERVAL")
    if raw is not None and raw.strip() != "":
        return float(raw)
    return {"groq": 30.0, "mistral": 1.05, "ollama": 0.0}.get(provider(), 30.0)


def build_llm(rate_limit_handler=None):
    """Return an LLMInterface for entity/relation extraction (kg_builder.py)."""
    which = provider()

    if which == "groq":
        from neo4j_graphrag.llm import OpenAILLM

        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise SystemExit("LLM_PROVIDER=groq but GROQ_API_KEY is not set.")
        return OpenAILLM(
            model_name=os.getenv("KG_LLM_MODEL") or DEFAULT_BUILD_MODEL,
            api_key=key,
            base_url=GROQ_BASE_URL,
            rate_limit_handler=rate_limit_handler,
        )

    if which == "mistral":
        from neo4j_graphrag.llm import MistralAILLM

        key = os.getenv("MISTRAL_API_KEY")
        if not key:
            raise SystemExit("LLM_PROVIDER=mistral but MISTRAL_API_KEY is not set.")
        return MistralAILLM(
            model_name=os.getenv("KG_LLM_MODEL") or "mistral-small-latest",
            api_key=key,
            rate_limit_handler=rate_limit_handler,
        )

    if which == "ollama":
        try:
            from neo4j_graphrag.llm import OllamaLLM
        except ImportError as exc:
            raise SystemExit(
                "LLM_PROVIDER=ollama needs the ollama client package:\n"
                "    pip install ollama\n"
                f"(underlying import error: {exc})"
            )
        base_url = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
        # The ollama client reads OLLAMA_HOST natively.
        os.environ["OLLAMA_HOST"] = base_url
        return OllamaLLM(
            model_name=os.getenv("KG_LLM_MODEL") or "qwen3:4b",
            model_params={"num_ctx": int(os.getenv("OLLAMA_NUM_CTX") or 8192)},
            rate_limit_handler=rate_limit_handler,
        )

    raise SystemExit(
        f"Unknown LLM_PROVIDER={which!r}. Use groq, mistral or ollama."
    )


def chat_model_kwargs() -> tuple[str, dict]:
    """(model_name, kwargs) for langchain init_chat_model, in agent.py."""
    which = provider()

    if which == "groq":
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise SystemExit("LLM_PROVIDER=groq but GROQ_API_KEY is not set.")
        return (
            os.getenv("CHAT_LLM_MODEL") or DEFAULT_CHAT_MODEL,
            {"model_provider": "openai", "api_key": key, "base_url": GROQ_BASE_URL},
        )

    if which == "mistral":
        key = os.getenv("MISTRAL_API_KEY")
        if not key:
            raise SystemExit("LLM_PROVIDER=mistral but MISTRAL_API_KEY is not set.")
        return (
            os.getenv("CHAT_LLM_MODEL") or "mistral-small-latest",
            {"model_provider": "mistralai", "api_key": key},
        )

    if which == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
        return (
            os.getenv("CHAT_LLM_MODEL") or "qwen3:4b",
            {"model_provider": "openai", "base_url": base_url.rstrip("/") + "/v1"},
        )

    raise SystemExit(
        f"Unknown LLM_PROVIDER={which!r}. Use groq, mistral or ollama."
    )
