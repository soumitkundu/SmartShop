import os


def getenv(name: str, default: str | None = None) -> str | None:
    """
    Small env loader wrapper.

    Keeps Phase 0 lightweight (no extra dependencies like pydantic-settings).
    """

    val = os.getenv(name)
    if val is None or val == "":
        return default
    return val


class Settings:
    APP_NAME: str = getenv("APP_NAME", "SmartShop") or "SmartShop"

    # RAG / retrieval knobs (used later)
    TOP_K_RESULTS: int = int(getenv("TOP_K_RESULTS", "5") or 5)

    # Phase 6 — bounded conversation memory (user+assistant turn pairs)
    MEMORY_WINDOW_TURNS: int = int(getenv("MEMORY_WINDOW_TURNS", "6") or 6)

    # Phase 6 — multimodal rank fusion (reciprocal rank fusion)
    FUSION_RRF_K: int = int(getenv("FUSION_RRF_K", "60") or 60)
    FUSION_TEXT_WEIGHT: float = float(getenv("FUSION_TEXT_WEIGHT", "1.0") or 1.0)
    FUSION_IMAGE_WEIGHT: float = float(getenv("FUSION_IMAGE_WEIGHT", "1.0") or 1.0)
    CHROMA_PATH: str = getenv("CHROMA_PATH", "./chroma_db") or "./chroma_db"
    IMAGE_COLLECTION_NAME: str = getenv("IMAGE_COLLECTION_NAME", "shopify_images") or "shopify_images"
    PRODUCT_CATALOG_PATH: str = getenv("PRODUCT_CATALOG_PATH", "./data/products.json") or "./data/products.json"
    TEXT_INDEX_PATH: str = getenv("TEXT_INDEX_PATH", "./data/text_index.json") or "./data/text_index.json"

    # Whisper model name (Phase 4)
    WHISPER_MODEL: str = getenv("WHISPER_MODEL", "base") or "base"

    # CLIP model name (Phase 5)
    CLIP_MODEL: str = getenv("CLIP_MODEL", "ViT-B/32") or "ViT-B/32"

    # LLM provider keys (used later)
    GROQ_API_KEY: str | None = getenv("GROQ_API_KEY")
    GOOGLE_API_KEY: str | None = getenv("GOOGLE_API_KEY")
    LLM_PROVIDER: str = (getenv("LLM_PROVIDER", "groq") or "groq").lower()
    GROQ_MODEL: str = getenv("GROQ_MODEL", "llama-3.1-8b-instant") or "llama-3.1-8b-instant"
    GEMINI_MODEL: str = getenv("GEMINI_MODEL", "gemini-1.5-flash") or "gemini-1.5-flash"

    # LangSmith observability (Phase 3+)
    LANGCHAIN_TRACING_V2: bool = (getenv("LANGCHAIN_TRACING_V2", "true") or "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    LANGCHAIN_API_KEY: str | None = getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT: str = getenv("LANGCHAIN_PROJECT", "SmartShop") or "SmartShop"


settings = Settings()

