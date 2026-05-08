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
    CHROMA_PATH: str = getenv("CHROMA_PATH", "./chroma_db") or "./chroma_db"
    PRODUCT_CATALOG_PATH: str = getenv("PRODUCT_CATALOG_PATH", "./data/products.json") or "./data/products.json"
    TEXT_INDEX_PATH: str = getenv("TEXT_INDEX_PATH", "./data/text_index.json") or "./data/text_index.json"

    # Whisper model name (used later)
    WHISPER_MODEL: str = getenv("WHISPER_MODEL", "base") or "base"

    # LLM provider keys (used later)
    GROQ_API_KEY: str | None = getenv("GROQ_API_KEY")
    GOOGLE_API_KEY: str | None = getenv("GOOGLE_API_KEY")
    LLM_PROVIDER: str = (getenv("LLM_PROVIDER", "groq") or "groq").lower()
    GROQ_MODEL: str = getenv("GROQ_MODEL", "llama-3.1-8b-instant") or "llama-3.1-8b-instant"
    GEMINI_MODEL: str = getenv("GEMINI_MODEL", "gemini-1.5-flash") or "gemini-1.5-flash"


settings = Settings()

