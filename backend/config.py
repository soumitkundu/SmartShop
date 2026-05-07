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

    # Whisper model name (used later)
    WHISPER_MODEL: str = getenv("WHISPER_MODEL", "base") or "base"

    # LLM provider keys (used later)
    GROQ_API_KEY: str | None = getenv("GROQ_API_KEY")
    GOOGLE_API_KEY: str | None = getenv("GOOGLE_API_KEY")


settings = Settings()

