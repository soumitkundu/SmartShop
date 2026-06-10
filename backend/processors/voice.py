import logging
import os
import tempfile

import whisper

from backend.config import settings

logger = logging.getLogger(__name__)

_model: whisper.Whisper | None = None

SUPPORTED_AUDIO_SUFFIXES = {".wav", ".mp3", ".ogg", ".m4a", ".webm", ".flac", ".mpeg", ".mp4"}


class TranscriptionError(Exception):
    """Raised when Whisper cannot transcribe the provided audio."""


def get_model() -> whisper.Whisper:
    """Load and cache the Whisper model (downloaded once on first use)."""
    global _model
    if _model is None:
        model_name = settings.WHISPER_MODEL
        logger.info("Loading Whisper model=%s", model_name)
        _model = whisper.load_model(model_name)
    return _model


def normalize_audio_suffix(suffix: str | None) -> str:
    """Ensure suffix is a supported extension for ffmpeg/Whisper."""
    if not suffix:
        return ".wav"
    normalized = suffix if suffix.startswith(".") else f".{suffix}"
    normalized = normalized.lower()
    if normalized not in SUPPORTED_AUDIO_SUFFIXES:
        return ".wav"
    return normalized


def transcribe(audio_bytes: bytes, suffix: str = ".wav") -> str:
    """Accept raw audio bytes and return transcribed text."""
    if not audio_bytes:
        raise TranscriptionError("Audio file is empty.")

    normalized_suffix = normalize_audio_suffix(suffix)
    tmp_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(suffix=normalized_suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        result = get_model().transcribe(tmp_path)
        text = (result.get("text") or "").strip()
        if not text:
            raise TranscriptionError(
                "No speech detected in the audio. Try a clearer recording or a supported format."
            )
        return text
    except TranscriptionError:
        raise
    except Exception as exc:
        raise TranscriptionError(
            f"Audio transcription failed: {exc}. "
            "Ensure ffmpeg is installed and the file is a valid audio format."
        ) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
