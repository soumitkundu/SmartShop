import logging
from io import BytesIO

import clip
import torch
from PIL import Image

from backend.config import settings

logger = logging.getLogger(__name__)

_model: torch.nn.Module | None = None
_preprocess = None

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}


class ImageEncodingError(Exception):
    """Raised when CLIP cannot encode the provided image."""


def get_clip() -> tuple[torch.nn.Module, object]:
    """Load and cache the CLIP model (downloaded once on first use)."""
    global _model, _preprocess
    if _model is None:
        device = "cpu"
        logger.info("Loading CLIP model=%s device=%s", settings.CLIP_MODEL, device)
        _model, _preprocess = clip.load(settings.CLIP_MODEL, device=device)
        _model.eval()
    return _model, _preprocess


def normalize_image_suffix(suffix: str | None) -> str:
    """Ensure suffix is a supported image extension."""
    if not suffix:
        return ".jpg"
    normalized = suffix if suffix.startswith(".") else f".{suffix}"
    normalized = normalized.lower()
    if normalized not in SUPPORTED_IMAGE_SUFFIXES:
        return ".jpg"
    return normalized


def encode_image(image_bytes: bytes) -> list[float]:
    """Accept raw image bytes and return a normalized CLIP embedding vector."""
    if not image_bytes:
        raise ImageEncodingError("Image file is empty.")

    try:
        model, preprocess = get_clip()
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            vec = model.encode_image(tensor)
            vec = vec / vec.norm(dim=-1, keepdim=True)
        return vec.cpu().numpy().tolist()[0]
    except ImageEncodingError:
        raise
    except Exception as exc:
        raise ImageEncodingError(
            f"Image encoding failed: {exc}. "
            "Ensure the file is a valid JPEG, PNG, or WebP image."
        ) from exc
