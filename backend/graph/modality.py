"""Helpers for detecting and describing multimodal request combinations."""

from __future__ import annotations

from typing import TypedDict


class ModalityFlags(TypedDict):
    has_text: bool
    has_audio: bool
    has_image: bool


# All seven supported input combinations (Phase 6).
MODALITY_COMBOS: tuple[str, ...] = (
    "text",
    "voice",
    "image",
    "text+voice",
    "text+image",
    "voice+image",
    "text+voice+image",
)


def detect_modalities(
    text_input: str | None,
    audio_bytes: bytes | None,
    image_bytes: bytes | None,
) -> ModalityFlags:
    return {
        "has_text": bool(text_input and text_input.strip()),
        "has_audio": bool(audio_bytes),
        "has_image": bool(image_bytes),
    }


def modality_label(flags: ModalityFlags) -> str:
    parts: list[str] = []
    if flags["has_text"]:
        parts.append("text")
    if flags["has_audio"]:
        parts.append("voice")
    if flags["has_image"]:
        parts.append("image")
    return "+".join(parts) if parts else "none"


def expected_graph_trace(flags: ModalityFlags) -> list[str]:
    """Return the node_trace we expect for a given modality combination."""
    trace = ["router"]
    if flags["has_audio"]:
        trace.append("voice")
    if flags["has_image"]:
        trace.append("image")
    trace.extend(["fuser", "retriever", "generator"])
    return trace
