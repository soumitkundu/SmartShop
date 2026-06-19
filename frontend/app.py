import io
import os
import wave
from pathlib import Path
from typing import Any
from uuid import uuid4

import chainlit as cl
import httpx
from dotenv import load_dotenv

load_dotenv()

BACKEND_SEARCH_URL = os.getenv("BACKEND_SEARCH_URL", "http://127.0.0.1:8781/api/search")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("CHAINLIT_REQUEST_TIMEOUT_SECONDS", "90"))
SHOPIFY_STORE_DOMAIN = (os.getenv("SHOPIFY_STORE_DOMAIN", "") or "").strip()
AUDIO_SAMPLE_RATE = int(os.getenv("CHAINLIT_AUDIO_SAMPLE_RATE", "44100"))
MIN_VOICE_RECORDING_SECONDS = float(os.getenv("CHAINLIT_MIN_VOICE_RECORDING_SECONDS", "0.5"))


def _clean_store_domain(raw_domain: str) -> str:
    domain = raw_domain.strip().replace("https://", "").replace("http://", "")
    return domain.strip("/")


def _build_shopify_product_url(product: dict[str, Any]) -> str | None:
    handle = (product.get("handle") or "").strip()
    if not handle:
        return None
    domain = _clean_store_domain(SHOPIFY_STORE_DOMAIN)
    if not domain:
        return None
    return f"https://{domain}/products/{handle}"


def _format_price(value: Any) -> str:
    try:
        return f"Rs. {float(value):,.2f}"
    except Exception:
        return "N/A"


def _is_audio_file(path: str) -> bool:
    ext = Path(path).suffix.lower()
    return ext in {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".webm"}


def _is_image_file(path: str) -> bool:
    ext = Path(path).suffix.lower()
    return ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


async def _call_backend(
    *,
    session_id: str,
    text: str | None,
    audio_path: str | None,
    audio_bytes: bytes | None,
    audio_name: str,
    image_path: str | None,
) -> dict[str, Any]:
    data = {"session_id": session_id}
    if text:
        data["text"] = text

    files: dict[str, tuple[str, bytes, str]] = {}
    if audio_path:
        audio_bytes = Path(audio_path).read_bytes()
        audio_name = Path(audio_path).name
    if audio_bytes:
        files["audio_file"] = (audio_name, audio_bytes, "audio/wav")
    if image_path:
        image_bytes = Path(image_path).read_bytes()
        files["image_file"] = (Path(image_path).name, image_bytes, "application/octet-stream")

    timeout = httpx.Timeout(timeout=REQUEST_TIMEOUT_SECONDS)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(BACKEND_SEARCH_URL, data=data, files=files or None)
        response.raise_for_status()
        return response.json()


def _build_wav_from_pcm_chunks(chunks: list[bytes], sample_rate: int = AUDIO_SAMPLE_RATE) -> bytes:
    pcm_data = b"".join(chunks)
    if not pcm_data:
        raise ValueError("No audio data captured.")

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)

    frames = len(pcm_data) // 2
    duration_seconds = frames / float(sample_rate)
    if duration_seconds < MIN_VOICE_RECORDING_SECONDS:
        raise ValueError(
            f"Recording is too short ({duration_seconds:.1f}s). "
            f"Please speak for at least {MIN_VOICE_RECORDING_SECONDS:.1f}s."
        )

    return wav_buffer.getvalue()


def _media_paths_from_elements(elements: list[Any] | None) -> tuple[str | None, str | None]:
    audio_path: str | None = None
    image_path: str | None = None

    for element in elements or []:
        path = getattr(element, "path", None)
        if not path:
            continue
        if audio_path is None and _is_audio_file(path):
            audio_path = path
            continue
        if image_path is None and _is_image_file(path):
            image_path = path

    return audio_path, image_path


async def _run_search(
    *,
    session_id: str,
    text: str | None,
    audio_path: str | None = None,
    audio_bytes: bytes | None = None,
    audio_name: str = "recording.wav",
    image_path: str | None = None,
) -> None:
    thinking = cl.Message(content="Searching your catalog...")
    await thinking.send()

    try:
        payload = await _call_backend(
            session_id=session_id,
            text=text,
            audio_path=audio_path,
            audio_bytes=audio_bytes,
            audio_name=audio_name,
            image_path=image_path,
        )
    except httpx.TimeoutException:
        await thinking.remove()
        await cl.Message(
            content="The request timed out while contacting the backend. Please try again in a moment."
        ).send()
        return
    except httpx.HTTPStatusError as exc:
        await thinking.remove()
        detail = exc.response.text
        await cl.Message(content=f"Backend returned an error: {detail}").send()
        return
    except Exception as exc:
        await thinking.remove()
        await cl.Message(content=f"Unexpected error: {exc}").send()
        return

    await thinking.remove()

    transcribed_text = (payload.get("transcribed_text") or "").strip()
    if transcribed_text:
        await cl.Message(content=f"**Voice query:** {transcribed_text}").send()

    answer = payload.get("answer") or "I found results, but could not generate a full answer."
    await cl.Message(content=answer).send()

    products = payload.get("products") or []
    if products:
        cards_html = _render_product_cards(products)
        await cl.Message(content=cards_html).send()
    else:
        await cl.Message(content="No matching products found in the store catalog for this request.").send()


def _render_product_cards(products: list[dict[str, Any]]) -> str:
    cards: list[str] = []
    for item in products:
        title = item.get("title") or "Untitled product"
        vendor = item.get("vendor") or "Unknown brand"
        price = _format_price(item.get("price"))
        stock = item.get("inventory_quantity")
        stock_label = str(stock) if stock is not None else "N/A"
        image_url = (item.get("image_url") or "").strip()
        product_url = _build_shopify_product_url(item)
        handle = (item.get("handle") or "").strip()

        image_block = (
            f'<img src="{image_url}" alt="{title}" '
            'style="width:100%;height:180px;object-fit:cover;border-radius:10px;" />'
            if image_url
            else '<div style="height:180px;background:#f3f4f6;border-radius:10px;"></div>'
        )
        link_block = (
            f'<a href="{product_url}" target="_blank" rel="noopener noreferrer" '
            'style="display:inline-block;margin-top:10px;padding:8px 12px;'
            'background:#111827;color:#ffffff;text-decoration:none;border-radius:8px;">'
            "View on Store</a>"
            if product_url
            else '<span style="display:inline-block;margin-top:10px;color:#6b7280;">'
            "Store link unavailable</span>"
        )

        cards.append(
            (
                '<div style="border:1px solid #e5e7eb;border-radius:12px;padding:12px;'
                'margin:10px 0;background:#ffffff;">'
                f"{image_block}"
                f'<h4 style="margin:10px 0 6px 0;">{title}</h4>'
                f'<p style="margin:0;color:#6b7280;"><strong>Brand:</strong> {vendor}</p>'
                f'<p style="margin:4px 0;color:#6b7280;"><strong>Price:</strong> {price}</p>'
                f'<p style="margin:4px 0;color:#6b7280;"><strong>In stock:</strong> {stock_label}</p>'
                f'<p style="margin:4px 0;color:#6b7280;"><strong>Handle:</strong> {handle or "N/A"}</p>'
                f"{link_block}"
                "</div>"
            )
        )
    return "\n".join(cards)


@cl.on_chat_start
async def on_chat_start() -> None:
    session_id = str(uuid4())
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("audio_chunks", [])
    await cl.Message(
        content=(
            "SmartShop assistant is ready.\n\n"
            "- Ask with text\n"
            "- Click the **microphone** in the input bar to record a voice query\n"
            "- Attach an image for visual matching\n"
            "- Attach an audio file for voice query\n\n"
            "When similar products are found, rich product cards will appear with links to your Shopify store."
        )
    ).send()


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk) -> None:
    if chunk.isStart:
        cl.user_session.set("audio_chunks", [])

    if not chunk.data:
        return

    audio_chunks: list[bytes] = cl.user_session.get("audio_chunks") or []
    audio_chunks.append(chunk.data)
    cl.user_session.set("audio_chunks", audio_chunks)


@cl.on_audio_end
async def on_audio_end(elements: list[Any]) -> None:
    session_id = cl.user_session.get("session_id") or str(uuid4())
    cl.user_session.set("session_id", session_id)

    audio_chunks: list[bytes] = cl.user_session.get("audio_chunks") or []
    cl.user_session.set("audio_chunks", [])

    attached_audio_path, image_path = _media_paths_from_elements(elements)

    try:
        recorded_audio = _build_wav_from_pcm_chunks(audio_chunks) if audio_chunks else None
    except ValueError as exc:
        await cl.Message(content=str(exc)).send()
        return

    if not any([recorded_audio, attached_audio_path, image_path]):
        await cl.Message(
            content="No voice input detected. Click the microphone, speak your query, then stop recording."
        ).send()
        return

    await _run_search(
        session_id=session_id,
        text=None,
        audio_path=attached_audio_path if not recorded_audio else None,
        audio_bytes=recorded_audio,
        image_path=image_path,
    )


@cl.on_message
async def on_message(message: cl.Message) -> None:
    session_id = cl.user_session.get("session_id") or str(uuid4())
    cl.user_session.set("session_id", session_id)

    text = (message.content or "").strip() or None
    audio_path: str | None = None
    image_path: str | None = None

    for element in message.elements or []:
        path = getattr(element, "path", None)
        if not path:
            continue
        if audio_path is None and _is_audio_file(path):
            audio_path = path
            continue
        if image_path is None and _is_image_file(path):
            image_path = path

    if not any([text, audio_path, image_path]):
        await cl.Message(
            content="Please send text, or attach an audio/image file so I can search the Shopify catalog."
        ).send()
        return

    await _run_search(
        session_id=session_id,
        text=text,
        audio_path=audio_path,
        image_path=image_path,
    )
