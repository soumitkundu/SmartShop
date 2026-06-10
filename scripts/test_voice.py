import argparse
import importlib
import logging
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TEXT_ONLY_TRACE = ["router", "fuser", "retriever", "generator"]
VOICE_TRACE = ["router", "voice", "fuser", "retriever", "generator"]


def _check_file(path: str) -> None:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Missing required file: {p}")


def _maybe_build_index(products_path: str, index_path: str) -> None:
    if Path(index_path).exists():
        return
    print(f"Index not found at {index_path}, building it now...")
    cmd = [
        sys.executable,
        "scripts/embed_products.py",
        "--input",
        products_path,
        "--out",
        index_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"Failed to build index:\n{result.stdout}\n{result.stderr}")
    if result.stdout:
        print(result.stdout.strip())


def _reload_backend_modules() -> None:
    modules = (
        "backend.config",
        "backend.processors.voice",
        "backend.graph.nodes",
        "backend.graph.agent",
        "backend.main",
    )
    for name in modules:
        if name in sys.modules:
            importlib.reload(sys.modules[name])


def _post_search(client: TestClient, data: dict, files: dict | None = None) -> dict:
    resp = client.post("/api/search", data=data, files=files or {})
    if resp.status_code != 200:
        raise SystemExit(f"/api/search failed: {resp.status_code} {resp.text}")
    return resp.json()


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4 voice input test")
    parser.add_argument("--query", default="show me walking shoes under 2000")
    parser.add_argument(
        "--audio-file",
        help="Path to WAV/MP3/OGG audio for voice-only or text+voice tests",
    )
    parser.add_argument(
        "--text-with-audio",
        default="budget option",
        help="Optional text to send together with --audio-file",
    )
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    products_path = os.getenv("PRODUCT_CATALOG_PATH", "./data/products.json")
    index_path = os.getenv("TEXT_INDEX_PATH", "./data/text_index.json")

    _check_file(products_path)
    _maybe_build_index(products_path, index_path)
    _reload_backend_modules()

    from backend.config import settings  # pylint: disable=import-outside-toplevel
    from backend.main import app  # pylint: disable=import-outside-toplevel

    print(f"Whisper model: {settings.WHISPER_MODEL}")

    client = TestClient(app)

    text_json = _post_search(
        client,
        {"text": args.query, "session_id": "phase4-text-session"},
    )
    if text_json.get("node_trace") != TEXT_ONLY_TRACE:
        raise SystemExit(f"Unexpected text-only trace: {text_json.get('node_trace')}")
    if text_json.get("rejected"):
        raise SystemExit("Text-only catalog query was unexpectedly rejected.")
    print("Text-only /api/search: PASS")
    print(f"Node trace: {' -> '.join(text_json['node_trace'])}")

    if not args.audio_file:
        print("No --audio-file provided; skipping voice integration checks.")
        print("Provide a short spoken product query (e.g. WAV) to validate transcription.")
        print("Phase 4 text regression passed.")
        return 0

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    with audio_path.open("rb") as audio_fp:
        voice_json = _post_search(
            client,
            {"session_id": "phase4-voice-session"},
            files={"audio_file": (audio_path.name, audio_fp, "audio/wav")},
        )

    if voice_json.get("node_trace") != VOICE_TRACE:
        raise SystemExit(f"Unexpected voice-only trace: {voice_json.get('node_trace')}")
    if not voice_json.get("transcribed_text"):
        raise SystemExit("Voice-only request did not return transcribed_text.")
    print("Voice-only /api/search: PASS")
    print(f"Transcription: {voice_json['transcribed_text']}")
    print(f"Fused query: {voice_json.get('fused_query')}")
    print(f"Node trace: {' -> '.join(voice_json['node_trace'])}")

    with audio_path.open("rb") as audio_fp:
        combo_json = _post_search(
            client,
            {
                "text": args.text_with_audio,
                "session_id": "phase4-combo-session",
            },
            files={"audio_file": (audio_path.name, audio_fp, "audio/wav")},
        )

    if combo_json.get("node_trace") != VOICE_TRACE:
        raise SystemExit(f"Unexpected text+voice trace: {combo_json.get('node_trace')}")
    fused = combo_json.get("fused_query") or ""
    if args.text_with_audio not in fused:
        raise SystemExit(f"Text+voice fused_query missing typed text: {fused!r}")
    if not combo_json.get("transcribed_text"):
        raise SystemExit("Text+voice request did not return transcribed_text.")
    print("Text+voice /api/search: PASS")
    print(f"Fused query: {fused}")
    print("All Phase 4 voice checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
