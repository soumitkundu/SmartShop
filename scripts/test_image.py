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
IMAGE_ONLY_TRACE = ["router", "image", "fuser", "retriever", "generator"]
TEXT_IMAGE_TRACE = IMAGE_ONLY_TRACE
VOICE_IMAGE_TRACE = ["router", "voice", "image", "fuser", "retriever", "generator"]


def _check_file(path: str) -> None:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Missing required file: {p}")


def _maybe_build_text_index(products_path: str, index_path: str) -> None:
    if Path(index_path).exists():
        return
    print(f"Text index not found at {index_path}, building it now...")
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
        raise SystemExit(f"Failed to build text index:\n{result.stdout}\n{result.stderr}")
    if result.stdout:
        print(result.stdout.strip())


def _maybe_build_image_index(products_path: str, chroma_path: str) -> None:
    report_path = Path(chroma_path) / "image_index_report.json"
    if report_path.exists():
        return
    print(f"Image index not found in {chroma_path}, building it now...")
    cmd = [
        sys.executable,
        "scripts/embed_product_images.py",
        "--input",
        products_path,
        "--chroma-path",
        chroma_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"Failed to build image index:\n{result.stdout}\n{result.stderr}")
    if result.stdout:
        print(result.stdout.strip())


def _reload_backend_modules() -> None:
    modules = (
        "backend.config",
        "backend.processors.image",
        "backend.rag.retriever",
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
    parser = argparse.ArgumentParser(description="Phase 5 image retrieval test")
    parser.add_argument("--query", default="show me walking shoes under 2000")
    parser.add_argument(
        "--image-file",
        help="Path to JPEG/PNG/WebP image for image-only or text+image tests",
    )
    parser.add_argument(
        "--audio-file",
        help="Optional audio file for voice+image trace validation",
    )
    parser.add_argument(
        "--text-with-image",
        default="similar style",
        help="Optional text to send together with --image-file",
    )
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    products_path = os.getenv("PRODUCT_CATALOG_PATH", "./data/products.json")
    index_path = os.getenv("TEXT_INDEX_PATH", "./data/text_index.json")
    chroma_path = os.getenv("CHROMA_PATH", "./chroma_db")

    _check_file(products_path)
    _maybe_build_text_index(products_path, index_path)
    _maybe_build_image_index(products_path, chroma_path)
    _reload_backend_modules()

    from backend.config import settings  # pylint: disable=import-outside-toplevel
    from backend.main import app  # pylint: disable=import-outside-toplevel
    from backend.processors.image import encode_image  # pylint: disable=import-outside-toplevel
    from backend.rag.retriever import ImageRetriever  # pylint: disable=import-outside-toplevel

    print(f"CLIP model: {settings.CLIP_MODEL}")
    print(f"Image collection: {settings.IMAGE_COLLECTION_NAME}")

    image_retriever = ImageRetriever(settings.CHROMA_PATH, settings.IMAGE_COLLECTION_NAME)
    client = TestClient(app)

    text_json = _post_search(
        client,
        {"text": args.query, "session_id": "phase5-text-session"},
    )
    if text_json.get("node_trace") != TEXT_ONLY_TRACE:
        raise SystemExit(f"Unexpected text-only trace: {text_json.get('node_trace')}")
    print("Text-only regression: PASS")

    if not args.image_file:
        print("No --image-file provided; skipping image integration checks.")
        print("Provide a product photo to validate CLIP retrieval.")
        print("Phase 5 text regression passed.")
        return 0

    image_path = Path(args.image_file)
    if not image_path.exists():
        raise SystemExit(f"Image file not found: {image_path}")

    with image_path.open("rb") as image_fp:
        image_bytes = image_fp.read()

    vec = encode_image(image_bytes)
    direct_hits = image_retriever.search(vec, top_k=5)
    if not direct_hits:
        raise SystemExit("ImageRetriever returned no hits for the provided image.")
    print("ImageRetriever direct search: PASS")
    print(f"Top visual hit: {direct_hits[0].get('title')} | score={direct_hits[0].get('score')}")

    with image_path.open("rb") as image_fp:
        image_json = _post_search(
            client,
            {"session_id": "phase5-image-session"},
            files={"image_file": (image_path.name, image_fp, "image/jpeg")},
        )

    if image_json.get("node_trace") != IMAGE_ONLY_TRACE:
        raise SystemExit(f"Unexpected image-only trace: {image_json.get('node_trace')}")
    if image_json.get("rejected"):
        raise SystemExit("Image-only catalog query was unexpectedly rejected.")
    if not image_json.get("products"):
        raise SystemExit("Image-only request returned empty products.")
    print("Image-only /api/search: PASS")
    print(f"Node trace: {' -> '.join(image_json['node_trace'])}")

    with image_path.open("rb") as image_fp:
        combo_json = _post_search(
            client,
            {
                "text": args.text_with_image,
                "session_id": "phase5-text-image-session",
            },
            files={"image_file": (image_path.name, image_fp, "image/jpeg")},
        )

    if combo_json.get("node_trace") != TEXT_IMAGE_TRACE:
        raise SystemExit(f"Unexpected text+image trace: {combo_json.get('node_trace')}")
    fused = combo_json.get("fused_query") or ""
    if args.text_with_image not in fused:
        raise SystemExit(f"Text+image fused_query missing typed text: {fused!r}")
    if not combo_json.get("products"):
        raise SystemExit("Text+image request returned empty products.")
    print("Text+image /api/search: PASS")
    print(f"Fused query: {fused}")

    if args.audio_file:
        audio_path = Path(args.audio_file)
        if not audio_path.exists():
            raise SystemExit(f"Audio file not found: {audio_path}")

        with image_path.open("rb") as image_fp, audio_path.open("rb") as audio_fp:
            voice_image_json = _post_search(
                client,
                {"session_id": "phase5-voice-image-session"},
                files={
                    "image_file": (image_path.name, image_fp, "image/jpeg"),
                    "audio_file": (audio_path.name, audio_fp, "audio/wav"),
                },
            )

        if voice_image_json.get("node_trace") != VOICE_IMAGE_TRACE:
            raise SystemExit(f"Unexpected voice+image trace: {voice_image_json.get('node_trace')}")
        if not voice_image_json.get("transcribed_text"):
            raise SystemExit("Voice+image request did not return transcribed_text.")
        print("Voice+image /api/search: PASS")
        print(f"Node trace: {' -> '.join(voice_image_json['node_trace'])}")

    print("All Phase 5 image checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
