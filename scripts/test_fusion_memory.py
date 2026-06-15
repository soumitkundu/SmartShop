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


def _reload_backend_modules() -> None:
    modules = (
        "backend.config",
        "backend.memory.session_store",
        "backend.graph.modality",
        "backend.rag.retriever",
        "backend.rag.prompt",
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


def _assert_trace(payload: dict, label: str) -> None:
    trace = payload.get("node_trace") or []
    expected = payload.get("expected_trace") or []
    if trace != expected:
        raise SystemExit(f"{label}: trace mismatch\n  got:      {trace}\n  expected: {expected}")
    print(f"{label}: PASS ({payload.get('modality')})")


def _test_modality_matrix(
    client: TestClient,
    image_path: Path | None,
    audio_path: Path | None,
) -> None:
    from backend.graph.modality import expected_graph_trace

    cases: list[tuple[str, dict, dict | None]] = [
        ("text", {"text": "show me shoes", "session_id": "phase6-text"}, None),
    ]

    if image_path:
        with image_path.open("rb") as image_fp:
            image_bytes = image_fp.read()
        cases.extend(
            [
                (
                    "image",
                    {"session_id": "phase6-image"},
                    {"image_file": (image_path.name, image_bytes, "image/jpeg")},
                ),
                (
                    "text+image",
                    {"text": "similar style", "session_id": "phase6-text-image"},
                    {"image_file": (image_path.name, image_bytes, "image/jpeg")},
                ),
            ]
        )

    if audio_path:
        with audio_path.open("rb") as audio_fp:
            audio_bytes = audio_fp.read()
        cases.extend(
            [
                (
                    "voice",
                    {"session_id": "phase6-voice"},
                    {"audio_file": (audio_path.name, audio_bytes, "audio/wav")},
                ),
                (
                    "text+voice",
                    {"text": "under 2000", "session_id": "phase6-text-voice"},
                    {"audio_file": (audio_path.name, audio_bytes, "audio/wav")},
                ),
            ]
        )

    if image_path and audio_path:
        with image_path.open("rb") as image_fp, audio_path.open("rb") as audio_fp:
            image_bytes = image_fp.read()
            audio_bytes = audio_fp.read()
        cases.extend(
            [
                (
                    "voice+image",
                    {"session_id": "phase6-voice-image"},
                    {
                        "image_file": (image_path.name, image_bytes, "image/jpeg"),
                        "audio_file": (audio_path.name, audio_bytes, "audio/wav"),
                    },
                ),
                (
                    "text+voice+image",
                    {"text": "compare options", "session_id": "phase6-full"},
                    {
                        "image_file": (image_path.name, image_bytes, "image/jpeg"),
                        "audio_file": (audio_path.name, audio_bytes, "audio/wav"),
                    },
                ),
            ]
        )

    for label, data, files in cases:
        payload = _post_search(client, data, files)
        _assert_trace(payload, label)

    # Unit-level check that all seven combos have defined traces.
    from backend.graph.modality import MODALITY_COMBOS

    for combo in MODALITY_COMBOS:
        flags = {
            "text": {"has_text": True, "has_audio": False, "has_image": False},
            "voice": {"has_text": False, "has_audio": True, "has_image": False},
            "image": {"has_text": False, "has_audio": False, "has_image": True},
            "text+voice": {"has_text": True, "has_audio": True, "has_image": False},
            "text+image": {"has_text": True, "has_audio": False, "has_image": True},
            "voice+image": {"has_text": False, "has_audio": True, "has_image": True},
            "text+voice+image": {"has_text": True, "has_audio": True, "has_image": True},
        }[combo]
        trace = expected_graph_trace(flags)
        if "fuser" not in trace or "retriever" not in trace:
            raise SystemExit(f"Invalid trace for combo {combo}: {trace}")

    print(f"All {len(MODALITY_COMBOS)} modality traces defined: PASS")


def _test_rrf_merge() -> None:
    from backend.rag.retriever import merge_retrieval_results

    text_hits = [
        {"handle": "a", "title": "A", "score": 0.9},
        {"handle": "b", "title": "B", "score": 0.7},
    ]
    image_hits = [
        {"handle": "b", "title": "B", "score": 0.95},
        {"handle": "c", "title": "C", "score": 0.8},
    ]
    merged = merge_retrieval_results(text_hits, image_hits, top_k=3)
    handles = [item["handle"] for item in merged]
    if "b" not in handles:
        raise SystemExit("RRF merge did not include overlapping product 'b'.")
    if merged[0]["handle"] != "b":
        raise SystemExit(f"Expected overlapping product 'b' to rank first, got {handles}")
    print("RRF merge: PASS")


def _test_memory_window(client: TestClient, session_id: str, window_turns: int) -> None:
    client.delete(f"/api/session/{session_id}")

    base_query = "show me walking shoes under 2000"
    follow_ups = [
        "any Nike options",
        "which has better stock",
        "compare the top two",
        "show cheaper ones",
        "anything in black",
        "any other brands",
        "show me more walking shoes",
    ]

    first = _post_search(client, {"text": base_query, "session_id": session_id})
    if first.get("rejected"):
        raise SystemExit("Initial memory test query was unexpectedly rejected.")

    sent = 1
    for follow_up in follow_ups:
        payload = _post_search(
            client,
            {"text": follow_up, "session_id": session_id},
        )
        sent += 1
        if payload.get("rejected"):
            raise SystemExit(f"Memory follow-up was unexpectedly rejected: {follow_up!r}")
        if sent > window_turns + 1:
            break

    final = _post_search(
        client,
        {"text": "what about Adidas", "session_id": session_id},
    )
    if final.get("rejected"):
        raise SystemExit("Final memory test query was unexpectedly rejected.")

    memory_turns = final.get("memory_turns")
    if memory_turns != window_turns:
        raise SystemExit(
            f"Expected memory_turns={window_turns} after window trim, got {memory_turns}"
        )
    print(f"Bounded memory window ({window_turns} turns): PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 6 multimodal fusion + memory test")
    parser.add_argument("--image-file", help="JPEG/PNG/WebP for image modality checks")
    parser.add_argument("--audio-file", help="Audio file for voice modality checks")
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    products_path = os.getenv("PRODUCT_CATALOG_PATH", "./data/products.json")
    index_path = os.getenv("TEXT_INDEX_PATH", "./data/text_index.json")

    _check_file(products_path)
    _maybe_build_text_index(products_path, index_path)
    _reload_backend_modules()

    from backend.config import settings  # pylint: disable=import-outside-toplevel
    from backend.main import app  # pylint: disable=import-outside-toplevel

    print(f"Memory window turns: {settings.MEMORY_WINDOW_TURNS}")
    print(f"Fusion RRF k: {settings.FUSION_RRF_K}")

    _test_rrf_merge()

    client = TestClient(app)

    image_path = Path(args.image_file) if args.image_file else None
    audio_path = Path(args.audio_file) if args.audio_file else None
    if image_path and not image_path.exists():
        raise SystemExit(f"Image file not found: {image_path}")
    if audio_path and not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    _test_modality_matrix(client, image_path, audio_path)
    _test_memory_window(client, "phase6-memory-session", settings.MEMORY_WINDOW_TURNS)

    print("All Phase 6 fusion + memory checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
