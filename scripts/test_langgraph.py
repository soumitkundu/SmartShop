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

EXPECTED_TRACE = ["router", "fuser", "retriever", "generator"]


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
    for name in ("backend.config", "backend.graph.agent", "backend.main"):
        if name in sys.modules:
            importlib.reload(sys.modules[name])


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 LangGraph skeleton test")
    parser.add_argument("--query", default="show me walking shoes under 2000")
    parser.add_argument(
        "--reject-query",
        default="Who won the FIFA world cup in 2010?",
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

    print(f"LangSmith tracing: {settings.LANGCHAIN_TRACING_V2}")
    print(f"LangSmith project: {settings.LANGCHAIN_PROJECT}")

    client = TestClient(app)

    ok_resp = client.post(
        "/api/search",
        data={"text": args.query, "session_id": "phase3-test-session"},
    )
    if ok_resp.status_code != 200:
        raise SystemExit(f"/api/search failed: {ok_resp.status_code} {ok_resp.text}")

    ok_json = ok_resp.json()
    if ok_json.get("rejected"):
        raise SystemExit("Catalog query was unexpectedly rejected.")
    if not ok_json.get("products"):
        raise SystemExit("Catalog query returned empty products.")

    trace = ok_json.get("node_trace") or []
    if trace != EXPECTED_TRACE:
        raise SystemExit(f"Unexpected node_trace: {trace} (expected {EXPECTED_TRACE})")

    print("/api/search catalog query: PASS")
    print(f"Node trace: {' -> '.join(trace)}")
    print(f"Provider used: {ok_json.get('provider')}")
    print(f"Answer preview: {(ok_json.get('answer') or '')[:140]}")

    reject_resp = client.post(
        "/api/search",
        data={"text": args.reject_query, "session_id": "phase3-test-session"},
    )
    if reject_resp.status_code != 200:
        raise SystemExit(
            f"/api/search failed for reject query: {reject_resp.status_code} {reject_resp.text}"
        )

    reject_json = reject_resp.json()
    if not reject_json.get("rejected"):
        raise SystemExit("Out-of-catalog query was not rejected.")

    reject_trace = reject_json.get("node_trace") or []
    if reject_trace != EXPECTED_TRACE:
        raise SystemExit(f"Unexpected reject node_trace: {reject_trace}")

    print("/api/search reject query: PASS")
    print(f"Reject node trace: {' -> '.join(reject_trace)}")
    print("All LangGraph checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
