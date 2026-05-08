import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi.testclient import TestClient

# Make project-root imports work when running from scripts/ on Windows.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick Phase 2 Text RAG test")
    parser.add_argument("--query", default="Show me a budget phone under 20000", help="Catalog query to test")
    parser.add_argument(
        "--reject-query",
        default="Who won the FIFA world cup in 2010?",
        help="Out-of-catalog query expected to be rejected",
    )
    args = parser.parse_args()

    load_dotenv()

    products_path = os.getenv("PRODUCT_CATALOG_PATH", "./data/products.json")
    index_path = os.getenv("TEXT_INDEX_PATH", "./data/text_index.json")

    _check_file(products_path)
    _maybe_build_index(products_path, index_path)

    # Ensure settings read current environment before importing app.
    if "backend.config" in sys.modules:
        importlib.reload(sys.modules["backend.config"])
    if "backend.main" in sys.modules:
        importlib.reload(sys.modules["backend.main"])

    from backend.main import app  # pylint: disable=import-outside-toplevel
    from backend.rag.retriever import TextRetriever  # pylint: disable=import-outside-toplevel

    retriever = TextRetriever(index_path)
    hits = retriever.search(args.query, top_k=5)
    if not hits:
        raise SystemExit("Retriever returned no hits for test query.")

    print("Retriever check: PASS")
    print(f"Top hit: {hits[0].get('title')} | score={hits[0].get('score')}")

    client = TestClient(app)

    ok_resp = client.post(
        "/api/search",
        data={
            "text": args.query,
            "session_id": "phase2-test-session",
        },
    )
    if ok_resp.status_code != 200:
        raise SystemExit(f"/api/search failed for catalog query: {ok_resp.status_code} {ok_resp.text}")
    ok_json = ok_resp.json()
    if ok_json.get("rejected"):
        raise SystemExit("Catalog query was unexpectedly rejected.")
    if not ok_json.get("products"):
        raise SystemExit("Catalog query returned empty products.")

    print("/api/search catalog query: PASS")
    print(f"Provider used: {ok_json.get('provider')}")
    print(f"Answer preview: {(ok_json.get('answer') or '')[:140]}")

    reject_resp = client.post(
        "/api/search",
        data={
            "text": args.reject_query,
            "session_id": "phase2-test-session",
        },
    )
    if reject_resp.status_code != 200:
        raise SystemExit(
            f"/api/search failed for reject query: {reject_resp.status_code} {reject_resp.text}"
        )
    reject_json = reject_resp.json()
    if not reject_json.get("rejected"):
        raise SystemExit("Out-of-catalog query was not rejected.")

    print("/api/search reject query: PASS")
    print("All Text RAG checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
