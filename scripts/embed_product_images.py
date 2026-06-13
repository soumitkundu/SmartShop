import argparse
import json
import sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import chromadb
import clip
import requests
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings  # noqa: E402


def _load_products(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    products = payload.get("products")
    if not isinstance(products, list):
        raise RuntimeError(f"Invalid products.json format at {path}")
    return products


def _product_metadata(product: dict[str, Any]) -> dict[str, Any]:
    tags = product.get("tags") or []
    if isinstance(tags, list):
        tags_str = ", ".join(str(t) for t in tags)
    else:
        tags_str = str(tags)

    return {
        "handle": str(product.get("handle") or ""),
        "title": str(product.get("title") or ""),
        "vendor": str(product.get("vendor") or ""),
        "type": str(product.get("type") or ""),
        "tags": tags_str,
        "price": float(product.get("price") or 0.0),
        "inventory_quantity": int(product.get("inventory_quantity") or 0),
        "image_url": str(product.get("image_url") or ""),
    }


def _encode_image_url(model, preprocess, image_url: str) -> list[float] | None:
    response = requests.get(image_url, timeout=10)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert("RGB")
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        vec = model.encode_image(tensor)
        vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec.cpu().numpy().tolist()[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Embed product images into ChromaDB shopify_images collection"
    )
    parser.add_argument("--input", default="data/products.json", help="Input products json path")
    parser.add_argument("--chroma-path", default=settings.CHROMA_PATH, help="ChromaDB persist path")
    parser.add_argument(
        "--collection",
        default=settings.IMAGE_COLLECTION_NAME,
        help="ChromaDB collection name for image vectors",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max products to embed (for quick tests)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    products = _load_products(input_path)
    if args.limit:
        products = products[: args.limit]

    device = "cpu"
    print(f"Loading CLIP model={settings.CLIP_MODEL} on {device}...")
    model, preprocess = clip.load(settings.CLIP_MODEL, device=device)
    model.eval()

    client = chromadb.PersistentClient(path=args.chroma_path)
    collection = client.get_or_create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine"},
    )

    ids: list[str] = []
    embeddings: list[list[float]] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    indexed = 0
    skipped = 0
    for product in products:
        handle = product.get("handle")
        image_url = product.get("image_url")
        if not handle or not image_url:
            skipped += 1
            continue

        try:
            vec = _encode_image_url(model, preprocess, image_url)
            if not vec:
                skipped += 1
                continue
        except Exception as exc:
            title = product.get("title") or handle
            print(f"Skipping image for {title}: {exc}")
            skipped += 1
            continue

        ids.append(f"{handle}_img")
        embeddings.append(vec)
        documents.append(str(product.get("title") or handle))
        metadatas.append(_product_metadata(product))
        indexed += 1

    if not ids:
        raise SystemExit("No product images were embedded. Check image_url fields in products.json.")

    # Replace collection contents on each run for a clean rebuild.
    try:
        client.delete_collection(args.collection)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    report = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "source": str(input_path).replace("\\", "/"),
        "chroma_path": str(Path(args.chroma_path)).replace("\\", "/"),
        "collection": args.collection,
        "indexed": indexed,
        "skipped": skipped,
        "clip_model": settings.CLIP_MODEL,
    }
    report_path = Path(args.chroma_path) / "image_index_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Indexed {indexed} product images -> {args.chroma_path}/{args.collection}")
    if skipped:
        print(f"Skipped {skipped} products (missing or failed image URLs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
