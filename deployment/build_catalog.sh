#!/usr/bin/env bash
# Build catalog artifacts baked into the Docker image (Phase 8).
set -euo pipefail

LIMIT="${CATALOG_LIMIT:-50}"

echo "[build] formatting Kaggle catalog (limit=${LIMIT})..."
python -m scripts.format_kaggle --limit "${LIMIT}"

echo "[build] building text index..."
python -m scripts.embed_products

echo "[build] building ChromaDB image index..."
python -m scripts.embed_product_images --limit "${LIMIT}"

python - <<'PY'
from pathlib import Path

products = Path("data/products.json")
text_index = Path("data/text_index.json")
chroma = Path("chroma_db")

missing = [p for p in (products, text_index) if not p.is_file()]
if not chroma.is_dir():
    missing.append(chroma)
if missing:
    raise SystemExit(f"Missing build artifacts: {missing}")

print("[build] catalog artifacts verified:")
print(f"  - {products}")
print(f"  - {text_index}")
print(f"  - {chroma}/")
PY
