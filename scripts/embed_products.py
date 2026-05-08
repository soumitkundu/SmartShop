import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _product_text(p: dict[str, Any]) -> str:
    parts = [
        str(p.get("title") or ""),
        str(p.get("vendor") or ""),
        str(p.get("type") or ""),
        " ".join(p.get("tags") or []),
    ]
    return " ".join(parts).strip()


def _load_products(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    products = payload.get("products")
    if not isinstance(products, list):
        raise RuntimeError(f"Invalid products.json format at {path}")
    return products


def main() -> int:
    parser = argparse.ArgumentParser(description="Build local text index from data/products.json")
    parser.add_argument("--input", default="data/products.json", help="Input products json path")
    parser.add_argument("--out", default="data/text_index.json", help="Output text index path")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_path = Path(args.out)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    products = _load_products(input_path)
    tokenized_docs: list[list[str]] = []
    for p in products:
        tokenized_docs.append(_tokenize(_product_text(p)))

    doc_count = len(tokenized_docs)
    if doc_count == 0:
        raise SystemExit("No products available to index.")

    # Document frequency
    df: Counter[str] = Counter()
    for toks in tokenized_docs:
        for tok in set(toks):
            df[tok] += 1

    # Smooth IDF
    idf: dict[str, float] = {
        tok: math.log((1 + doc_count) / (1 + freq)) + 1.0 for tok, freq in df.items()
    }

    doc_vectors: list[dict[str, float]] = []
    doc_norms: list[float] = []
    for toks in tokenized_docs:
        if not toks:
            doc_vectors.append({})
            doc_norms.append(0.0)
            continue

        tf = Counter(toks)
        size = len(toks)
        vec: dict[str, float] = {}
        norm_sq = 0.0
        for tok, count in tf.items():
            weight = (count / size) * idf.get(tok, 0.0)
            if weight == 0.0:
                continue
            vec[tok] = weight
            norm_sq += weight * weight

        doc_vectors.append(vec)
        doc_norms.append(math.sqrt(norm_sq))

    out_payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "source": str(input_path).replace("\\", "/"),
        "count": len(products),
        "idf": idf,
        "doc_vectors": doc_vectors,
        "doc_norms": doc_norms,
        "products": products,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, ensure_ascii=False), encoding="utf-8")
    print(f"Indexed {len(products)} products -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
