import json
import math
import re
from pathlib import Path
from typing import Any

import chromadb


TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class TextRetriever:
    def __init__(self, index_path: str):
        self.index_path = Path(index_path)
        self.products: list[dict[str, Any]] = []
        self.idf: dict[str, float] = {}
        self.doc_vectors: list[dict[str, float]] = []
        self.doc_norms: list[float] = []
        self._load_index()

    def _load_index(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"Text index not found at {self.index_path}. Run scripts/embed_products.py first."
            )

        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.products = payload.get("products", [])
        self.idf = payload.get("idf", {})
        self.doc_vectors = payload.get("doc_vectors", [])
        self.doc_norms = payload.get("doc_norms", [])

    def _query_vector(self, text: str) -> tuple[dict[str, float], float]:
        tokens = _tokenize(text)
        if not tokens:
            return {}, 0.0

        tf: dict[str, int] = {}
        for tok in tokens:
            if tok in self.idf:
                tf[tok] = tf.get(tok, 0) + 1

        if not tf:
            return {}, 0.0

        size = len(tokens)
        vec: dict[str, float] = {}
        norm_sq = 0.0
        for tok, count in tf.items():
            weight = (count / size) * self.idf[tok]
            vec[tok] = weight
            norm_sq += weight * weight

        return vec, math.sqrt(norm_sq)

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        exclude_out_of_stock: bool = False,
    ) -> list[dict[str, Any]]:
        q_vec, q_norm = self._query_vector(query)
        if not q_vec or q_norm == 0.0:
            return []

        scored: list[tuple[float, int]] = []
        for idx, d_vec in enumerate(self.doc_vectors):
            d_norm = self.doc_norms[idx]
            if d_norm == 0.0:
                continue

            dot = 0.0
            for tok, q_w in q_vec.items():
                d_w = d_vec.get(tok)
                if d_w is not None:
                    dot += q_w * d_w

            if dot == 0.0:
                continue

            score = dot / (q_norm * d_norm)
            if score > 0.0:
                scored.append((score, idx))

        scored.sort(key=lambda x: x[0], reverse=True)

        fetch_k = top_k * _fetch_multiplier(top_k, exclude_out_of_stock=exclude_out_of_stock)
        out: list[dict[str, Any]] = []
        for score, idx in scored[:fetch_k]:
            product = dict(self.products[idx])
            product["score"] = round(score, 4)
            out.append(product)

        if exclude_out_of_stock:
            return filter_in_stock(out, top_k)
        return out[:top_k]


def _metadata_to_product(metadata: dict[str, Any], score: float) -> dict[str, Any]:
    tags_raw = metadata.get("tags") or ""
    tags = [t.strip() for t in str(tags_raw).split(",") if t.strip()]
    return {
        "handle": metadata.get("handle"),
        "title": metadata.get("title"),
        "vendor": metadata.get("vendor"),
        "type": metadata.get("type"),
        "tags": tags,
        "price": metadata.get("price"),
        "inventory_quantity": metadata.get("inventory_quantity"),
        "image_url": metadata.get("image_url"),
        "score": round(score, 4),
    }


class ImageRetriever:
    def __init__(self, chroma_path: str, collection_name: str):
        self.chroma_path = Path(chroma_path)
        self.collection_name = collection_name
        self._collection = None
        self._load_collection()

    def _load_collection(self) -> None:
        if not self.chroma_path.exists():
            raise FileNotFoundError(
                f"ChromaDB path not found at {self.chroma_path}. "
                "Run scripts/embed_product_images.py first."
            )

        client = chromadb.PersistentClient(path=str(self.chroma_path))
        try:
            self._collection = client.get_collection(name=self.collection_name)
        except Exception as exc:
            raise FileNotFoundError(
                f"Image collection '{self.collection_name}' not found in {self.chroma_path}. "
                "Run scripts/embed_product_images.py first."
            ) from exc

    def search(
        self,
        image_vector: list[float],
        top_k: int = 5,
        *,
        exclude_out_of_stock: bool = False,
    ) -> list[dict[str, Any]]:
        if not image_vector:
            return []

        fetch_k = top_k * _fetch_multiplier(top_k, exclude_out_of_stock=exclude_out_of_stock)
        results = self._collection.query(
            query_embeddings=[image_vector],
            n_results=fetch_k,
            include=["metadatas", "distances", "documents"],
        )

        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        if not metadatas:
            return []

        out: list[dict[str, Any]] = []
        for metadata, distance in zip(metadatas, distances):
            if not metadata:
                continue
            score = max(0.0, 1.0 - float(distance))
            if score <= 0.0:
                continue
            out.append(_metadata_to_product(metadata, score))

        if exclude_out_of_stock:
            return filter_in_stock(out, top_k)
        return out[:top_k]


def is_in_stock(product: dict[str, Any]) -> bool:
    """Treat missing inventory as available; exclude explicit zero stock."""
    qty = product.get("inventory_quantity")
    if qty is None:
        return True
    try:
        return int(qty) > 0
    except (TypeError, ValueError):
        return True


def filter_in_stock(hits: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    return [hit for hit in hits if is_in_stock(hit)][:top_k]


def _fetch_multiplier(top_k: int, *, exclude_out_of_stock: bool) -> int:
    if not exclude_out_of_stock:
        return 1
    return max(4, min(20, top_k * 4))


def _product_key(hit: dict[str, Any]) -> str:
    return str(hit.get("handle") or hit.get("title") or "")


def merge_retrieval_results(
    text_hits: list[dict[str, Any]],
    image_hits: list[dict[str, Any]],
    top_k: int,
    *,
    rrf_k: int = 60,
    text_weight: float = 1.0,
    image_weight: float = 1.0,
) -> list[dict[str, Any]]:
    """Merge text and image hits with weighted reciprocal rank fusion (RRF).

    RRF avoids comparing incompatible raw scores (TF-IDF cosine vs CLIP distance).
    Products appearing in both lists get a boost from combined ranks.
    """
    if not text_hits:
        return image_hits[:top_k]
    if not image_hits:
        return text_hits[:top_k]

    fusion_scores: dict[str, float] = {}
    docs: dict[str, dict[str, Any]] = {}
    text_scores: dict[str, float] = {}
    image_scores: dict[str, float] = {}

    for rank, hit in enumerate(text_hits):
        key = _product_key(hit)
        if not key:
            continue
        text_scores[key] = float(hit.get("score", 0.0))
        fusion_scores[key] = fusion_scores.get(key, 0.0) + text_weight / (rrf_k + rank + 1)
        docs[key] = dict(hit)

    for rank, hit in enumerate(image_hits):
        key = _product_key(hit)
        if not key:
            continue
        image_scores[key] = float(hit.get("score", 0.0))
        fusion_scores[key] = fusion_scores.get(key, 0.0) + image_weight / (rrf_k + rank + 1)
        if key not in docs:
            docs[key] = dict(hit)

    ranked_keys = sorted(fusion_scores.keys(), key=lambda k: fusion_scores[k], reverse=True)

    out: list[dict[str, Any]] = []
    for key in ranked_keys[:top_k]:
        product = dict(docs[key])
        product["fusion_score"] = round(fusion_scores[key], 6)
        product["text_score"] = round(text_scores.get(key, 0.0), 4)
        product["image_score"] = round(image_scores.get(key, 0.0), 4)
        product["score"] = product["fusion_score"]
        out.append(product)
    return out
