import json
import math
import re
from pathlib import Path
from typing import Any


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

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
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

        out: list[dict[str, Any]] = []
        for score, idx in scored[:top_k]:
            product = dict(self.products[idx])
            product["score"] = round(score, 4)
            out.append(product)
        return out
