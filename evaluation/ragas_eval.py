"""
SmartShop evaluation runner (Phase 8).

Runs catalog queries from evaluation/test_queries.json against /api/search and
reports practical proxy metrics. Optional RAGAS metrics when ragas is installed.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_QUERIES_PATH = PROJECT_ROOT / "evaluation" / "test_queries.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "evaluation" / "eval_report.json"


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    should_reject: bool
    rejected: bool
    product_count: int
    in_stock_ok: bool
    keyword_hit: bool | None
    latency_ms: float
    error: str | None = None
    notes: list[str] = field(default_factory=list)


def _load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("queries")
    if not isinstance(cases, list) or not cases:
        raise SystemExit(f"No queries found in {path}")
    return cases


def _product_blob(products: list[dict[str, Any]]) -> str:
  parts: list[str] = []
  for product in products:
    parts.extend(
      [
        str(product.get("title") or ""),
        str(product.get("vendor") or ""),
        str(product.get("type") or ""),
        " ".join(str(t) for t in (product.get("tags") or [])),
      ]
    )
  return " ".join(parts).lower()


def _keyword_hit(products: list[dict[str, Any]], keywords: list[str]) -> bool:
  if not keywords:
    return True
  blob = _product_blob(products)
  return any(keyword.lower() in blob for keyword in keywords)


def _in_stock_ok(products: list[dict[str, Any]]) -> bool:
  if not products:
    return True
  for product in products:
    qty = product.get("inventory_quantity")
    if qty is None:
      continue
    try:
      if int(qty) <= 0:
        return False
    except (TypeError, ValueError):
      continue
  return True


def _run_case(client: TestClient, case: dict[str, Any], session_prefix: str) -> CaseResult:
  case_id = str(case.get("id") or "unknown")
  should_reject = bool(case.get("should_reject"))
  require_in_stock = bool(case.get("require_in_stock"))
  keywords = [str(k) for k in (case.get("expect_keywords_in_products") or [])]

  started = time.perf_counter()
  try:
    response = client.post(
      "/api/search",
      data={
        "text": case.get("text") or "",
        "session_id": f"{session_prefix}-{case_id}",
      },
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    if response.status_code != 200:
      return CaseResult(
        case_id=case_id,
        passed=False,
        should_reject=should_reject,
        rejected=False,
        product_count=0,
        in_stock_ok=False,
        keyword_hit=None,
        latency_ms=latency_ms,
        error=f"HTTP {response.status_code}: {response.text[:240]}",
      )

    payload = response.json()
    rejected = bool(payload.get("rejected"))
    products = payload.get("products") or []
    keyword_ok = _keyword_hit(products, keywords) if not should_reject else True
    stock_ok = _in_stock_ok(products) if require_in_stock or not should_reject else True

    notes: list[str] = []
    passed = True

    if rejected != should_reject:
      passed = False
      notes.append(f"rejection mismatch (expected={should_reject}, got={rejected})")

    if not should_reject and not products and not rejected:
      passed = False
      notes.append("expected products for in-catalog query")

    if not should_reject and keywords and not keyword_ok:
      passed = False
      notes.append("retrieval keywords not found in product fields")

    if require_in_stock and not stock_ok:
      passed = False
      notes.append("returned out-of-stock product(s)")

    answer = (payload.get("answer") or "").strip()
    if not should_reject and not answer:
      passed = False
      notes.append("empty answer for in-catalog query")

    return CaseResult(
      case_id=case_id,
      passed=passed,
      should_reject=should_reject,
      rejected=rejected,
      product_count=len(products),
      in_stock_ok=stock_ok,
      keyword_hit=keyword_ok if keywords else None,
      latency_ms=round(latency_ms, 1),
      notes=notes,
    )
  except Exception as exc:
    latency_ms = (time.perf_counter() - started) * 1000.0
    return CaseResult(
      case_id=case_id,
      passed=False,
      should_reject=should_reject,
      rejected=False,
      product_count=0,
      in_stock_ok=False,
      keyword_hit=None,
      latency_ms=round(latency_ms, 1),
      error=str(exc),
    )


def _maybe_run_ragas(cases: list[dict[str, Any]], client: TestClient) -> dict[str, Any] | None:
  try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness
  except ImportError:
    return None

  rows: list[dict[str, str]] = []
  for case in cases:
    if case.get("should_reject"):
      continue
    response = client.post(
      "/api/search",
      data={
        "text": case.get("text") or "",
        "session_id": f"ragas-{case.get('id')}",
      },
    )
    if response.status_code != 200:
      continue
    payload = response.json()
    if payload.get("rejected"):
      continue
    products = payload.get("products") or []
    context = "\n".join(
      f"{p.get('title')} | price={p.get('price')} | stock={p.get('inventory_quantity')}"
      for p in products
    )
    rows.append(
      {
        "question": str(case.get("text") or ""),
        "answer": str(payload.get("answer") or ""),
        "contexts": [context],
      }
    )

  if not rows:
    return {"skipped": True, "reason": "no in-catalog rows for RAGAS"}

  dataset = Dataset.from_list(rows)
  result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
  scores = {key: float(value) for key, value in result.items()}
  return {"scores": scores, "sample_size": len(rows)}


def _summarize(results: list[CaseResult]) -> dict[str, Any]:
  total = len(results)
  passed = sum(1 for r in results if r.passed)
  reject_cases = [r for r in results if r.should_reject]
  catalog_cases = [r for r in results if not r.should_reject]
  reject_acc = (
    sum(1 for r in reject_cases if r.rejected == r.should_reject) / len(reject_cases)
    if reject_cases
    else 1.0
  )
  retrieval_hit = (
    sum(1 for r in catalog_cases if r.keyword_hit is not False) / len(catalog_cases)
    if catalog_cases
    else 1.0
  )
  in_stock_rate = (
    sum(1 for r in catalog_cases if r.in_stock_ok) / len(catalog_cases)
    if catalog_cases
    else 1.0
  )
  avg_latency = round(sum(r.latency_ms for r in results) / total, 1) if total else 0.0

  return {
    "total_cases": total,
    "passed_cases": passed,
    "pass_rate": round(passed / total, 4) if total else 0.0,
    "rejection_accuracy": round(reject_acc, 4),
    "retrieval_keyword_hit_rate": round(retrieval_hit, 4),
    "in_stock_rate": round(in_stock_rate, 4),
    "avg_latency_ms": avg_latency,
  }


def main() -> int:
  parser = argparse.ArgumentParser(description="Run SmartShop evaluation suite")
  parser.add_argument("--queries", default=str(DEFAULT_QUERIES_PATH), help="Path to test_queries.json")
  parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH), help="Output JSON report path")
  parser.add_argument("--session-prefix", default="eval", help="Session id prefix")
  parser.add_argument("--ragas", action="store_true", help="Run optional RAGAS metrics if installed")
  args = parser.parse_args()

  load_dotenv()

  queries_path = Path(args.queries)
  if not queries_path.exists():
    raise SystemExit(f"Queries file not found: {queries_path}")

  products_path = Path(os.getenv("PRODUCT_CATALOG_PATH", "./data/products.json"))
  index_path = Path(os.getenv("TEXT_INDEX_PATH", "./data/text_index.json"))
  if not index_path.exists():
    raise SystemExit(
      f"Missing text index at {index_path}. Run: py scripts/embed_products.py --input {products_path}"
    )

  for name in ("backend.config", "backend.graph.agent", "backend.main"):
    if name in sys.modules:
      importlib.reload(sys.modules[name])

  from backend.main import app  # pylint: disable=import-outside-toplevel

  cases = _load_cases(queries_path)
  client = TestClient(app)

  health = client.get("/api/health")
  if health.status_code != 200:
    raise SystemExit(f"/api/health failed: {health.status_code}")

  results = [_run_case(client, case, args.session_prefix) for case in cases]
  summary = _summarize(results)

  ragas_block: dict[str, Any] | None = None
  if args.ragas:
    ragas_block = _maybe_run_ragas(cases, client)

  report = {
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "health": health.json(),
    "summary": summary,
    "ragas": ragas_block,
    "cases": [
      {
        "id": r.case_id,
        "passed": r.passed,
        "should_reject": r.should_reject,
        "rejected": r.rejected,
        "product_count": r.product_count,
        "in_stock_ok": r.in_stock_ok,
        "keyword_hit": r.keyword_hit,
        "latency_ms": r.latency_ms,
        "error": r.error,
        "notes": r.notes,
      }
      for r in results
    ],
  }

  report_path = Path(args.report)
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

  print("SmartShop evaluation summary")
  print(json.dumps(summary, indent=2))
  if ragas_block:
    print("RAGAS:")
    print(json.dumps(ragas_block, indent=2))
  print(f"Report written to: {report_path}")

  failed = [r for r in results if not r.passed]
  if failed:
    print("\nFailed cases:")
    for item in failed:
      print(f"- {item.case_id}: {item.error or '; '.join(item.notes) or 'failed'}")
    return 1

  print("\nAll evaluation cases passed.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
