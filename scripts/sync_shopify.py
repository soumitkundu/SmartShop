import argparse
import asyncio
import csv
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv


DEFAULT_API_VERSION = os.getenv("SHOPIFY_API_VERSION", "2025-01")


def _env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


def _as_bool(s: str | None, default: bool = False) -> bool:
    if s is None:
        return default
    return s.strip().lower() in {"1", "true", "yes", "y", "on"}


def _split_tags(s: str) -> list[str]:
    parts = [p.strip() for p in (s or "").split(",") if p.strip()]
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        k = p.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out


@dataclass(frozen=True)
class ShopifyCsvRow:
    title: str
    handle: str
    body_html: str
    vendor: str
    product_type: str
    tags: list[str]
    status: str
    sku: str
    price: str
    compare_at_price: str | None
    inventory_quantity: int
    image_url: str | None
    image_alt: str | None


def _read_shopify_csv(path: Path, limit: int | None) -> list[ShopifyCsvRow]:
    out: list[ShopifyCsvRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = (row.get("Title") or "").strip()
            handle = (row.get("URL handle") or "").strip()
            if not handle:
                continue
            inv_qty = int((row.get("Inventory quantity") or "0").strip() or "0")
            image_url = (row.get("Product image URL") or "").strip() or None
            image_alt = (row.get("Image alt text") or "").strip() or None
            compare_at = (row.get("Compare-at price") or "").strip() or None
            out.append(
                ShopifyCsvRow(
                    title=title,
                    handle=handle,
                    body_html=(row.get("Description") or "").strip(),
                    vendor=(row.get("Vendor") or "").strip(),
                    product_type=(row.get("Type") or "").strip(),
                    tags=_split_tags(row.get("Tags") or ""),
                    status=(row.get("Status") or "Draft").strip().capitalize(),
                    sku=(row.get("SKU") or "").strip(),
                    price=(row.get("Price") or "0").strip(),
                    compare_at_price=compare_at,
                    inventory_quantity=inv_qty,
                    image_url=image_url,
                    image_alt=image_alt,
                )
            )
            if limit is not None and len(out) >= limit:
                break
    return out


def _base_url(store_domain: str, api_version: str) -> str:
    # store_domain should be like 'soumit-dev-app.myshopify.com'
    return f"https://{store_domain}/admin/api/{api_version}"


class ShopifyClient:
    def __init__(self, store_domain: str, access_token: str, api_version: str = DEFAULT_API_VERSION):
        self.store_domain = store_domain
        self.api_version = api_version
        self._client = httpx.AsyncClient(
            base_url=_base_url(store_domain, api_version),
            headers={
                "X-Shopify-Access-Token": access_token,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=httpx.Timeout(30.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request_with_retries(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: Any | None = None,
        max_attempts: int = 8,
    ) -> httpx.Response:
        attempt = 0
        last_exc: Exception | None = None

        while attempt < max_attempts:
            attempt += 1
            try:
                resp = await self._client.request(method, url, params=params, json=json_body)

                if resp.status_code in (429, 500, 502, 503, 504):
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = float(retry_after)
                        except ValueError:
                            delay = None
                    else:
                        delay = None

                    if delay is None:
                        # Exponential backoff + jitter
                        delay = min(30.0, (2 ** (attempt - 1)) * 0.5) + random.random()
                    await asyncio.sleep(delay)
                    continue

                return resp
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_exc = e
                delay = min(30.0, (2 ** (attempt - 1)) * 0.5) + random.random()
                await asyncio.sleep(delay)

        raise RuntimeError(f"Shopify request failed after {max_attempts} attempts: {last_exc}")

    async def list_products_by_handle(self) -> dict[str, int]:
        """
        Paginate through products and build a {handle -> product_id} index.

        Uses since_id pagination for stability (REST).
        """
        handle_to_id: dict[str, int] = {}
        since_id = 0

        while True:
            resp = await self._request_with_retries(
                "GET",
                "/products.json",
                params={"limit": 250, "since_id": since_id},
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Failed to list products: {resp.status_code} {resp.text}")

            data = resp.json()
            items = data.get("products") or []
            if not items:
                break

            for p in items:
                h = p.get("handle")
                pid = p.get("id")
                if h and pid:
                    handle_to_id[str(h)] = int(pid)
                since_id = max(since_id, int(pid))

        return handle_to_id

    async def create_product(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = await self._request_with_retries("POST", "/products.json", json_body={"product": payload})
        if resp.status_code not in (200, 201):
            raise RuntimeError(f"Create product failed: {resp.status_code} {resp.text}")
        return resp.json()["product"]

    async def delete_product(self, product_id: int) -> None:
        resp = await self._request_with_retries("DELETE", f"/products/{product_id}.json")
        if resp.status_code not in (200, 202):
            # If it's already gone, ignore
            if resp.status_code == 404:
                return
            raise RuntimeError(f"Delete product failed: {resp.status_code} {resp.text}")


def _product_payload(row: ShopifyCsvRow) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    if row.image_url:
        img: dict[str, Any] = {"src": row.image_url}
        if row.image_alt:
            img["alt"] = row.image_alt
        images.append(img)

    variant: dict[str, Any] = {
        "sku": row.sku or row.handle,
        "price": row.price,
        "inventory_management": "shopify",
        "inventory_policy": "deny",
        "inventory_quantity": row.inventory_quantity,
    }
    if row.compare_at_price:
        variant["compare_at_price"] = row.compare_at_price

    payload: dict[str, Any] = {
        "title": row.title,
        "handle": row.handle,
        "body_html": row.body_html,
        "vendor": row.vendor,
        "product_type": row.product_type,
        "tags": ", ".join(row.tags),
        "status": row.status.lower(),
        "variants": [variant],
    }
    if images:
        payload["images"] = images
    return payload


async def run(args: argparse.Namespace) -> int:
    load_dotenv()

    store_domain = args.store_domain or _env("SHOPIFY_STORE_DOMAIN")
    token = args.access_token or _env("SHOPIFY_ADMIN_API_ACCESS_TOKEN")
    api_version = args.api_version or DEFAULT_API_VERSION

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise RuntimeError(f"CSV not found: {csv_path}")

    rows = _read_shopify_csv(csv_path, args.limit)
    if not rows:
        print("No rows found in CSV (nothing to sync).")
        return 0

    client = ShopifyClient(store_domain=store_domain, access_token=token, api_version=api_version)
    try:
        existing = await client.list_products_by_handle()
        created = 0
        skipped = 0
        deleted = 0
        failures: list[dict[str, Any]] = []

        for r in rows:
            existing_id = existing.get(r.handle)
            if existing_id is not None and not args.overwrite:
                skipped += 1
                continue

            if existing_id is not None and args.overwrite:
                try:
                    await client.delete_product(existing_id)
                    deleted += 1
                except Exception as e:
                    failures.append({"handle": r.handle, "action": "delete", "error": str(e)})
                    continue

            try:
                await client.create_product(_product_payload(r))
                created += 1
            except Exception as e:
                failures.append({"handle": r.handle, "action": "create", "error": str(e)})

        summary = {
            "store_domain": store_domain,
            "api_version": api_version,
            "csv": str(csv_path).replace("\\", "/"),
            "total_rows": len(rows),
            "created": created,
            "skipped_existing": skipped,
            "deleted_before_create": deleted,
            "failures": failures,
        }

        out_path = Path(args.report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print(json.dumps(summary, indent=2))

        return 1 if failures else 0
    finally:
        await client.aclose()


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync products from Shopify import CSV into a Shopify store.")
    parser.add_argument("--csv", default="data/shopify_import.csv", help="Shopify import-ready CSV path")
    parser.add_argument("--report", default="data/shopify_sync_report.json", help="Write sync report JSON here")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows to sync")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, delete existing products with matching handle then recreate",
    )
    parser.add_argument("--store-domain", default=None, help="Override SHOPIFY_STORE_DOMAIN")
    parser.add_argument("--access-token", default=None, help="Override SHOPIFY_ADMIN_API_ACCESS_TOKEN")
    parser.add_argument("--api-version", default=None, help="Override SHOPIFY_API_VERSION (default 2025-01)")

    args = parser.parse_args()

    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

