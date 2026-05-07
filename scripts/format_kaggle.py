import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


SHOPIFY_TEMPLATE_PATH = Path("data/Shopify-product_template.csv")


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    text = re.sub(r"^-+|-+$", "", text)
    return text or "product"


def _parse_float(val: str | None) -> float | None:
    if val is None:
        return None
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_int(val: str | None) -> int | None:
    f = _parse_float(val)
    if f is None:
        return None
    return int(f)


def _first_image(images_field: str | None) -> str | None:
    if not images_field:
        return None
    parts = [p.strip() for p in images_field.split("|") if p.strip()]
    return parts[0] if parts else None


def _normalize_tags(tags: Iterable[str]) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for t in tags:
        t2 = re.sub(r"\s+", " ", (t or "").strip())
        if not t2:
            continue
        key = t2.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t2)
    return ", ".join(out)


def _pick_type_from_breadcrumbs(breadcrumbs: str | None) -> str | None:
    if not breadcrumbs:
        return None
    parts = [p.strip() for p in breadcrumbs.split("|") if p.strip()]
    if not parts:
        return None
    # Use the leaf category as "Type" (Shopify 'Product type' concept)
    return parts[-1][:255]


def _safe_vendor(brand: str | None) -> str:
    b = (brand or "").strip()
    return b if b else "Unknown"


@dataclass(frozen=True)
class FormattedProduct:
    title: str
    handle: str
    body_html: str
    vendor: str
    product_type: str | None
    tags: str
    status: str
    sku: str
    barcode: str | None
    price: str
    compare_at_price: str | None
    inventory_quantity: int
    image_url: str | None
    image_alt: str | None
    source_url: str | None
    raw: dict[str, Any]


def _format_row(row: dict[str, str]) -> FormattedProduct:
    title = (row.get("Product Title") or "").strip()
    if not title:
        title = f"Product {row.get('Skuid') or row.get('Uniq Id') or ''}".strip()

    handle = _slugify(title)
    sku = (row.get("Skuid") or "").strip() or handle
    vendor = _safe_vendor(row.get("Brand"))
    product_type = _pick_type_from_breadcrumbs(row.get("Breadcrumbs"))

    list_price = _parse_float(row.get("List Price"))
    sale_price = _parse_float(row.get("Sale Price"))
    chosen_price = sale_price if sale_price is not None and sale_price > 0 else list_price
    if chosen_price is None or chosen_price <= 0:
        chosen_price = 0.0

    compare_at: float | None = None
    if list_price is not None and list_price > chosen_price:
        compare_at = list_price

    in_stock = (row.get("Stock Availability") or "").strip().lower() == "in stock"
    status = "Active" if in_stock else "Draft"
    inv_qty = 10 if in_stock else 0

    image_url = _first_image(row.get("Image Urls"))
    image_alt = title if image_url else None

    color = (row.get("Color") or "").strip()
    breadcrumbs = (row.get("Breadcrumbs") or "").strip()
    tags = _normalize_tags(
        [
            vendor,
            color,
            *(p.strip() for p in breadcrumbs.split("|")),
        ]
    )

    rating = _parse_float(row.get("Average Rating"))
    num_ratings = _parse_int(row.get("Number Of Ratings"))
    seller = (row.get("Seller Name") or "").strip()

    # Keep description short (Shopify import supports long body, but Kaggle 'Product Specs' is noisy).
    specs = (row.get("Product Specs") or "").strip()
    if len(specs) > 4000:
        specs = specs[:3997] + "..."

    pieces: list[str] = []
    if specs:
        pieces.append(specs)
    if seller:
        pieces.append(f"Seller: {seller}")
    if rating is not None:
        pieces.append(f"Rating: {rating}")
    if num_ratings is not None:
        pieces.append(f"Ratings count: {num_ratings}")
    if breadcrumbs:
        pieces.append(f"Category: {breadcrumbs.replace('|', ' > ')}")

    body_html = "<br>".join(re.sub(r"\s+", " ", p).strip() for p in pieces if p.strip())

    return FormattedProduct(
        title=title[:255],
        handle=handle[:255],
        body_html=body_html,
        vendor=vendor[:255],
        product_type=product_type,
        tags=tags,
        status=status,
        sku=sku[:255],
        barcode=None,
        price=f"{chosen_price:.2f}",
        compare_at_price=f"{compare_at:.2f}" if compare_at is not None else None,
        inventory_quantity=inv_qty,
        image_url=image_url,
        image_alt=image_alt,
        source_url=(row.get("Pageurl") or "").strip() or None,
        raw=row,
    )


def _read_shopify_header(template_path: Path) -> list[str]:
    with template_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    return header


def _build_shopify_row(header: list[str], p: FormattedProduct) -> dict[str, str]:
    out: dict[str, str] = {h: "" for h in header}

    out["Title"] = p.title
    out["URL handle"] = p.handle
    out["Description"] = p.body_html
    out["Vendor"] = p.vendor
    out["Type"] = p.product_type or ""
    out["Tags"] = p.tags
    out["Published on online store"] = "TRUE"
    out["Status"] = p.status

    out["SKU"] = p.sku
    out["Barcode"] = p.barcode or ""
    out["Option1 name"] = "Title"
    out["Option1 value"] = "Default Title"

    out["Price"] = p.price
    out["Compare-at price"] = p.compare_at_price or ""
    out["Charge tax"] = "TRUE"

    out["Inventory tracker"] = "shopify"
    out["Inventory quantity"] = str(p.inventory_quantity)
    out["Continue selling when out of stock"] = "DENY"

    out["Requires shipping"] = "TRUE"
    out["Fulfillment service"] = "manual"

    if p.image_url:
        out["Product image URL"] = p.image_url
        out["Image position"] = "1"
        out["Image alt text"] = p.image_alt or ""

    # Keep original URL as an ads/custom label (free-form-ish field in template)
    if p.source_url:
        host = urlparse(p.source_url).netloc
        out["Google Shopping / Custom label 0"] = host[:255]
        out["Google Shopping / Custom label 1"] = p.source_url[:255]

    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Kaggle e-commerce CSV into Shopify import CSV + products.json"
    )
    parser.add_argument(
        "--input",
        default="data/kaggle-datasets/E-Commerce and Retail Datasets - E-Commerce and Retail_2025.csv",
        help="Path to Kaggle CSV",
    )
    parser.add_argument(
        "--shopify-template",
        default=str(SHOPIFY_TEMPLATE_PATH),
        help="Path to Shopify template CSV (used for header)",
    )
    parser.add_argument(
        "--out-csv",
        default="data/shopify_import.csv",
        help="Path to write Shopify import-ready CSV",
    )
    parser.add_argument(
        "--out-json",
        default="data/products.json",
        help="Path to write normalized JSON products",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Max number of products to format (for fast iteration)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    template_path = Path(args.shopify_template)
    out_csv_path = Path(args.out_csv)
    out_json_path = Path(args.out_json)

    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")
    if not template_path.exists():
        raise SystemExit(f"Shopify template CSV not found: {template_path}")

    header = _read_shopify_header(template_path)

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)

    formatted: list[FormattedProduct] = []
    seen_handles: set[str] = set()

    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = _format_row(row)
            if p.handle in seen_handles:
                continue
            seen_handles.add(p.handle)
            formatted.append(p)
            if args.limit and len(formatted) >= args.limit:
                break

    with out_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for p in formatted:
            writer.writerow(_build_shopify_row(header, p))

    products_json: dict[str, Any] = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "source": str(input_path).replace("\\", "/"),
        "count": len(formatted),
        "products": [
            {
                "title": p.title,
                "handle": p.handle,
                "sku": p.sku,
                "vendor": p.vendor,
                "type": p.product_type,
                "tags": [t.strip() for t in p.tags.split(",") if t.strip()],
                "status": p.status,
                "price": float(p.price),
                "compare_at_price": float(p.compare_at_price) if p.compare_at_price else None,
                "inventory_quantity": p.inventory_quantity,
                "image_url": p.image_url,
                "source_url": p.source_url,
                "raw": {
                    "brand": p.raw.get("Brand"),
                    "breadcrumbs": p.raw.get("Breadcrumbs"),
                    "color": p.raw.get("Color"),
                    "rating": _parse_float(p.raw.get("Average Rating")),
                    "num_ratings": _parse_int(p.raw.get("Number Of Ratings")),
                    "seller": p.raw.get("Seller Name"),
                },
            }
            for p in formatted
        ],
    }

    out_json_path.write_text(json.dumps(products_json, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote Shopify CSV: {out_csv_path} ({len(formatted)} products)")
    print(f"Wrote JSON:       {out_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

