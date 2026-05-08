from typing import Any


OUT_OF_CATALOG_REPLY = (
    "I can only help with products from this store catalog. "
    "Please ask about available products, price, stock, brand, or category."
)


def should_reject_query(query: str, hits: list[dict[str, Any]]) -> bool:
    if not hits:
        return True
    if hits[0].get("score", 0.0) < 0.08:
        return True
    return False


def build_catalog_prompt(query: str, hits: list[dict[str, Any]]) -> str:
    product_lines: list[str] = []
    for i, item in enumerate(hits, start=1):
        title = item.get("title", "Unknown")
        price = item.get("price")
        stock = item.get("inventory_quantity")
        vendor = item.get("vendor") or "Unknown"
        item_type = item.get("type") or "Unknown"
        tags = ", ".join(item.get("tags") or [])
        product_lines.append(
            f"{i}. {title} | price={price} | stock={stock} | vendor={vendor} | "
            f"type={item_type} | tags={tags}"
        )

    context = "\n".join(product_lines)
    return (
        "You are SmartShop assistant.\n"
        "Rules:\n"
        "- Answer ONLY from the catalog context.\n"
        "- If information is missing, say it is not available in the catalog.\n"
        "- Do not invent products or prices.\n"
        "- Keep response concise and helpful.\n\n"
        f"User query: {query}\n\n"
        "Catalog context:\n"
        f"{context}\n"
    )
