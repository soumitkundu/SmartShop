from typing import Any


FOLLOW_UP_CUES = (
    "it",    "its",
    "that",
    "those",
    "them",
    "one",
    "ones",
    "other",
    "others",
    "similar",
    "cheaper",
    "expensive",
    "compare",
    "first",
    "second",
    "third",
    "more",
    "another",
    "instead",
    "which",
    "what about",
    "how about",
)


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


def format_chat_history(chat_history: list[dict[str, Any]] | None) -> str:
    if not chat_history:
        return ""

    lines: list[str] = []
    for msg in chat_history:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content}")

    if not lines:
        return ""

    return "Previous conversation:\n" + "\n".join(lines) + "\n\n"


def is_follow_up_query(query: str) -> bool:
    normalized = query.lower().strip()
    if not normalized:
        return False
    if len(normalized.split()) <= 6:
        return any(cue in normalized for cue in FOLLOW_UP_CUES)
    return False


def is_standalone_new_topic(query: str) -> bool:
    """Detect general-knowledge questions that should not inherit shopping context."""
    normalized = query.lower().strip()
    if not normalized:
        return False

    standalone_starts = ("who ", "when ", "where ", "why ", "how ")
    if normalized.endswith("?") and normalized.startswith(standalone_starts):
        return True

    standalone_markers = ("won the", "capital of", "president of", "weather in")
    return any(marker in normalized for marker in standalone_markers)


def should_expand_with_history(query: str) -> bool:
    if not query:
        return True
    if is_standalone_new_topic(query):
        return False
    if is_follow_up_query(query):
        return True
    return len(query.split()) <= 4


def build_retrieval_query(
    fused_query: str,
    chat_history: list[dict[str, Any]] | None,
    *,
    has_image: bool = False,
) -> str:
    """Expand short follow-up queries with recent user context for retrieval."""
    query = fused_query.strip()
    history = chat_history or []

    if not history:
        return query

    if query and not should_expand_with_history(query):
        return query

    prior_users = [
        (msg.get("content") or "").strip()
        for msg in history
        if msg.get("role") == "user" and (msg.get("content") or "").strip()
    ][-2:]

    parts = [*prior_users]
    if query:
        parts.append(query)
    elif has_image:
        parts.append("visual product match")

    return " ".join(parts).strip()


def build_user_turn_summary(
    text_input: str | None,
    transcribed_text: str | None,
    *,
    has_image: bool = False,
) -> str:
    parts: list[str] = []
    if text_input and text_input.strip():
        parts.append(text_input.strip())
    if transcribed_text and transcribed_text.strip():
        parts.append(transcribed_text.strip())
    if has_image:
        parts.append("[image attached]")
    return " ".join(parts).strip()


def build_catalog_prompt(
    query: str,
    hits: list[dict[str, Any]],
    chat_history: list[dict[str, Any]] | None = None,
) -> str:
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
    history_block = format_chat_history(chat_history)
    return (
        "You are SmartShop assistant.\n"
        "Rules:\n"
        "- Answer ONLY from the catalog context.\n"
        "- If information is missing, say it is not available in the catalog.\n"
        "- Do not invent products or prices.\n"
        "- Only recommend in-stock items from the context; never suggest zero-stock products.\n"
        "- Use prior conversation only to resolve follow-up references (e.g. 'the first one').\n"
        "- Keep response concise and helpful.\n\n"
        f"{history_block}"
        f"User query: {query}\n\n"
        "Catalog context:\n"
        f"{context}\n"
    )
