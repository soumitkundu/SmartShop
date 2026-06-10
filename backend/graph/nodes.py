import logging

from backend.config import settings
from backend.graph.state import AgentState
from backend.llm import generate_answer
from backend.processors.voice import transcribe
from backend.rag.prompt import OUT_OF_CATALOG_REPLY, build_catalog_prompt, should_reject_query
from backend.rag.retriever import TextRetriever

logger = logging.getLogger(__name__)

_retriever: TextRetriever | None = None


def _get_retriever() -> TextRetriever:
    global _retriever
    if _retriever is None:
        _retriever = TextRetriever(settings.TEXT_INDEX_PATH)
    return _retriever


def route_inputs(state: AgentState) -> dict:
    """Entry node — validates and logs which modalities are present."""
    logger.info(
        "[router] text=%s audio=%s image=%s session=%s",
        bool(state.get("text_input")),
        bool(state.get("audio_bytes")),
        bool(state.get("image_bytes")),
        state.get("session_id"),
    )
    return {"node_trace": ["router"]}


def process_voice(state: AgentState) -> dict:
    """Run Whisper STT on uploaded or recorded audio."""
    audio_bytes = state.get("audio_bytes")
    if not audio_bytes:
        return {"node_trace": ["voice"]}

    suffix = state.get("audio_suffix") or ".wav"
    transcribed = transcribe(audio_bytes, suffix=suffix)
    logger.info("[voice] transcribed=%r", transcribed[:120] if transcribed else "")
    return {
        "transcribed_text": transcribed,
        "node_trace": ["voice"],
    }


def process_image(state: AgentState) -> dict:
    """CLIP encoding — implemented in Phase 5."""
    logger.warning("[image] Image processing is not enabled until Phase 5")
    return {"node_trace": ["image"]}


def fuse_inputs(state: AgentState) -> dict:
    """Combine all text signals into a single query string."""
    parts: list[str] = []
    if state.get("text_input"):
        parts.append(state["text_input"])
    if state.get("transcribed_text"):
        parts.append(state["transcribed_text"])

    fused = " ".join(parts).strip()
    logger.info("[fuser] fused_query=%r", fused[:120] if fused else "")
    return {
        "fused_query": fused,
        "node_trace": ["fuser"],
    }


def retrieve_products(state: AgentState) -> dict:
    """Query the text index and apply store-scoped guardrails."""
    query = (state.get("fused_query") or "").strip()
    retriever = _get_retriever()
    hits = retriever.search(query, top_k=settings.TOP_K_RESULTS)
    rejected = should_reject_query(query, hits)

    logger.info(
        "[retriever] query=%r hits=%d rejected=%s top_score=%s",
        query[:120] if query else "",
        len(hits),
        rejected,
        hits[0].get("score") if hits else None,
    )

    context = None if rejected else build_catalog_prompt(query, hits)
    return {
        "retrieved_docs": hits,
        "context": context,
        "rejected": rejected,
        "node_trace": ["retriever"],
    }


async def generate_response(state: AgentState) -> dict:
    """Call the LLM with retrieved context, or return the out-of-catalog reply."""
    if state.get("rejected"):
        logger.info("[generator] skipped — out-of-catalog query")
        return {
            "response": OUT_OF_CATALOG_REPLY,
            "provider": None,
            "node_trace": ["generator"],
        }

    prompt = state.get("context") or ""
    query = state.get("fused_query") or ""

    try:
        answer, provider = await generate_answer(prompt)
        logger.info("[generator] provider=%s query=%r", provider, query[:80])
        return {
            "response": answer,
            "provider": provider,
            "node_trace": ["generator"],
        }
    except Exception as exc:
        logger.warning("[generator] LLM unavailable: %s", exc)
        return {
            "response": (
                "I found matching items from the store catalog, but LLM generation is unavailable. "
                "Returning top products directly."
            ),
            "provider": None,
            "node_trace": ["generator"],
        }
