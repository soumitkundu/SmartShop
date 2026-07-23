import logging

from backend.config import settings
from backend.graph.modality import detect_modalities, modality_label
from backend.graph.state import AgentState
from backend.llm import generate_answer
from backend.processors.image import ImageEncodingError, encode_image
from backend.processors.voice import transcribe
from backend.rag.prompt import (
    OUT_OF_CATALOG_REPLY,
    build_catalog_prompt,
    build_retrieval_query,
    build_user_turn_summary,
    should_reject_query,
)
from backend.rag.retriever import ImageRetriever, TextRetriever, filter_in_stock, merge_retrieval_results

logger = logging.getLogger(__name__)

_text_retriever: TextRetriever | None = None
_image_retriever: ImageRetriever | None = None


def _get_text_retriever() -> TextRetriever:
    global _text_retriever
    if _text_retriever is None:
        _text_retriever = TextRetriever(settings.TEXT_INDEX_PATH)
    return _text_retriever


def _get_image_retriever() -> ImageRetriever | None:
    global _image_retriever
    if _image_retriever is None:
        try:
            _image_retriever = ImageRetriever(settings.CHROMA_PATH, settings.IMAGE_COLLECTION_NAME)
        except FileNotFoundError as exc:
            logger.warning("[retriever] image index unavailable: %s", exc)
            return None
    return _image_retriever


def route_inputs(state: AgentState) -> dict:
    """Entry node — validates and logs which modalities are present."""
    flags = detect_modalities(
        state.get("text_input"),
        state.get("audio_bytes"),
        state.get("image_bytes"),
    )
    logger.info(
        "[router] modality=%s session=%s history_turns=%d",
        modality_label(flags),
        state.get("session_id"),
        len(state.get("chat_history") or []) // 2,
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
    """Run CLIP encoding on uploaded or captured image bytes."""
    image_bytes = state.get("image_bytes")
    if not image_bytes:
        return {"node_trace": ["image"]}

    try:
        image_vector = encode_image(image_bytes)
        logger.info("[image] encoded vector dim=%d", len(image_vector))
        return {
            "image_vector": image_vector,
            "image_error": None,
            "node_trace": ["image"],
        }
    except ImageEncodingError as exc:
        logger.warning("[image] encoding failed: %s", exc)
        return {
            "image_vector": None,
            "image_error": str(exc),
            "node_trace": ["image"],
        }


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
    """Query text and/or image indexes and apply store-scoped guardrails."""
    fused_query = (state.get("fused_query") or "").strip()
    image_vector = state.get("image_vector")
    chat_history = state.get("chat_history") or []
    retrieval_query = build_retrieval_query(
        fused_query,
        chat_history,
        has_image=bool(image_vector),
    )

    text_retriever = _get_text_retriever()
    image_retriever = _get_image_retriever()
    exclude_oos = settings.EXCLUDE_OUT_OF_STOCK
    top_k = settings.TOP_K_RESULTS
    fetch_k = top_k * 4 if exclude_oos else top_k

    hits: list[dict] = []
    if image_vector and retrieval_query:
        text_hits = text_retriever.search(
            retrieval_query,
            top_k=fetch_k,
            exclude_out_of_stock=False,
        )
        if image_retriever is not None:
            image_hits = image_retriever.search(
                image_vector,
                top_k=fetch_k,
                exclude_out_of_stock=False,
            )
            hits = merge_retrieval_results(
                text_hits,
                image_hits,
                fetch_k,
                rrf_k=settings.FUSION_RRF_K,
                text_weight=settings.FUSION_TEXT_WEIGHT,
                image_weight=settings.FUSION_IMAGE_WEIGHT,
            )
        else:
            hits = text_hits
    elif image_vector and image_retriever is not None:
        hits = image_retriever.search(
            image_vector,
            top_k=fetch_k,
            exclude_out_of_stock=exclude_oos,
        )
    elif retrieval_query:
        hits = text_retriever.search(
            retrieval_query,
            top_k=fetch_k,
            exclude_out_of_stock=exclude_oos,
        )

    if exclude_oos and image_vector and retrieval_query:
        hits = filter_in_stock(hits, top_k)
    elif exclude_oos and hits:
        hits = hits[:top_k]

    prompt_query = retrieval_query or fused_query or ("visual product match" if image_vector else "")
    rejected = should_reject_query(prompt_query, hits)

    logger.info(
        "[retriever] fused=%r retrieval=%r image=%s hits=%d rejected=%s top_score=%s",
        fused_query[:120] if fused_query else "",
        retrieval_query[:120] if retrieval_query else "",
        bool(image_vector),
        len(hits),
        rejected,
        hits[0].get("score") if hits else None,
    )

    context = None if rejected else build_catalog_prompt(prompt_query, hits, chat_history)
    return {
        "retrieved_docs": hits,
        "context": context,
        "rejected": rejected,
        "node_trace": ["retriever"],
    }


async def generate_response(state: AgentState) -> dict:
    """Call the LLM with retrieved context, or return the out-of-catalog reply."""
    flags = detect_modalities(
        state.get("text_input"),
        state.get("audio_bytes"),
        state.get("image_bytes"),
    )
    user_turn = build_user_turn_summary(
        state.get("text_input"),
        state.get("transcribed_text"),
        has_image=flags["has_image"],
    )

    if state.get("image_error") and not state.get("fused_query") and not state.get("retrieved_docs"):
        logger.info("[generator] image processing failed with no text fallback")
        response = (
            "I could not process the uploaded image. "
            f"{state['image_error']} "
            "Try a clearer JPEG/PNG photo or add a text description."
        )
        return {
            "response": response,
            "provider": None,
            "user_turn_summary": user_turn,
            "node_trace": ["generator"],
        }

    if state.get("rejected"):
        logger.info("[generator] skipped — out-of-catalog query")
        return {
            "response": OUT_OF_CATALOG_REPLY,
            "provider": None,
            "user_turn_summary": user_turn,
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
            "user_turn_summary": user_turn,
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
            "user_turn_summary": user_turn,
            "node_trace": ["generator"],
        }
