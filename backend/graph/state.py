import operator
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):
    # Raw inputs
    text_input: str | None
    audio_bytes: bytes | None
    audio_suffix: str | None
    image_bytes: bytes | None
    session_id: str

    # Processed signals (voice/image filled in later phases)
    transcribed_text: str | None
    image_vector: list[float] | None
    fused_query: str | None

    # RAG outputs
    retrieved_docs: list[dict[str, Any]]
    context: str | None
    rejected: bool

    # Final output
    response: str | None
    provider: str | None
    chat_history: list[Any]

    # Observability — each node appends its name
    node_trace: Annotated[list[str], operator.add]
