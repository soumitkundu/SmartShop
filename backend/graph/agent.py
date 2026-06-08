import logging
import os

from langgraph.graph import END, StateGraph

from backend.config import settings
from backend.graph.nodes import (
    fuse_inputs,
    generate_response,
    process_image,
    process_voice,
    retrieve_products,
    route_inputs,
)
from backend.graph.state import AgentState

logger = logging.getLogger(__name__)

_agent = None


def _configure_langsmith() -> None:
    """Enable LangSmith tracing when configured with a valid API key."""
    if not settings.LANGCHAIN_TRACING_V2:
        logger.info("LangSmith tracing disabled (LANGCHAIN_TRACING_V2=false)")
        return

    if not settings.LANGCHAIN_API_KEY:
        logger.warning(
            "LangSmith tracing requested but LANGCHAIN_API_KEY is missing - "
            "add it to .env to send traces to project=%s",
            settings.LANGCHAIN_PROJECT,
        )
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT

    logger.info("LangSmith tracing enabled (project=%s)", settings.LANGCHAIN_PROJECT)


def decide_next_nodes(state: AgentState) -> str:
    """Route to voice/image processors only when those inputs are present."""
    if state.get("audio_bytes"):
        return "voice"
    if state.get("image_bytes"):
        return "image"
    return "fuser"


def build_agent():
    """Compile the LangGraph agent (text path: router → fuser → retriever → generator)."""
    _configure_langsmith()

    graph = StateGraph(AgentState)

    graph.add_node("router", route_inputs)
    graph.add_node("voice", process_voice)
    graph.add_node("image", process_image)
    graph.add_node("fuser", fuse_inputs)
    graph.add_node("retriever", retrieve_products)
    graph.add_node("generator", generate_response)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        decide_next_nodes,
        {
            "voice": "voice",
            "image": "image",
            "fuser": "fuser",
        },
    )

    graph.add_edge("voice", "fuser")
    graph.add_edge("image", "fuser")
    graph.add_edge("fuser", "retriever")
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", END)

    return graph.compile()


def get_agent():
    """Return a singleton compiled agent."""
    global _agent
    if _agent is None:
        _agent = build_agent()
        logger.info("LangGraph agent compiled")
    return _agent
