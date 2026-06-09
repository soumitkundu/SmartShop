import logging

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.graph.agent import get_agent
from backend.graph.state import AgentState
from backend.rag.retriever import TextRetriever

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

app = FastAPI(title=settings.APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


def _ensure_text_index() -> None:
    try:
        TextRetriever(settings.TEXT_INDEX_PATH)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/search")
async def search(
    text: str | None = Form(None),
    audio_file: UploadFile | None = File(None),
    image_file: UploadFile | None = File(None),
    session_id: str = Form(...),
):
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="`text` is required for Phase 3.")
    if audio_file is not None or image_file is not None:
        raise HTTPException(status_code=400, detail="Audio/image is not supported until Phase 4/5.")

    _ensure_text_index()

    initial_state: AgentState = {
        "text_input": text.strip(),
        "audio_bytes": None,
        "audio_suffix": None,
        "image_bytes": None,
        "session_id": session_id,
        "transcribed_text": None,
        "image_vector": None,
        "fused_query": None,
        "retrieved_docs": [],
        "context": None,
        "rejected": False,
        "response": None,
        "provider": None,
        "chat_history": [],
        "node_trace": [],
    }

    agent = get_agent()
    result = await agent.ainvoke(initial_state)

    node_trace = result.get("node_trace") or []
    logging.getLogger(__name__).info("Graph node_trace: %s", " -> ".join(node_trace))

    return {
        "session_id": session_id,
        "rejected": bool(result.get("rejected")),
        "answer": result.get("response"),
        "products": result.get("retrieved_docs") or [],
        "provider": result.get("provider"),
        "node_trace": node_trace,
    }
