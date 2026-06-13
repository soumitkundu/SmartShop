import logging

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.graph.agent import get_agent
from backend.graph.state import AgentState
from backend.processors.image import ImageEncodingError
from backend.processors.voice import TranscriptionError
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


def _audio_suffix_from_filename(filename: str | None) -> str:
    if not filename or "." not in filename:
        return ".wav"
    return f".{filename.rsplit('.', 1)[-1].lower()}"


@app.post("/api/search")
async def search(
    text: str | None = Form(None),
    audio_file: UploadFile | None = File(None),
    image_file: UploadFile | None = File(None),
    session_id: str = Form(...),
):
    text_input = text.strip() if text and text.strip() else None
    audio_bytes: bytes | None = None
    audio_suffix: str | None = None
    image_bytes: bytes | None = None

    if audio_file is not None:
        audio_bytes = await audio_file.read()
        audio_suffix = _audio_suffix_from_filename(audio_file.filename)
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Audio file is empty.")

    if image_file is not None:
        image_bytes = await image_file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Image file is empty.")

    if not text_input and not audio_bytes and not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of `text`, `audio_file`, or `image_file`.",
        )

    _ensure_text_index()

    initial_state: AgentState = {
        "text_input": text_input,
        "audio_bytes": audio_bytes,
        "audio_suffix": audio_suffix,
        "image_bytes": image_bytes,
        "session_id": session_id,
        "transcribed_text": None,
        "image_vector": None,
        "image_error": None,
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
    try:
        result = await agent.ainvoke(initial_state)
    except TranscriptionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImageEncodingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    node_trace = result.get("node_trace") or []
    logging.getLogger(__name__).info("Graph node_trace: %s", " -> ".join(node_trace))

    return {
        "session_id": session_id,
        "rejected": bool(result.get("rejected")),
        "answer": result.get("response"),
        "transcribed_text": result.get("transcribed_text"),
        "image_error": result.get("image_error"),
        "fused_query": result.get("fused_query"),
        "products": result.get("retrieved_docs") or [],
        "provider": result.get("provider"),
        "node_trace": node_trace,
    }
