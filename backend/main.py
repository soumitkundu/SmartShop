import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.rag.prompt import OUT_OF_CATALOG_REPLY, build_catalog_prompt, should_reject_query
from backend.rag.retriever import TextRetriever

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


def _load_retriever() -> TextRetriever:
    try:
        return TextRetriever(settings.TEXT_INDEX_PATH)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


async def _call_groq(prompt: str) -> str:
    if not settings.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing")

    headers = {
        "Authorization": f"Bearer {settings.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    async with httpx.AsyncClient(timeout=25.0) as client:
        resp = await client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Groq error: {resp.status_code} {resp.text}")
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


async def _call_gemini(prompt: str) -> str:
    if not settings.GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is missing")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.GEMINI_MODEL}:generateContent?key={settings.GOOGLE_API_KEY}"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    async with httpx.AsyncClient(timeout=25.0) as client:
        resp = await client.post(url, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini error: {resp.status_code} {resp.text}")
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()


async def _generate_answer(prompt: str) -> tuple[str, str]:
    providers = ["groq", "gemini"] if settings.LLM_PROVIDER == "groq" else ["gemini", "groq"]
    last_error = "No provider configured"

    for provider in providers:
        try:
            if provider == "groq":
                return await _call_groq(prompt), "groq"
            return await _call_gemini(prompt), "gemini"
        except Exception as e:
            last_error = str(e)
            continue

    raise RuntimeError(last_error)


@app.post("/api/search")
async def search(
    text: str | None = Form(None),
    audio_file: UploadFile | None = File(None),
    image_file: UploadFile | None = File(None),
    session_id: str = Form(...),
):
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="`text` is required for Phase 2.")
    if audio_file is not None or image_file is not None:
        raise HTTPException(status_code=400, detail="Audio/image is not supported in Phase 2.")

    retriever = _load_retriever()
    hits = retriever.search(text, top_k=settings.TOP_K_RESULTS)

    if should_reject_query(text, hits):
        return {
            "session_id": session_id,
            "rejected": True,
            "answer": OUT_OF_CATALOG_REPLY,
            "products": [],
            "provider": None,
        }

    prompt = build_catalog_prompt(text, hits)
    try:
        answer, provider = await _generate_answer(prompt)
    except Exception:
        provider = None
        answer = (
            "I found matching items from the store catalog, but LLM generation is unavailable. "
            "Returning top products directly."
        )

    return {
        "session_id": session_id,
        "rejected": False,
        "answer": answer,
        "products": hits,
        "provider": provider,
    }

