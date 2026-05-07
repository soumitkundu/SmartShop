from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings

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


@app.post("/api/search")
async def search(
    text: str | None = Form(None),
    audio_file: UploadFile | None = File(None),
    image_file: UploadFile | None = File(None),
    session_id: str = Form(...),
):
    raise HTTPException(
        status_code=501,
        detail="Not implemented yet (Phase 0 stub).",
    )

