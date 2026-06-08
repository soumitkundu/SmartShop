import httpx

from backend.config import settings


async def call_groq(prompt: str) -> str:
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
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Groq error: {resp.status_code} {resp.text}")
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


async def call_gemini(prompt: str) -> str:
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


async def generate_answer(prompt: str) -> tuple[str, str]:
    providers = ["groq", "gemini"] if settings.LLM_PROVIDER == "groq" else ["gemini", "groq"]
    last_error = "No provider configured"

    for provider in providers:
        try:
            if provider == "groq":
                return await call_groq(prompt), "groq"
            return await call_gemini(prompt), "gemini"
        except Exception as e:
            last_error = str(e)
            continue

    raise RuntimeError(last_error)
