# SmartShop
This is upgraded version of initial repo ShopSenseAI with ZERO development cost, RAG implemented on Shopify store. The bot will search products from its own store only.

## Phase 0 — Environment Setup

**Phase 0**: **Python 3.11** Environment Creation with `.venv`:

- Created a fresh `.venv` using `py -3.11`.
- Installed dependencies from `requirements.txt`.
- Smoke-tested backend import (`backend.main`).
- Started FastAPI and verified health endpoint returns:
  - GET /api/health → {"status":"ok"}

**Implemented changes**
- Added `.gitignore`
- Added `requirements.txt` (Phase 0 minimal set)
- Added `backend/__init__.py`
- Added `backend/config.py`
- Added `backend/main.py` with:
  - `GET /api/health` working
  - `POST /api/search` stub (501 for now)

---

## Phase 1 — Data Ingestion (Kaggle → Shopify → JSON)

**Implemented (code)**

Added:

- `scripts/format_kaggle.py`: reads Kaggle CSV and writes
  - `data/shopify_import.csv` (Shopify import-ready)
  - `data/products.json` (normalized JSON for later RAG phases)
- `scripts/sync_shopify.py`: reads `data/shopify_import.csv`, **paginates** all existing Shopify products (REST `since_id`), and **creates missing products** with retries/backoff for 429/5xx.
- Updated `.gitignore` to ignore generated artifacts:
  - `data/shopify_import.csv`
  - `data/shopify_sync_report.json` 
  - `data/products.json`

---

### Instruction to create the Shopify custom app token

Open Shopify admin for `{shop}.myshopify.com`:

1. Go to **Settings → Apps and sales channels**
2. Click **Develop apps** (it may need to enable for the first time)
3. Click **Create an app** → name it e.g. `SmartShop Sync`
4. Open the app → **Configuration**
5. Under **Admin API integration**, click **Configure**
6. Enable these **Admin API scopes**:
   - **Products**: `read_products`, `write_products`
   - **Inventory**: `read_inventory`, `write_inventory`
   - (Recommended) **Locations**: `read_locations` (helps with inventory-related flows)
7. Click **Save**
8. Go to **API credentials** → click **Install app**
9. Copy the **Admin API access token** (this is what the script uses)

Now create a local `.env` file in the repo root (`SmartShop/.env`) with:

```env
SHOPIFY_STORE_DOMAIN={shop}.myshopify.com
SHOPIFY_ADMIN_API_ACCESS_TOKEN=shpat_...paste_yours...
SHOPIFY_API_VERSION=2025-01
```

add `.env` in `.gitignore`

---

### Run Phase 1

Install deps (prefer a venv if you can):

```powershell
py -m pip install -r requirements.txt
```

Generate the import CSV + JSON (start small first):

```powershell
py -m scripts.format_kaggle --limit 50
```

Sync into Shopify:

```powershell
py -m scripts.sync_shopify --limit 50
```

- This will create `data/shopify_sync_report.json` with a summary + any failures.
- To recreate products that already exist (matched by **handle**), run:

```powershell
py -m scripts.sync_shopify --limit 50 --overwrite
```

---

## Phase 2 — Text RAG Core (No multimodal yet)

1. Create environment file from template and fill keys:
   - copy `example.env` to `.env`
   - add either `GROQ_API_KEY` or `GOOGLE_API_KEY` (or both)
2. Build normalized catalog (if missing):
   - `py scripts/format_kaggle.py --limit 200`
3. Build text index:
   - `py scripts/embed_products.py --input data/products.json --out data/text_index.json`
4. Run API:
   - `uvicorn backend.main:app --reload`

`POST /api/search` now supports text-only queries and rejects out-of-catalog questions.

### Phase 2 test script

Run:

- `py scripts/test_text_rag.py`

Optional custom queries:

- `py scripts/test_text_rag.py --query "show me nike shoes" --reject-query "what is the weather today"`

#### Test Text RAG

**Groq**

```powershell
python -c "from dotenv import load_dotenv; load_dotenv(); import os, requests; k=os.getenv('GROQ_API_KEY'); assert k, 'Missing GROQ_API_KEY'; r=requests.post('https://api.groq.com/openai/v1/chat/completions', headers={'Authorization':f'Bearer {k}','Content-Type':'application/json'}, json={'model':'llama-3.1-8b-instant','messages':[{'role':'user','content':'Reply only: OK_GROQ'}]}); print('Groq:', r.status_code, r.json()['choices'][0]['message']['content'])"
```

**Gemini**

```powershell
py -c "from dotenv import load_dotenv; load_dotenv(); import os, json, requests; k=os.getenv('GOOGLE_API_KEY'); assert k, 'Missing GOOGLE_API_KEY'; m=os.getenv('GEMINI_MODEL','gemini-2.0-flash-lite'); u=f'https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={k}'; r=requests.post(u, json={'contents':[{'parts':[{'text':'Reply only: OK_GEMINI'}]}]}); print('Status:', r.status_code); data=r.json(); print('Model:', m); txt=(data.get('candidates') or [{}])[0].get('content',{}).get('parts',[{}])[0].get('text'); print('Reply:', txt if txt else '<no reply>'); err=data.get('error'); print('Error:', json.dumps(err, indent=2) if err else '<none>')"
```

---

### Instruction to Create and Setup Free API Keys

**`Groq` key (free tier)**
1. Go to [console.groq.com](https://console.groq.com/).
2. Sign up/login (GitHub or Google works).
3. Open **API Keys** section.
4. Click **Create API Key**.
5. Copy key (starts with `gsk_...`) and save it (shown only once).

**`Gemini` key (free tier)**
1. Go to [Google AI Studio](https://aistudio.google.com/).
2. Login with Google account.
3. Open **Get API key** / **API Keys**.
4. Click **Create API key** (or create in new/existing project).
5. Copy key (usually starts with `AIza...`).

---

**Where to add these in the project**

In project root, create / edit `.env` from `example.env`, then fill:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...your_real_key...
GOOGLE_API_KEY=AIza...your_real_key...
```

Optional model fields (already in `example.env`):
```env
GROQ_MODEL=llama-3.1-8b-instant
GEMINI_MODEL=gemini-2.0-flash-lite
```

- TO set Groq primary + Gemini fallback, keep `LLM_PROVIDER=groq`.
- To set Gemini primary + Groq fallback, set `LLM_PROVIDER=gemini`.

---

## Phase 3 — LangGraph Skeleton (Text path through graph)

Text queries now flow through a LangGraph agent:

`router -> fuser -> retriever -> generator`

LangSmith tracing is **enabled by default** (`LANGCHAIN_TRACING_V2=true`).

### Setup

1. Add LangSmith credentials to `.env` (from `example.env`):
   - `LANGCHAIN_API_KEY=lsv2_...`
   - `LANGCHAIN_PROJECT=SmartShop`
2. Install Phase 3 dependencies:
   - `py -m pip install -r requirements.txt`
3. Run API:
   - `uvicorn backend.main:app --reload`

### Phase 3 test script

```powershell
py scripts/test_langgraph.py
```

This verifies:
- Same `/api/search` response contract as Phase 2
- Node execution order: `router -> fuser -> retriever -> generator`
- Out-of-catalog rejection still works

### LangSmith

1. Sign up at [smith.langchain.com](https://smith.langchain.com)
2. Create an API key under **Settings → API Keys**
3. Add it to `.env` as `LANGCHAIN_API_KEY`
4. After a query, open your **SmartShop** project in LangSmith to inspect per-node traces

---

## Phase 4 — Voice Input (Whisper)

Voice queries are transcribed locally with **OpenAI Whisper** (offline, no API cost) and routed through the LangGraph `voice` node before retrieval.

Supported input modes on `POST /api/search`:

- **Text only** (unchanged from Phase 3)
- **Voice only** — upload `audio_file` (WAV, MP3, OGG, M4A, etc.)
- **Text + voice** — typed `text` is fused with the Whisper transcription

Image upload is still blocked until Phase 5.

### Graph paths

- Text only: `router -> fuser -> retriever -> generator`
- Voice (with or without text): `router -> voice -> fuser -> retriever -> generator`

### Implemented (code)

Added:

- `backend/processors/voice.py` — Whisper model load + `transcribe(audio_bytes, suffix)`
- Wired `process_voice` in `backend/graph/nodes.py`
- Updated `POST /api/search` in `backend/main.py` to accept `audio_file`
- `scripts/test_voice.py` — Phase 4 validation script
- `openai-whisper` in `requirements.txt`
- `WHISPER_MODEL` in `example.env` (default: `base`)

### Prerequisites

1. **ffmpeg** must be on your PATH (Whisper uses it to decode audio).
   - Verify: `ffmpeg -version`
2. Add to `.env` (optional — `base` is the default):

```env
WHISPER_MODEL=base   # tiny | base | small | medium
```

| Model  | Size  | Speed (CPU) | Recommended for        |
|--------|-------|-------------|------------------------|
| `tiny` | 39 MB | Very fast   | Quick dev smoke tests  |
| `base` | 74 MB | Fast        | **Default — best balance** |
| `small`| 244 MB| Moderate    | Better accuracy        |

### Install Phase 4 dependencies

```powershell
py -m pip install -r requirements.txt
```

On Windows, if `openai-whisper` fails to build:

```powershell
py -m pip install setuptools wheel
py -m pip install --no-build-isolation openai-whisper==20240930
```

The first voice request downloads the Whisper model (~74 MB for `base`).

### Phase 4 test script

Text regression (no audio file needed):

```powershell
py scripts/test_voice.py
```

Full voice integration (provide your own audio, or generate a sample):

```powershell
py -m pip install gtts
py -c "from gtts import gTTS; gTTS('show me walking shoes under two thousand').save('data/test_voice_query.mp3')"
py scripts/test_voice.py --audio-file data/test_voice_query.mp3
```

This verifies:

- Text-only path still works (`router -> fuser -> retriever -> generator`)
- Voice-only transcription and search (`router -> voice -> fuser -> retriever -> generator`)
- Text + voice fusion in `fused_query`
- API returns `transcribed_text` and `fused_query` in the JSON response

### Manual API test

Start the server:

```powershell
uvicorn backend.main:app --reload
```

Send a multipart form request to `POST /api/search` with:

- `session_id` (required)
- `audio_file` (optional) — recorded or uploaded audio
- `text` (optional) — at least one of `text` or `audio_file` is required

Example response fields:

```json
{
  "session_id": "...",
  "transcribed_text": "show me walking shoes under 2000",
  "fused_query": "budget option show me walking shoes under 2000",
  "answer": "...",
  "products": [...],
  "node_trace": ["router", "voice", "fuser", "retriever", "generator"]
}
```
