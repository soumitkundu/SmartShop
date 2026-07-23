# SmartShop — Hugging Face Spaces Deployment

Deploy **soumitkundu/smartshop** from GitHub (recommended) so Hugging Face rebuilds on each push.

Public Space URL after deploy:

`https://huggingface.co/spaces/soumitkundu/smartshop`

---

## 1. Create the Space (one-time)

1. Open [https://huggingface.co/new-space](https://huggingface.co/new-space)
2. Set:
   - **Owner:** `soumitkundu`
   - **Space name:** `smartshop`
   - **SDK:** `Docker`
   - **Hardware:** `CPU basic` (free)
   - **Visibility:** `Public`
3. Click **Create Space**

---

## 2. Connect GitHub repository

1. Open [https://huggingface.co/spaces/soumitkundu/smartshop/settings](https://huggingface.co/spaces/soumitkundu/smartshop/settings)
2. Under **Repository**, choose **Connect to GitHub**
3. Authorize Hugging Face if prompted
4. Select your SmartShop GitHub repository
5. Select branch: `main` (or your default deploy branch)
6. Save

From now on, every push to the linked branch triggers a Space rebuild.

> **Requirement:** `Dockerfile` must exist at the repository root (already added in Phase 8).

---

## 3. Add Space secrets

In Space **Settings → Repository secrets**, add:

| Secret | Required | Notes |
|--------|----------|-------|
| `GROQ_API_KEY` | Yes (or Gemini) | Primary LLM |
| `GOOGLE_API_KEY` | Optional | Gemini fallback |
| `LLM_PROVIDER` | Optional | `groq` (default) or `gemini` |
| `SHOPIFY_STORE_DOMAIN` | Recommended | e.g. `{shop}.myshopify.com` for product links in UI |
| `SHOPIFY_ADMIN_API_ACCESS_TOKEN` | Optional | Only needed for live sync scripts |
| `LANGCHAIN_API_KEY` | Optional | LangSmith tracing |
| `LANGCHAIN_TRACING_V2` | Optional | Set `false` on HF to reduce noise |
| `EXCLUDE_OUT_OF_STOCK` | Optional | Default `true` in Docker image |
| `BACKEND_SEARCH_URL` | Optional | Defaults to `http://127.0.0.1:8781/api/search` in container |

Do **not** commit `.env` to GitHub.

---

## 4. Container architecture

Single Docker container runs both services via `deployment/start.sh`:

1. **FastAPI backend** on `0.0.0.0:8781` (`/api/health`, `/api/search`)
2. **Chainlit frontend** on `0.0.0.0:7860` (public HF port)

Chainlit calls the backend using `BACKEND_SEARCH_URL` inside the container.

---

## 5. What the Docker image builds

Build pipeline (see root `Dockerfile`):

1. **`deployment/preload_models.sh`** — caches Whisper + CLIP weights at build time
2. **`deployment/build_catalog.sh`** — generates bundled data artifacts:
   - `data/products.json` ← `scripts/format_kaggle`
   - `data/text_index.json` ← `scripts/embed_products`
   - `chroma_db/shopify_images` ← `scripts/embed_product_images`
3. **`deployment/start.sh`** — runs FastAPI + Chainlit in one container at runtime

Default catalog size: **50 products** (`ARG CATALOG_LIMIT=50`). Increase only if you accept longer HF build times.

---

## 6. Space README card metadata

HF reads `README.md` frontmatter for the Space card. The repository `README.md` already includes:

```yaml
sdk: docker
app_port: 7860
```

No extra setup needed when deploying from GitHub.

---

## 7. Verify deployment

After build completes (Logs tab in HF Space):

1. Open the Space URL
2. Ask a catalog query in Chainlit
3. Confirm backend health inside container logs:
   - `[start] backend is healthy`
4. Optional local check before push:

```powershell
docker build -t smartshop .
docker run --rm -p 7860:7860 --env-file .env smartshop
```

Then open `http://localhost:7860`.

---

## 8. Troubleshooting

| Symptom | Likely fix |
|---------|------------|
| Build fails on CLIP/Whisper | Rebuild; ensure `ffmpeg` and `git` are available (already in Dockerfile) |
| Chainlit loads but queries fail | Add `GROQ_API_KEY` or `GOOGLE_API_KEY` in Space secrets |
| No product links in cards | Set `SHOPIFY_STORE_DOMAIN` secret |
| Slow first query | Expected — models warm up on first voice/image request |
| Image search empty | Rebuild image; check embed logs for skipped image URLs |

---

## 9. Evaluation before/after deploy

Run locally:

```powershell
py evaluation/ragas_eval.py
```

Optional RAGAS metrics (requires `pip install ragas datasets`):

```powershell
py evaluation/ragas_eval.py --ragas
```

Report is written to `evaluation/eval_report.json`.
