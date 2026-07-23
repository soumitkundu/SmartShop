# SmartShop — Hugging Face Spaces (Docker SDK)
# Space: https://huggingface.co/spaces/soumitkundu/smartshop
#
# Single container: FastAPI (8781) + Chainlit (7860) via deployment/start.sh
# Catalog artifacts (products.json, text_index.json, chroma_db/) built at image build time.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BACKEND_SEARCH_URL=http://127.0.0.1:8781/api/search \
    LANGCHAIN_TRACING_V2=false \
    EXCLUDE_OUT_OF_STOCK=true \
    WHISPER_MODEL=base \
    CLIP_MODEL=ViT-B/32 \
    CHROMA_PATH=./chroma_db \
    PRODUCT_CATALOG_PATH=./data/products.json \
    TEXT_INDEX_PATH=./data/text_index.json

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# setuptools>=82 dropped pkg_resources; openai-whisper still imports it at build time.
# --no-build-isolation uses the pinned setuptools already installed in the image.
RUN pip install --upgrade pip "setuptools<81" wheel \
    && pip install --no-build-isolation -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend
COPY scripts ./scripts
COPY data ./data
COPY public ./public
COPY chainlit.md ./chainlit.md
COPY .chainlit ./.chainlit
COPY deployment ./deployment

# Normalize CRLF -> LF (Windows checkout) so bash can parse set -o pipefail.
RUN sed -i 's/\r$//' deployment/*.sh \
    && chmod +x deployment/start.sh deployment/build_catalog.sh deployment/preload_models.sh

# 1) Cache Whisper + CLIP weights before indexing (reduces runtime cold start).
RUN bash deployment/preload_models.sh

# 2) Build bundled catalog + retrieval indexes inside the image.
ARG CATALOG_LIMIT=50
ENV CATALOG_LIMIT=${CATALOG_LIMIT}
RUN bash deployment/build_catalog.sh

EXPOSE 7860

CMD ["bash", "deployment/start.sh"]
