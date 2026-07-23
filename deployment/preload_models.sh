#!/usr/bin/env bash
# Pre-download Whisper + CLIP weights at image build time (Phase 8).
set -euo pipefail

WHISPER_MODEL="${WHISPER_MODEL:-base}"
CLIP_MODEL="${CLIP_MODEL:-ViT-B/32}"

echo "[build] pre-downloading Whisper model=${WHISPER_MODEL}..."
python -c "import os, whisper; whisper.load_model(os.environ['WHISPER_MODEL'])"

echo "[build] pre-downloading CLIP model=${CLIP_MODEL}..."
python -c "import os, clip, torch; clip.load(os.environ['CLIP_MODEL'], device='cpu')"

echo "[build] local models cached"
