# SmartShop: Development Plan

# AI-Powered Multimodal Product Discovery Chatbot

### Text · Image · Voice — RAG over Shopify Dev Store

> **Zero-Cost Stack**

---

## Table of Contents

1. [Project Overview & Goals](#1-project-overview--goals)
2. [System Architecture](#2-system-architecture)
3. [Tech Stack & Tools](#3-tech-stack--tools)
4. [Data Pipeline — Shopify + Kaggle](#4-data-pipeline--shopify--kaggle)
5. [Multimodal Input System](#5-multimodal-input-system)
6. [RAG Pipeline Design](#6-rag-pipeline-design)
7. [LangGraph Agentic Workflow](#7-langgraph-agentic-workflow)
8. [Backend API](#8-backend-api)
9. [Frontend Chatbot UI](#9-frontend-chatbot-ui)
10. [Project Folder Structure](#10-project-folder-structure)
11. [Development Phases & Milestones](#11-development-phases--milestones)
12. [Environment Setup Guide](#12-environment-setup-guide)
13. [Evaluation & Testing](#13-evaluation--testing)
14. [Deployment](#14-deployment)
15. [Resume & Portfolio Tips](#15-resume--portfolio-tips)

---

## 1. Project Overview & Goals

### 1.1 What We Are Building

An **in-store AI Shopping Assistant** that lets users search for products using any combination of **text, voice, and image inputs** — all within a single chat interface. The chatbot retrieves results exclusively from your Shopify store's product catalog via a **RAG (Retrieval-Augmented Generation) pipeline**, guaranteeing answers are always grounded in real store inventory.

The agentic layer is powered by **LangGraph**, which orchestrates the multimodal processing steps as a stateful directed graph — making the system transparent, debuggable, and extensible.

### 1.2 Key Objectives

- Build a production-grade multimodal chatbot for an e-commerce website
- Demonstrate Shopify API integration, RAG architecture, and AI engineering skills
- Use **LangGraph** for agentic orchestration + **LangChain** for LLM/prompt utilities
- Keep the entire stack free — no paid cloud services required
- Deploy a publicly shareable demo URL

### 1.3 APP Features

| Feature                                                                     |
| --------------------------------------------------------------------------- |
| Shopify Admin API as live data source                                       |
| Multimodal input (text + voice + image)                                     |
| LangGraph agentic orchestration                                             |
| RAG scoped to store only (implement RAG guardrails and knowledge grounding) |
| Free full-stack deployment (Cost-conscious architecture)                    |
| Whisper for speech recognition                                              |
| CLIP for image embeddings (Vision-language model integration)               |

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                              USER                               │
│     Types text  /  Uploads image  /  Speaks or uploads audio    │
└─────────────────────────────┬───────────────────────────────────┘
                              │  Single multimodal request
                              |          (FormData)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CHAINLIT FRONTEND                           │
│    Text input · Mic recorder · Camera capture · File upload     │
└─────────────────────────────┬───────────────────────────────────┘
                              │  POST /api/search
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          FASTAPI BACKEND                        │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                     LANGGRAPH AGENT GRAPH                │   │
│  │                                                          │   │
│  │   [Input Router] ──► [Voice Processor (Whisper STT)]     │   │
│  │         │       ──► [Image Processor (CLIP)]             │   │
│  │         │       ──► [Text Processor]                     │   │
│  │         │                                                │   │
│  │         ▼                                                │   │
│  │   [Multimodal Fuser] ──► [RAG Retriever (ChromaDB)]      │   │
│  │                                ▼                         │   │
│  │                     [LLM Generator (Groq)]               │   │
│  │                                ▼                         │   │
│  │                     [Response Formatter]                 │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────────┘
                               │  Streamed response
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              CHROMADB (Local Persistent Vector Store)           │
│   shopify_text collection  ·  shopify_images collection         │
└─────────────────────────────────────────────────────────────────┘
                               ▲
                               │  Sync via GraphQL Admin API
┌─────────────────────────────────────────────────────────────────┐
│                      SHOPIFY DEV STORE                          │
│          Products · Variants · Inventory · Collections          │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow Summary

| Step | Component              | Input                     | Output                           |
| ---- | ---------------------- | ------------------------- | -------------------------------- |
| 1    | Shopify Sync Script    | Shopify Admin GraphQL API | Product JSON documents           |
| 2    | Embedding Pipeline     | Product text + images     | ChromaDB vector store            |
| 3    | LangGraph Input Router | User audio / image / text | Routed to correct processor node |
| 4    | Multimodal Fuser Node  | Processed signals         | Unified query vector(s)          |
| 5    | RAG Retriever Node     | Query vector              | Top-K relevant products          |
| 6    | LLM Generator Node     | Products + user query     | Natural language response        |

---

## 3. Tech Stack & Tools

> 💡 **Cost Note:** Every tool listed below is 100% free for this project. No credit card required.

### 3.1 Backend

| Layer                     | Tool / Library                         | Purpose                                                | Cost      |
| ------------------------- | -------------------------------------- | ------------------------------------------------------ | --------- |
| API Server                | FastAPI (Python)                       | REST API — receives multimodal requests                | Free      |
| **Agentic Orchestration** | **LangGraph**                          | Stateful directed graph for multimodal agent workflow  | Free      |
| LLM & Prompt Utilities    | LangChain                              | Prompt templates, memory, LLM wrappers, output parsers | Free      |
| Speech-to-Text            | OpenAI Whisper (`openai-whisper` pip)  | Offline speech recognition — runs on CPU               | Free      |
| Image Embedding           | CLIP (`openai/clip` via transformers)  | Image-to-vector for image search                       | Free      |
| Text Embedding            | `sentence-transformers` (MiniLM-L6-v2) | Text-to-vector for product retrieval                   | Free      |
| Vector Database           | ChromaDB (local, persistent)           | Store & search product embeddings                      | Free      |
| LLM — Primary             | Groq API (LLaMA 3.3 70B)               | Fast inference for chat responses                      | Free tier |
| LLM — Backup              | Google Gemini 1.5 Flash API            | Backup LLM with generous free quota                    | Free tier |
| Shopify Client            | ShopifyAPI (pip)                       | Pull products from dev store                           | Free      |

### 3.2 LangChain vs LangGraph — Roles in This Project

These two libraries work **together**, not as alternatives. Here is exactly what each one does in this project:

| Library       | Role                | Used For                                                                                                                                                                                               |
| ------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **LangChain** | Utility layer       | Prompt templates, LLM wrappers (Groq/Gemini), output parsers, conversation memory (`ConversationBufferWindowMemory`), document loaders                                                                 |
| **LangGraph** | Orchestration layer | Defining the multimodal agent as a stateful graph — routing inputs, managing node execution order, handling conditional logic (e.g., skip image node if no image sent), state persistence across turns |

Think of it this way: **LangChain provides the building blocks; LangGraph assembles them into a controllable workflow.**

### 3.3 Frontend

| Layer        | Tool                     | Purpose                                           | Cost |
| ------------ | ------------------------ | ------------------------------------------------- | ---- |
| Chatbot UI   | Chainlit                 | Python-native chat UI — zero frontend code needed | Free |
| Mic Input    | Web Audio API (built-in) | Browser microphone access for live recording      | Free |
| Camera Input | MediaDevices API         | Camera capture — sends image to backend           | Free |
| File Upload  | Chainlit file upload     | User uploads audio file or product photo          | Free |

### 3.4 Data & Infrastructure

| Purpose              | Tool                         | Details                                                        |
| -------------------- | ---------------------------- | -------------------------------------------------------------- |
| Product data source  | Shopify Admin GraphQL API    | Live sync from dev store — free with Partner account           |
| Raw dataset          | Kaggle Flipkart / Amazon CSV | Initial product catalog — imported via Shopify CSV bulk upload |
| Evaluation framework | RAGAS                        | Measures RAG faithfulness, relevance, and recall               |
| Observability        | LangSmith (free tier)        | Trace and debug LangGraph node executions                      |
| Version control      | GitHub                       | Public repo                                                    |
| Deployment           | Hugging Face Spaces          | Free public sharable URL                                       |

---

## 4. Data Pipeline — Shopify + Kaggle

### 4.1 Step 1 — Prepare Kaggle Dataset

Download the **Flipkart Products Dataset** from Kaggle (search: `flipkart product dataset`). Alternatively, use the **Amazon Products Dataset** for richer descriptions.

Open the CSV in Python/Pandas and map columns to Shopify's import format:

```
Kaggle Column              →   Shopify CSV Column
──────────────────────────────────────────────────
product_name               →   Title
description                →   Body (HTML)
retail_price               →   Variant Price
product_category_tree      →   Type, Tags
image                      →   Image Src
(set stock = 50 for all)   →   Variant Inventory Qty
```

### 4.2 Step 2 — Shopify Dev Store Setup

1. Go to `partners.shopify.com` → Sign up free as a Shopify Partner
2. Create a new **Development Store** from the Partner Dashboard
3. In Shopify Admin → **Products → Import** → upload your formatted CSV
4. Create a **Custom App**: Settings → Apps → Develop Apps
5. Grant scopes: `read_products`, `read_inventory`, `read_collections`
6. Copy the **Admin API Access Token** — store in your `.env` file

### 4.3 Step 3 — Shopify Sync Script

```python
# scripts/sync_shopify.py
import shopify, json, os
from dotenv import load_dotenv

load_dotenv()

session = shopify.Session(
    os.getenv("SHOPIFY_STORE_URL"),
    "2024-10",
    os.getenv("SHOPIFY_ACCESS_TOKEN")
)
shopify.ShopifyResource.activate_session(session)

QUERY = """
{
  products(first: 250) {
    edges {
      node {
        id title descriptionHtml productType tags
        variants(first: 10) {
          edges { node { price inventoryQuantity sku } }
        }
        images(first: 1) {
          edges { node { url } }
        }
      }
    }
  }
}
"""

result = shopify.GraphQL().execute(QUERY)
products = json.loads(result)["data"]["products"]["edges"]

documents = []
for edge in products:
    p = edge["node"]
    variants = p["variants"]["edges"]
    price = variants[0]["node"]["price"] if variants else "N/A"
    stock = variants[0]["node"]["inventoryQuantity"] if variants else 0
    image = p["images"]["edges"][0]["node"]["url"] if p["images"]["edges"] else ""

    doc = {
        "id": p["id"],
        "title": p["title"],
        "type": p["productType"],
        "tags": ", ".join(p["tags"]),
        "price": price,
        "stock": stock,
        "image_url": image,
        "description": p["descriptionHtml"],
        # Full text chunk for embedding
        "text": f"{p['title']}. {p['productType']}. Tags: {', '.join(p['tags'])}. "
                f"Price: ₹{price}. In stock: {stock}. {p['descriptionHtml']}"
    }
    documents.append(doc)

with open("data/products.json", "w") as f:
    json.dump(documents, f, indent=2)

print(f"Synced {len(documents)} products.")
```

### 4.4 Step 4 — Embedding & Indexing into ChromaDB

```python
# scripts/embed_products.py
import json, chromadb
from sentence_transformers import SentenceTransformer
import clip, torch
from PIL import Image
import requests
from io import BytesIO

# ── Text embeddings ──────────────────────────────────────────────
text_model = SentenceTransformer("all-MiniLM-L6-v2")
client     = chromadb.PersistentClient(path="./chroma_db")
text_col   = client.get_or_create_collection("shopify_text")
image_col  = client.get_or_create_collection("shopify_images")

with open("data/products.json") as f:
    products = json.load(f)

texts = [p["text"] for p in products]
ids   = [p["id"]   for p in products]
meta  = [{"title": p["title"], "price": p["price"],
          "stock": p["stock"], "image_url": p["image_url"]} for p in products]

text_embeddings = text_model.encode(texts).tolist()
text_col.add(embeddings=text_embeddings, documents=texts, ids=ids, metadatas=meta)
print(f"Indexed {len(texts)} text embeddings.")

# ── Image embeddings ─────────────────────────────────────────────
clip_model, preprocess = clip.load("ViT-B/32", device="cpu")

for p in products:
    if not p["image_url"]:
        continue
    try:
        img_data = requests.get(p["image_url"], timeout=5).content
        img      = preprocess(Image.open(BytesIO(img_data))).unsqueeze(0)
        with torch.no_grad():
            vec = clip_model.encode_image(img).numpy().tolist()[0]
        image_col.add(embeddings=[vec], documents=[p["title"]],
                      ids=[p["id"] + "_img"], metadatas=[meta[products.index(p)]])
    except Exception as e:
        print(f"Skipping image for {p['title']}: {e}")

print("Image embeddings indexed.")
```

---

## 5. Multimodal Input System

> **Core Rule:** A single user request can combine any or all three input modes simultaneously. The backend fuses all signals before retrieval.

### 5.1 Text Input

Standard text input — handled natively by Chainlit's chat interface. No special setup needed. Text is passed directly to the embedding model.

### 5.2 Voice Input (Whisper STT — Free, Offline)

We use OpenAI's **open-source Whisper model** (the pip package — not the API) for speech recognition. It runs locally on CPU with no API cost or quota limit.

**Two voice input modes:**

- **Live microphone recording** — user clicks mic button in Chainlit UI, browser captures audio via Web Audio API, sends WAV blob to backend
- **Audio file upload** — user uploads an MP3, WAV, OGG, or M4A file; backend processes it identically

**Whisper model selection:**

| Model    | Size   | Speed (CPU) | Accuracy  | Recommended For            |
| -------- | ------ | ----------- | --------- | -------------------------- |
| `tiny`   | 39 MB  | Very fast   | Good      | Development / testing      |
| `base`   | 74 MB  | Fast        | Better    | **Default — best balance** |
| `small`  | 244 MB | Moderate    | Very good | Production quality         |
| `medium` | 769 MB | Slow on CPU | Excellent | Only if GPU available      |

> **Recommendation:** Use `base` during development. Switch to `small` before deployment for better accuracy with Indian-accented English.

```python
# backend/processors/voice.py
import whisper, tempfile, os

_model = None

def get_model():
    global _model
    if _model is None:
        _model = whisper.load_model("base")  # downloaded once, cached
    return _model

def transcribe(audio_bytes: bytes, suffix: str = ".wav") -> str:
    """Accepts raw audio bytes, returns transcribed text."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        result = get_model().transcribe(tmp_path)
        return result["text"].strip()
    finally:
        os.unlink(tmp_path)
```

### 5.3 Image Input (CLIP — Free, Offline)

We use OpenAI's **CLIP** model to convert product images into vectors searchable against product descriptions. This enables cross-modal search — finding text-described products using an image query.

**Two image input modes:**

- **Camera capture** — user clicks camera button, browser opens device camera, captures photo, sends JPEG to backend
- **Image file upload** — user uploads any JPEG, PNG, or WebP product photo

```python
# backend/processors/image.py
import clip, torch
from PIL import Image
from io import BytesIO

_model, _preprocess = None, None

def get_clip():
    global _model, _preprocess
    if _model is None:
        _model, _preprocess = clip.load("ViT-B/32", device="cpu")
    return _model, _preprocess

def encode_image(image_bytes: bytes) -> list[float]:
    """Accepts raw image bytes, returns CLIP embedding vector."""
    model, preprocess = get_clip()
    img = preprocess(Image.open(BytesIO(image_bytes))).unsqueeze(0)
    with torch.no_grad():
        vec = model.encode_image(img)
    return vec.numpy().tolist()[0]
```

### 5.4 Multimodal Fusion — All Input Combinations

| Scenario                 | Processing Logic                                             |
| ------------------------ | ------------------------------------------------------------ |
| Text only                | Embed text → ChromaDB text search                            |
| Voice only               | Whisper STT → text → embed → ChromaDB text search            |
| Image only               | CLIP encode → ChromaDB image search                          |
| Text + Voice             | Concatenate transcription + text → single text query         |
| Text + Image             | Text embed + image embed → both collections → merge results  |
| Voice + Image            | STT → text + CLIP image embed → fused search                 |
| **Text + Voice + Image** | **All three → fuse text signals → combine with image embed** |

---

## 6. RAG Pipeline Design

### 6.1 Two Vector Collections in ChromaDB

| Collection       | What's Stored                                | Embedding Model | Used For            |
| ---------------- | -------------------------------------------- | --------------- | ------------------- |
| `shopify_text`   | Product text docs (title, desc, tags, price) | `MiniLM-L6-v2`  | Text & voice search |
| `shopify_images` | Product image vectors                        | `CLIP ViT-B/32` | Image search        |

### 6.2 Retrieval Strategy

- **Text/voice queries:** cosine similarity search in `shopify_text`, top 5 results
- **Image queries:** cosine similarity search in `shopify_images`, top 5 results
- **Fused queries:** retrieve from both collections, merge + deduplicate, take top 5 unique
- **Inventory filter:** optionally exclude products where `stock = 0`

```python
# backend/rag/retriever.py
import chromadb
from sentence_transformers import SentenceTransformer

client     = chromadb.PersistentClient(path="./chroma_db")
text_col   = client.get_collection("shopify_text")
image_col  = client.get_collection("shopify_images")
text_model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_by_text(query: str, top_k: int = 5) -> list[dict]:
    vec     = text_model.encode([query]).tolist()
    results = text_col.query(query_embeddings=vec, n_results=top_k)
    return _format(results)

def retrieve_by_image(image_vec: list[float], top_k: int = 5) -> list[dict]:
    results = image_col.query(query_embeddings=[image_vec], n_results=top_k)
    return _format(results)

def retrieve_fused(query: str, image_vec: list[float], top_k: int = 5) -> list[dict]:
    text_results  = retrieve_by_text(query, top_k)
    image_results = retrieve_by_image(image_vec, top_k)
    seen, merged  = set(), []
    for item in text_results + image_results:
        if item["id"] not in seen:
            seen.add(item["id"])
            merged.append(item)
    return merged[:top_k]

def _format(results) -> list[dict]:
    out = []
    for i, doc in enumerate(results["documents"][0]):
        out.append({
            "id":       results["ids"][0][i],
            "text":     doc,
            "metadata": results["metadatas"][0][i],
        })
    return out
```

### 6.3 Prompt Design (Store-Scoped)

```python
# backend/rag/prompt.py
from langchain.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are a helpful shopping assistant for THIS store only.
Answer ONLY based on the product information provided in the context below.
If no matching products are found, say so honestly — never suggest products outside this store.
Always mention the product name, price, and availability in your response.
If multiple products match, compare them helpfully.

Store Products Context:
{context}
"""

CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("placeholder", "{chat_history}"),
    ("human",  "{question}"),
])
```

---

## 7. LangGraph Agentic Workflow

This is the core architectural differentiator. Instead of a simple linear chain, **LangGraph** models the entire multimodal processing pipeline as a **stateful directed graph** where each node is a discrete, testable processing step.

### 7.1 Why LangGraph for This Project

| Without LangGraph                            | With LangGraph                                        |
| -------------------------------------------- | ----------------------------------------------------- |
| Linear chain — hard to add conditional logic | Conditional edges — skip image node if no image sent  |
| Monolithic function — hard to debug          | Each node is isolated and traceable in LangSmith      |
| No built-in state management                 | `AgentState` TypedDict persists data across all nodes |
| Retry logic is manual                        | Built-in support for node-level retries               |
| Hard to extend (e.g., add a filter node)     | Add a new node and wire it in — no refactoring        |

### 7.2 Agent State Schema

```python
# backend/graph/state.py
from typing import TypedDict, Optional

class AgentState(TypedDict):
    # ── Raw inputs ───────────────────────────────────────────────
    text_input:       Optional[str]
    audio_bytes:      Optional[bytes]
    audio_suffix:     Optional[str]      # ".wav", ".mp3", etc.
    image_bytes:      Optional[bytes]
    session_id:       str

    # ── Processed signals ────────────────────────────────────────
    transcribed_text: Optional[str]      # Whisper output
    image_vector:     Optional[list]     # CLIP output
    fused_query:      Optional[str]      # Combined text query

    # ── RAG outputs ──────────────────────────────────────────────
    retrieved_docs:   list[dict]         # ChromaDB results
    context:          Optional[str]      # Formatted context string

    # ── Final output ─────────────────────────────────────────────
    response:         Optional[str]
    chat_history:     list
```

### 7.3 Graph Definition

```python
# backend/graph/agent.py
from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import (
    route_inputs,
    process_voice,
    process_image,
    fuse_inputs,
    retrieve_products,
    generate_response,
)

def build_agent() -> StateGraph:
    graph = StateGraph(AgentState)

    # ── Register nodes ───────────────────────────────────────────
    graph.add_node("router",    route_inputs)
    graph.add_node("voice",     process_voice)
    graph.add_node("image",     process_image)
    graph.add_node("fuser",     fuse_inputs)
    graph.add_node("retriever", retrieve_products)
    graph.add_node("generator", generate_response)

    # ── Entry point ──────────────────────────────────────────────
    graph.set_entry_point("router")

    # ── Conditional edges from router ────────────────────────────
    graph.add_conditional_edges(
        "router",
        decide_next_nodes,          # returns list of next nodes
        {
            "voice":  "voice",
            "image":  "image",
            "fuser":  "fuser",
        }
    )

    # ── Linear edges after processing ────────────────────────────
    graph.add_edge("voice",     "fuser")
    graph.add_edge("image",     "fuser")
    graph.add_edge("fuser",     "retriever")
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", END)

    return graph.compile()

def decide_next_nodes(state: AgentState) -> str:
    """Route to voice/image processors only if those inputs are present."""
    if state.get("audio_bytes"):
        return "voice"
    if state.get("image_bytes"):
        return "image"
    return "fuser"   # text only — skip straight to fusion
```

### 7.4 Node Implementations

```python
# backend/graph/nodes.py
from .state import AgentState
from backend.processors.voice  import transcribe
from backend.processors.image  import encode_image
from backend.rag.retriever     import retrieve_by_text, retrieve_by_image, retrieve_fused
from langchain_groq            import ChatGroq
from backend.rag.prompt        import CHAT_PROMPT

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, streaming=True)

# ── Node: route_inputs ────────────────────────────────────────────────────
def route_inputs(state: AgentState) -> AgentState:
    """Entry node — validates and logs what modalities are present."""
    print(f"[router] text={bool(state.get('text_input'))} "
          f"audio={bool(state.get('audio_bytes'))} "
          f"image={bool(state.get('image_bytes'))}")
    return state

# ── Node: process_voice ───────────────────────────────────────────────────
def process_voice(state: AgentState) -> AgentState:
    """Run Whisper STT on audio bytes."""
    if state.get("audio_bytes"):
        state["transcribed_text"] = transcribe(
            state["audio_bytes"],
            state.get("audio_suffix", ".wav")
        )
    return state

# ── Node: process_image ───────────────────────────────────────────────────
def process_image(state: AgentState) -> AgentState:
    """Run CLIP on image bytes to get embedding vector."""
    if state.get("image_bytes"):
        state["image_vector"] = encode_image(state["image_bytes"])
    return state

# ── Node: fuse_inputs ─────────────────────────────────────────────────────
def fuse_inputs(state: AgentState) -> AgentState:
    """Combine all text signals into a single query string."""
    parts = []
    if state.get("text_input"):        parts.append(state["text_input"])
    if state.get("transcribed_text"):  parts.append(state["transcribed_text"])
    state["fused_query"] = " ".join(parts).strip() or "show me products"
    return state

# ── Node: retrieve_products ───────────────────────────────────────────────
def retrieve_products(state: AgentState) -> AgentState:
    """Query ChromaDB using fused text and/or image vector."""
    query = state["fused_query"]
    vec   = state.get("image_vector")

    if query and vec:
        docs = retrieve_fused(query, vec)
    elif vec:
        docs = retrieve_by_image(vec)
    else:
        docs = retrieve_by_text(query)

    state["retrieved_docs"] = docs
    state["context"] = "\n\n".join(
        f"Product: {d['metadata']['title']}\n"
        f"Price: ₹{d['metadata']['price']}\n"
        f"Stock: {d['metadata']['stock']} units\n"
        f"Details: {d['text']}"
        for d in docs
    )
    return state

# ── Node: generate_response ───────────────────────────────────────────────
def generate_response(state: AgentState) -> AgentState:
    """Call LLM with retrieved context to generate final response."""
    chain    = CHAT_PROMPT | llm
    response = chain.invoke({
        "context":      state["context"],
        "question":     state["fused_query"],
        "chat_history": state.get("chat_history", []),
    })
    state["response"] = response.content
    return state
```

---

## 8. Backend API

### 8.1 API Endpoints

| Method | Endpoint           | Input                                                         | Description                     |
| ------ | ------------------ | ------------------------------------------------------------- | ------------------------------- |
| `POST` | `/api/search`      | FormData: `text?`, `audio_file?`, `image_file?`, `session_id` | Main multimodal search endpoint |
| `POST` | `/api/sync`        | None (Admin-only)                                             | Re-sync products from Shopify   |
| `GET`  | `/api/products`    | None                                                          | List all products in vector DB  |
| `GET`  | `/api/health`      | None                                                          | Health check for deployment     |
| `GET`  | `/api/collections` | None                                                          | ChromaDB collection stats       |

### 8.2 FastAPI App

```python
# backend/main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from backend.graph.agent import build_agent

app   = FastAPI(title="SmartShop: Shopify AI Chatbot API")
agent = build_agent()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

@app.post("/api/search")
async def search(
    text:       str        = Form(None),
    audio_file: UploadFile = File(None),
    image_file: UploadFile = File(None),
    session_id: str        = Form(...),
):
    state = {
        "text_input":  text,
        "audio_bytes": await audio_file.read() if audio_file else None,
        "audio_suffix": f".{audio_file.filename.split('.')[-1]}" if audio_file else ".wav",
        "image_bytes": await image_file.read() if image_file else None,
        "session_id":  session_id,
        "chat_history": [],
        "retrieved_docs": [],
    }

    result = agent.invoke(state)
    return {"response": result["response"], "products": result["retrieved_docs"]}


@app.get("/api/health")
def health():
    return {"status": "ok"}
```

---

## 9. Frontend Chatbot UI

### 9.1 Why Chainlit

Chainlit is a Python library that gives you a production-quality chat UI with zero frontend development. It natively supports file uploads, streaming responses, and custom elements.

### 9.2 UI Features

| Feature          | How                             | User Action                       |
| ---------------- | ------------------------------- | --------------------------------- |
| Text input       | Chainlit built-in               | Type and press Enter              |
| Voice — live mic | JavaScript Web Audio API widget | Click mic icon, speak, click stop |
| Voice — file     | Chainlit file upload            | Click attach, select audio file   |
| Image — camera   | JavaScript MediaDevices widget  | Click camera icon, take photo     |
| Image — file     | Chainlit file upload            | Click attach, select image file   |
| Product cards    | Chainlit custom elements        | Bot returns styled product cards  |
| Streaming reply  | LangChain streaming + Chainlit  | Response appears token by token   |

### 9.3 Chainlit App

```python
# frontend/app.py
import chainlit as cl
import httpx, uuid

BACKEND_URL = "http://localhost:8000/api/search"

@cl.on_chat_start
async def start():
    cl.user_session.set("session_id", str(uuid.uuid4()))
    await cl.Message(
        content="👋 Hi! I'm your store assistant. Ask me anything — or upload a photo/voice message to search for products!"
    ).send()

@cl.on_message
async def handle_message(message: cl.Message):
    session_id = cl.user_session.get("session_id")

    # Collect inputs
    text       = message.content or None
    audio_file = None
    image_file = None

    for el in (message.elements or []):
        if el.mime and el.mime.startswith("audio"):
            audio_file = (el.name, el.content, el.mime)
        elif el.mime and el.mime.startswith("image"):
            image_file = (el.name, el.content, el.mime)

    # Build multipart request
    data  = {"session_id": session_id}
    files = {}
    if text:        data["text"] = text
    if audio_file:  files["audio_file"] = audio_file
    if image_file:  files["image_file"] = image_file

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(BACKEND_URL, data=data, files=files)
        result = resp.json()

    await cl.Message(content=result["response"]).send()
```

---

## 10. Project Folder Structure

```
SmartShop/
├── .env                            # API keys (never commit)
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/
│   ├── kaggle_raw/                 # Downloaded Kaggle CSV files
│   ├── shopify_formatted.csv       # Reformatted for Shopify import
│   └── products.json               # Synced product data from Shopify API
│
├── scripts/
│   ├── format_kaggle.py            # Reformat Kaggle CSV → Shopify CSV
│   ├── sync_shopify.py             # Pull products from Shopify GraphQL API
│   └── embed_products.py           # Embed products → ChromaDB (text + images)
│
├── backend/
│   ├── main.py                     # FastAPI app — main entry point
│   ├── config.py                   # Settings (env vars, model names)
│   │
│   ├── graph/                      # ◄ LangGraph agentic layer
│   │   ├── agent.py                # Graph definition + compilation
│   │   ├── state.py                # AgentState TypedDict
│   │   └── nodes.py                # All node functions
│   │
│   ├── processors/
│   │   ├── voice.py                # Whisper STT logic
│   │   ├── image.py                # CLIP image encoding
│   │   └── text.py                 # Text embedding logic
│   │
│   ├── rag/
│   │   ├── retriever.py            # ChromaDB retrieval (text + image + fused)
│   │   ├── prompt.py               # LangChain prompt templates
│   │   └── chain.py                # LangChain LLM chain utilities
│   │
│   └── shopify/
│       └── client.py               # Shopify GraphQL API client
│
├── frontend/
│   ├── app.py                      # Chainlit chatbot app
│   ├── components/
│   │   ├── mic_recorder.js         # Browser mic recording widget
│   │   └── camera_capture.js       # Browser camera widget
│   └── public/                     # Static assets
│
├── chroma_db/                      # ChromaDB persistent storage (gitignored)
│
├── evaluation/
│   ├── test_queries.json           # Sample test queries per modality
│   └── ragas_eval.py               # RAGAS evaluation script
│
└── deployment/
    ├── Dockerfile
    └── README_deploy.md            # HuggingFace Spaces deploy guide
```

---

## 11. Development Phases & Milestones

| Phase       | Name                     | Key Tasks                                                                                                                                                                          | Est. Time |
| ----------- | ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| **Phase 1** | Foundation & Data        | Setup Shopify Partner account + dev store. Download & format Kaggle dataset. Import products via CSV. Setup Custom App + get API token. Write `sync_shopify.py`.                   | 2–3 days  |
| **Phase 2** | RAG Core (Text Only)     | Install ChromaDB + SentenceTransformers. Write `embed_products.py`. Build basic FastAPI backend. Integrate Groq API. Test text search end-to-end. Validate store-scoped responses. | 3–4 days  |
| **Phase 3** | LangGraph Agent Skeleton | Define `AgentState`. Build graph with router + fuser + retriever + generator nodes. Wire linear flow for text-only. Verify tracing in LangSmith.                                   | 2–3 days  |
| **Phase 4** | Voice Input              | Install Whisper. Build `voice.py` processor. Add voice node to LangGraph. Test with live mic recording and audio file upload. Tune with `base` model.                              | 2–3 days  |
| **Phase 5** | Image Input              | Install CLIP. Build `image.py` processor. Create `shopify_images` ChromaDB collection. Embed all product images. Add image node to LangGraph. Test image upload search.            | 3–4 days  |
| **Phase 6** | Multimodal Fusion        | Build conditional edges in LangGraph for all 7 input combinations. Add conversation memory via LangChain `ConversationBufferWindowMemory`. Test complex mixed queries.             | 2–3 days  |
| **Phase 7** | Chainlit Frontend        | Setup Chainlit app. Add mic recording JS widget. Add camera capture JS widget. Build product card custom elements. Connect to FastAPI backend.                                     | 3–4 days  |
| **Phase 8** | Evaluation & Polish      | Write RAGAS evaluation script. Run test suite across all modalities. Fix retrieval gaps. Add inventory filtering. Polish UI and error handling.                                    | 2–3 days  |
| **Phase 9** | Deployment               | Create Dockerfile. Deploy to HuggingFace Spaces. Test public URL. Record demo video. Update GitHub README with architecture diagram.                                               | 1–2 days  |

> ⏱ **Total Estimated Time:** 20–29 days of focused part-time development (2–4 hours/day)

---

## 12. Environment Setup Guide

### 12.1 Prerequisites

- Python 3.10+ installed
- Cursor IDE installed (`cursor.sh` — free tier)
- GitHub account
- Shopify Partner account (free — `partners.shopify.com`)
- Groq API key (free — `console.groq.com`)
- Google Gemini API key (free — `aistudio.google.com`)
- LangSmith API key (free — `smith.langchain.com`)

### 12.2 requirements.txt

```txt
# Core
fastapi==0.115.0
uvicorn[standard]==0.30.0
python-multipart==0.0.9
pydantic==2.7.0
python-dotenv==1.0.0
httpx==0.27.0

# LangChain + LangGraph
langchain==0.2.0
langchain-groq==0.1.0
langchain-google-genai==1.0.0
langgraph==0.1.0

# RAG & Embeddings
chromadb==0.5.0
sentence-transformers==3.0.0

# Multimodal
openai-whisper==20240930
openai-clip==1.0.1
torch==2.3.0
Pillow==10.3.0
ffmpeg-python==0.2.0

# Shopify
ShopifyAPI==12.7.0

# Frontend
chainlit==1.1.0

# Evaluation & Observability
ragas==0.1.0
langsmith==0.1.0
```

### 12.3 .env File Template

```env
# Shopify
SHOPIFY_STORE_URL=your-store.myshopify.com
SHOPIFY_ACCESS_TOKEN=shpat_xxxxx

# LLM Providers
GROQ_API_KEY=gsk_xxxxx
GOOGLE_API_KEY=AIzaxxxxx

# LangSmith (Observability)
LANGCHAIN_API_KEY=ls_xxxxx
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=SmartShop

# Settings
WHISPER_MODEL=base
TOP_K_RESULTS=5
CHROMA_PATH=./chroma_db
```

### 12.4 First-Run Commands

```bash
# 1. Clone your repo and create virtual environment
git clone https://github.com/soumitkundu/SmartShop
cd SmartShop
py -m venv .venv && .venv\Scripts\activate   # Mac/Linux: source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install ffmpeg (required by Whisper)
sudo apt install ffmpeg      # Linux / WSL
brew install ffmpeg          # macOS

# 4. Sync products from Shopify
python scripts/sync_shopify.py

# 5. Embed products into ChromaDB
python scripts/embed_products.py

# 6. Start backend (Terminal 1)
uvicorn backend.main:app --reload --port 8000

# 7. Start chatbot UI (Terminal 2)
chainlit run frontend/app.py
```

---

## 13. Evaluation & Testing

### 13.1 RAGAS Metrics

| Metric            | What It Measures                                       | Target Score |
| ----------------- | ------------------------------------------------------ | ------------ |
| Faithfulness      | Does the answer only use retrieved product info?       | > 0.85       |
| Answer Relevance  | Is the response relevant to the user's query?          | > 0.80       |
| Context Recall    | Are the right products retrieved?                      | > 0.75       |
| Context Precision | Are retrieved products actually useful for the answer? | > 0.75       |

### 13.2 LangSmith — LangGraph Tracing

Because the app uses LangGraph, every node execution is automatically traced in LangSmith when `LANGCHAIN_TRACING_V2=true`. This lets you:

- See exactly which nodes ran for a given query
- Measure latency per node (e.g., Whisper takes 800ms, retrieval takes 120ms)
- Inspect the state after each node for debugging
- Compare runs across modalities side by side

This is a significant portfolio differentiator — you can **show interviewers a LangSmith trace** of your agent handling a multimodal query.

### 13.3 Test Query Suite

| Modality     | Example Test Queries                                                                |
| ------------ | ----------------------------------------------------------------------------------- |
| Text only    | `"Show me wireless earphones under ₹2000"` / `"What gaming keyboards do you have?"` |
| Voice only   | Speak: `"I need running shoes for men size 10"` / Upload same as audio file         |
| Image only   | Upload photo of a blue denim jacket — expect visually similar products              |
| Text + Voice | Type `"budget option"` + speak `"for women's yoga pants"`                           |
| Text + Image | Type `"similar but cheaper"` + upload product photo                                 |
| All three    | Speak `"something like this in red"` + type `"under 1500"` + upload image           |

---

## 14. Deployment

### 14.1 Hugging Face Spaces (Recommended — Free)

1. Create an account at `huggingface.co`
2. New Space → SDK: **Docker** → Visibility: **Public**
3. Push your code with a `Dockerfile`
4. Add all `.env` keys as **HF Space Secrets**
5. Your app gets a public URL: `https://your-username.hf.space`

### 14.2 Dockerfile

```dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download Whisper model at build time (avoids cold-start delay)
RUN python -c "import whisper; whisper.load_model('base')"

EXPOSE 7860

CMD ["chainlit", "run", "frontend/app.py", "--host", "0.0.0.0", "--port", "7860"]
```

### 14.3 Alternative Deployment Options

| Platform            | Free Tier                                       | Best For                      |
| ------------------- | ----------------------------------------------- | ----------------------------- |
| Hugging Face Spaces | Free CPU instances, persistent storage          | **Final portfolio demo**      |
| Render.com          | Free web service (spins down after 15 min idle) | Backend API only              |
| Railway.app         | $5 free credit/month                            | Full-stack with persistent DB |
| Google Colab        | Free GPU runtime (session-limited)              | Development testing only      |

---

## 15. Resume & Portfolio Tips

### 15.1 Suggested Resume Bullet Points

> Copy-ready — optimised for ATS systems and technical interviews.

- Built a multimodal AI shopping assistant integrating **OpenAI Whisper** (STT), **CLIP** (image search), and **LLaMA 3.3 70B** via Groq API, orchestrated end-to-end using a **LangGraph** stateful agent graph
- Designed a **ChromaDB** vector store with dual text and image collections enabling cross-modal product retrieval using cosine similarity search, backed by a live **Shopify Admin GraphQL API** sync pipeline
- Implemented multimodal input fusion allowing simultaneous text, voice, and image queries in a single request via **FastAPI**, with all processing steps modelled as discrete **LangGraph** nodes for full observability
- Used **LangChain** prompt templates and `ConversationBufferWindowMemory` to enforce store-scoped RAG guardrails and support multi-turn product discovery conversations
- Deployed a publicly accessible chatbot demo on **Hugging Face Spaces** with **Chainlit** UI; evaluated RAG quality using the **RAGAS** framework with faithfulness scores above 0.85

### 15.2 Skills This Project Demonstrates

| Skill Category      | Specific Skills                                                                     |
| ------------------- | ----------------------------------------------------------------------------------- |
| AI / ML Engineering | RAG architecture, vector embeddings, multimodal AI, LLM prompting, RAGAS evaluation |
| Agentic AI          | LangGraph stateful graphs, conditional routing, node-based agent design             |
| Backend Development | FastAPI, async Python, REST API design, multipart form handling, Pydantic           |
| Data Engineering    | Shopify GraphQL API, ChromaDB, batch embedding pipelines, data sync scripts         |
| Audio / Vision AI   | OpenAI Whisper STT, CLIP image embeddings, cross-modal search                       |
| E-Commerce Domain   | Shopify Partner ecosystem, product catalog management, inventory data               |
| DevOps / Deployment | Docker, Hugging Face Spaces, environment management, LangSmith observability        |

### 15.3 Interview Talking Points

- _"I chose RAG over fine-tuning because the product catalog changes frequently — retraining would be impractical and expensive for a real store."_
- _"I used LangGraph instead of a simple LangChain chain because I needed conditional logic — for example, the image processing node should only run if the user actually sent an image."_
- _"The system prompt explicitly restricts the LLM to retrieved context only — this is how I guarantee the bot never hallucinates products from outside the store."_
- _"I used Whisper locally instead of the API to avoid per-call costs and to demonstrate that the speech recognition component works fully offline."_
- _"LangSmith gave me full visibility into each LangGraph node's latency — I could see Whisper was taking 800ms and optimised by pre-loading the model at startup."_

---

_End of Development Plan · AI-Powered Multimodal Product Discovery Chatbot_
_Portfolio Project · Zero-Cost Stack · LangGraph + LangChain + Shopify + Whisper + CLIP_
