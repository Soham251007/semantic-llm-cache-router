# Semantic LLM Cache & Tiered Complexity Router

[![Go Version](https://img.shields.io/badge/Go-1.22+-00ADD8?style=flat&logo=go)](https://golang.org)
[![Python Version](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python)](https://python.org)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-INT8_Quantized-005CED?style=flat&logo=onnx)](https://onnxruntime.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-Microservice-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Redis Stack](https://img.shields.io/badge/Redis_Stack-Vector_Search-DC382D?style=flat&logo=redis)](https://redis.io)
[![Prometheus](https://img.shields.io/badge/Prometheus-Metrics-E6522C?style=flat&logo=prometheus)](https://prometheus.io)
[![OpenAI Compatible](https://img.shields.io/badge/API-OpenAI_Compatible-412991?style=flat&logo=openai)](https://platform.openai.com/docs/api-reference)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Building an AI wrapper is easy; building the cost and latency control plane in front of it is what enterprise engineering teams actually pay for.**

A high-performance, drop-in reverse proxy and control plane built in **Go** and backed by a **Python ML microservice** running an **INT8-quantized `all-MiniLM-L6-v2` embedding engine** on ONNX Runtime and a **calibrated DistilBERT prompt complexity router**.

Acts as an **OpenAI-compatible drop-in replacement (`POST /v1/chat/completions`)** that automatically intercepts prompts, executes semantic vector similarity checks, and routes difficult reasoning tasks to frontier models while deflecting simple extraction/formatting queries to high-speed tier-1 models.

---

## Architecture

```mermaid
flowchart TD
    Client["Client Application\n(OpenAI SDK, LangChain, LlamaIndex, cURL)"]
    
    subgraph ControlPlane ["Go Control Plane & Reverse Proxy (:8080)"]
        Proxy["HTTP Reverse Proxy\nPOST /v1/chat/completions"]
        L1["L1 Exact Cache\n(O(1) SHA-256 Hash Index)"]
        L2["L2 Semantic Vector Cache\n(In-Memory Cosine / Redis Stack HNSW)"]
        Streamer["SSE Token Streamer\n(Chunk Replayer)"]
        Metrics["Prometheus Exporter\n(/metrics)"]
    end
    
    subgraph MLService ["Python ML Microservice (:8001)"]
        ONNX["ONNX Runtime Engine\n(all-MiniLM-L6-v2 INT8 — 384 dim)"]
        DistilBERT["Complexity Classifier\n(DistilBERT Sequence Classifier + Signals)"]
    end
    
    subgraph UpstreamTiers ["Upstream Model Tiers"]
        Tier1["Tier 1: Fast Model\n(Llama-3-8B / Gemini Flash / GPT-4o-mini)\nCost: ~$0.15 / 1M tokens"]
        Tier2["Tier 2: Frontier Model\n(GPT-4o / Claude 3.5 Sonnet / Gemini Pro)\nCost: ~$2.50 / 1M tokens"]
    end

    Client -->|1. Prompt Request| Proxy
    Proxy -->|2. Check Exact Hash| L1
    L1 -->|Hit <1ms| Streamer -->|Cached Response| Client
    
    L1 -->|Miss| ONNX
    ONNX -->|384-dim Vector + Complexity Score| Proxy
    
    Proxy -->|3. Check Vector Cosine >= 0.94| L2
    L2 -->|Hit <15ms| Streamer -->|Cached Response| Client
    
    L2 -->|Miss| DistilBERT
    DistilBERT -->|Score < 0.45 (Simple)| Tier1
    DistilBERT -->|Score >= 0.45 (Complex)| Tier2
    
    Tier1 -->|Generate| Proxy
    Tier2 -->|Generate| Proxy
    Proxy -.->|Async Cache Ingestion| L1
    Proxy -.->|Async Cache Ingestion| L2
    Proxy -->|Stream or JSON Response| Client
```

---

## Features

- **Asymmetrical Dual-Tier Caching (L1 + L2)**:
  - **L1 Exact Match Hash**: Sub-millisecond ($<1\text{ms}$) lock-free SHA-256 canonical hash index. Zero ML compute tax on repeated queries.
  - **L2 Semantic Vector Cache**: In-memory 384-dimensional cosine similarity index with 4-way loop-unrolled SIMD dot products ($<15\text{ms}$ responses).
- **Strict Similarity Calibration ($\tau = 0.94$)**: Prevents dangerous false-positive semantic cache hits while safely capturing phrasing variations.
- **Calibrated DistilBERT Complexity Routing**: Evaluates prompt difficulty ($C \in [0, 1]$). Automatically routes simple tasks (formatting, extraction, summarization, translation) to low-cost tier-1 models and reserves expensive frontier models for deep reasoning and code synthesis.
- **Single-Hop ML Analysis**: Go queries the ML microservice's `/analyze` endpoint in a single roundtrip over keep-alive HTTP, retrieving both the vector embedding and the task difficulty score simultaneously.
- **OpenAI-Compatible Drop-In**: Full support for `stream: false` (JSON) and `stream: true` (Server-Sent Events `text/event-stream`), including cached token stream replaying.
- **Dual Vector Engine Modes**:
  - **Standalone In-Memory**: Concurrency-safe, zero external infrastructure required.
  - **Distributed Redis Stack**: High-scale HNSW vector indexing (`FT.SEARCH` with KNN cosine similarity).
- **Enterprise Observability**: Real-time response telemetry headers (`X-Cache-Status`, `X-Cache-Similarity`, `X-Router-Tier`, `X-Latency-Ms`, `X-Tokens-Saved`, `X-Cost-Saved-USD`) plus built-in Prometheus metrics (`/metrics`).

---

## Quickstart

### Prerequisites
- **Go 1.22+**
- **Python 3.10+**
- *(Optional)* **Docker & Docker Compose**

### 1. Clone & Build
```bash
git clone https://github.com/<your-username>/semantic-llm-cache-router.git
cd semantic-llm-cache-router

# Build the Go gateway
go mod download
go build -o bin/gateway ./cmd/gateway
```

### 2. Start Services Locally

**On Windows (PowerShell):**
```powershell
.\scripts\start_services.ps1
```

**On Linux / macOS:**
```bash
./scripts/start_services.sh
```

The services will initialize:
- **Go Control Plane:** `http://localhost:8080/v1/chat/completions`
- **ML Engine:** `http://localhost:8001/health`
- **Prometheus Metrics:** `http://localhost:8080/metrics`
- **Cache Stats:** `http://localhost:8080/api/cache/stats`

---

## Drop-In Integration Guide

Because the control plane implements the standard OpenAI specification, integrating with existing projects requires changing only the `base_url`:

### Python (Official OpenAI SDK)
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="your-key-or-dummy",
)

response = client.chat.completions.create(
    model="router-auto",
    messages=[
        {"role": "user", "content": "How do I reset my company account password?"}
    ],
    stream=True,  # Full SSE chunk streaming supported
)

for chunk in response:
  if chunk.choices[0].delta.content:
    print(chunk.choices[0].delta.content, end="", flush=True)
```

### TypeScript / Node.js
```typescript
import OpenAI from "openai";

const openai = new OpenAI({
  baseURL: "http://localhost:8080/v1",
  apiKey: "your-key-or-dummy",
});

const completion = await openai.chat.completions.create({
  model: "router-auto",
  messages: [{ role: "user", content: "Summarize this quarterly earnings call in 3 bullets." }],
});

console.log(completion.choices[0].message.content);
```

### cURL
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "router-auto",
    "messages": [
      {"role": "user", "content": "How do I reset my account password?"}
    ],
    "stream": false
  }'
```

---

## Telemetry Response Headers

Every response processed through the gateway injects real-time telemetry headers:

```http
HTTP/1.1 200 OK
Content-Type: application/json
X-Cache-Status: HIT-SEMANTIC
X-Cache-Similarity: 0.9622
X-Router-Complexity: 0.0105
X-Router-Tier: tier-1-fast
X-Latency-Ms: 14.80
X-Tokens-Saved: 420
X-Cost-Saved-USD: 0.001050
```

| Header | Possible Values | Description |
| :--- | :--- | :--- |
| `X-Cache-Status` | `HIT-EXACT`, `HIT-SEMANTIC`, `MISS` | Cache resolution status |
| `X-Cache-Similarity` | `-1.0000` to `1.0000` | Cosine similarity score against nearest cached vector |
| `X-Router-Complexity` | `0.0000` to `1.0000` | DistilBERT task difficulty score |
| `X-Router-Tier` | `tier-1-fast`, `tier-2-frontier` | Target model tier chosen by the router |
| `X-Latency-Ms` | Numeric (e.g. `14.80`) | End-to-end processing latency in milliseconds |
| `X-Tokens-Saved` | Integer (e.g. `420`) | Estimated tokens saved if served from cache |
| `X-Cost-Saved-USD` | Numeric (e.g. `0.001050`) | Estimated USD cost saved by caching or tier routing |

---

## Configuration Reference

Configure the proxy using standard environment variables:

| Environment Variable | Default | Description |
| :--- | :--- | :--- |
| `PORT` | `8080` | Port for the Go reverse proxy gateway |
| `ML_SERVICE_URL` | `http://localhost:8001` | URL of the Python ONNX / DistilBERT microservice |
| `CACHE_BACKEND` | `memory` | Cache backend: `memory` or `redis` |
| `REDIS_URL` | `redis://localhost:6379` | Connection string for Redis Stack |
| `SIMILARITY_THRESHOLD` | `0.94` | Cosine similarity cutoff for semantic cache hits |
| `COMPLEXITY_ALPHA` | `0.45` | Difficulty score decision threshold for Tier-1 vs Tier-2 |
| `UPSTREAM_PROVIDER` | `simulator` | Upstream provider: `simulator`, `openai`, or `gemini` |
| `OPENAI_API_KEY` | `""` | API key if forwarding to OpenAI |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Base URL for OpenAI-compatible upstreams (e.g., Ollama, vLLM) |
| `GEMINI_API_KEY` | `""` | API key if forwarding to Google Gemini |
| `TIER1_MODEL` | `gpt-4o-mini` | Fast model name for simple queries |
| `TIER2_MODEL` | `gpt-4o` | Frontier model name for complex reasoning queries |
| `CACHE_TTL_SECONDS` | `86400` | TTL expiration for cached entries (24 hours) |
| `ENABLE_METRICS` | `true` | Expose Prometheus metrics on `/metrics` |

---

## Production Deployment with Docker Compose

Deploy the complete multi-container stack with Redis Stack and Prometheus with one command:

```bash
cd deployments
docker compose up --build -d
```

### Services Deployed:
- **Go Gateway Control Plane**: `http://localhost:8080`
- **Python ML Inference Microservice**: `http://localhost:8001`
- **Redis Stack (Vector Index)**: `localhost:6379`
- **Prometheus Dashboard**: `http://localhost:9090`

---

## Automated Test Suite

Both Go and Python components include unit and integration tests:

```bash
# 1. Run Go Unit and Integration Tests
go test -v ./...

# 2. Run Python ML Vector and Classifier Tests
python services/ml_service/test_ml_service.py
```

---

## License

MIT License. Designed and engineered for high-concurrency production LLM deployments.
