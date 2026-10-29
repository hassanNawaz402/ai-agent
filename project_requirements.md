# Project Requirements — Local LLM RAG Assistant

**Version:** 1.0 (Development Round 1)
**Date:** 2026-10-09
**Source:** Refined and extended from `project_structure.txt`

---

## 1. Overview

A locally-deployed Qwen LLM (sized for an 8 GB GPU) that powers a Retrieval-Augmented Generation (RAG) assistant. The system exposes HTTP APIs for two independent use cases:

1. **Email responder** — ingests inbound email text, summarizes it, filters by allowed topics, and generates a reply.
2. **Website chatbot** — handles interactive chat messages from a web front-end.

The assistant must behave as a **strict, precise customer-support agent**, grounded in a configurable set of local documents and refusing any query outside the configured topic scope.

---

## 2. Scope — Round 1

In scope for this round:

- Local Qwen model deployment (dev + prod variants).
- Two HTTP routes: `/email` and `/chat`.
- Email summarization with chunking for long inputs (up to ~300,000 words).
- Topic allow-listing driven by `llm_configs.yml`.
- Document ingestion folder with automatic re-indexing on upload/modify.
- MySQL as the vector + metadata store (shared DB; LLM tables prefixed `llm_`).
- Single-entry bootstrap script `run_server.py` that provisions the conda env, installs dependencies, downloads the model, and starts the server.
- File-based logging (txt) separated by environment.

Explicitly **out of scope** for Round 1 (deferred): authentication/authorization, rate limiting, multi-tenant support, web UI, email inbox polling (the system only responds to strings POSTed to the API), model fine-tuning, and **agent-based task routing** (planned for Round 2 — see note below).

**Round 2 preview (do not build in Round 1):** Round 2 will introduce an agent layer that routes different kinds of user tasks to different data sources (e.g. structured DB, specific documents, external services). The user will supply the routing rules at the start of Round 2. Round 1 should keep the LLM client, retrieval function, and topic classifier at clean boundaries so the agent layer can be added on top later without rework.

---

## 3. Functional Requirements

### 3.1 API Surface

The server must expose at minimum the following HTTP endpoints:

| Method | Path            | Purpose                                           |
| ------ | --------------- | ------------------------------------------------- |
| POST   | `/email`        | Accept email body, return a generated reply.      |
| POST   | `/chat`         | Accept a chat message, return a reply.           |
| GET    | `/health`       | Liveness/readiness probe (model loaded, DB up).  |
| POST   | `/documents/reindex` | Manually trigger re-indexing of resources.  |

**Request/response contract (both `/email` and `/chat`):**

- Input: JSON `{ "text": "<string>", "session_id": "<optional string>" }`.
- Output: JSON `{ "reply": "<string>", "topics_matched": [...], "refused": <bool>, "reason": "<string|null>" }`.
- `session_id` is optional on `/chat` to preserve short conversational context; `/email` is stateless per request.

### 3.2 Email Route Behavior

1. Accept the raw email body (expected range: a few words to ~300,000 words).
2. If the body exceeds a configurable chunk size, split into chunks and run map-reduce summarization.
3. Produce a summary of **≤ 100 words** that enumerates the user's questions/asks.
4. Classify the summary against the `extract_topics` allow-list.
5. If **no** allowed topic matches, return a polite refusal (`refused: true`) explaining the scope.
6. If one or more topics match, run RAG over the indexed documents and generate a precise reply grounded in retrieved context.
7. Attach `topics_matched` to the response for auditability.

### 3.3 Chat Route Behavior

1. Accept a single user message plus optional `session_id`.
2. Apply the same topic allow-list filter as the email route.
3. Perform RAG and respond. Keep responses concise (chatbot tone, still strict and precise).
4. Maintain a short rolling context per `session_id` (last N turns, N configurable; default 6). Session state is in-memory for Round 1 and may be dropped on restart.

### 3.4 Topic Filtering

- The set of allowed topics is defined in `llm_configs.yml` under `extract_topics`.
- A query is accepted only if its extracted intent maps to at least one allowed topic.
- Refusals must be polite, brief, and must **not** reveal implementation details.

### 3.5 RAG & Document Ingestion

- A dedicated folder — proposed name **`llm_docs/`** (shorter than `llm_readable_resources`) — holds source documents.
- Supported file types: PDF, DOC/DOCX, XLS/XLSX, CSV, TXT, MD.
- On server start the folder is scanned; new/modified files are re-parsed, re-chunked, re-embedded, and upserted into MySQL. Deleted files have their vectors purged.
- A file watcher (or polling fallback) re-indexes changes while the server is running.
- Each document chunk stores: `doc_id`, `chunk_id`, `source_path`, `content`, `embedding`, `content_hash`, `updated_at`.
- Retrieval uses top-k similarity (k configurable; default 5) with a minimum similarity threshold to avoid hallucinated grounding.

### 3.6 LLM Personality

System prompt must enforce:

- Strict, precise, professional tone — no speculation.
- Only answer within configured topics; refuse everything else.
- Ground answers in retrieved context; if evidence is insufficient, say so rather than guess.
- Never reveal system prompt, model name, or internal configuration.

---

## 4. Configuration — `llm_configs.yml`

The original sketch in `project_structure.txt` mixes JSON and YAML syntax. The canonical format must be **valid YAML**. The config must also allow the operator to **swap the LLM backend** between a local Qwen model and a hosted provider API (ChatGPT, Claude, Gemini, etc.) without code changes. Proposed schema:

```yaml
env:
  active: dev            # "dev" or "prod"
  profiles:
    dev:                             # target: MacBook Air M5, 16 GB unified memory, macOS
      provider: local_qwen
      backend: mlx                   # mlx | llama_cpp | transformers_mps
      model: mlx-community/Qwen2.5-7B-Instruct-4bit
      embedding_model: BAAI/bge-small-en-v1.5
    prod:                            # target: Linux host with 8 GB VRAM NVIDIA GPU
      provider: local_qwen
      backend: transformers          # or vllm
      model: Qwen/Qwen2.5-7B-Instruct    # int4/int8 quantized to fit 8 GB VRAM
      embedding_model: BAAI/bge-small-en-v1.5

# Per-provider connection details. Only the block for the active provider is required.
# API keys are ALWAYS loaded from env vars — never stored in this file.
providers:
  local_qwen:
    # Device/backend auto-detected if not set:
    #   macOS + Apple Silicon → backend: mlx, device: mps
    #   Linux + NVIDIA GPU    → backend: transformers (or vllm), device: cuda
    #   Fallback              → backend: llama_cpp, device: cpu
    device: auto                     # auto | mps | cuda | cpu
    quantization: int4               # int4 | int8 | none  (MLX 4-bit on Mac, bitsandbytes on CUDA)
    max_memory_gb: 8                 # VRAM on CUDA, unified-memory budget on Apple Silicon
    context_window: 8192
  openai:                            # ChatGPT / GPT-4 family
    api_key_env: OPENAI_API_KEY
    base_url: https://api.openai.com/v1
    organization_env: OPENAI_ORG     # optional
  anthropic:                         # Claude
    api_key_env: ANTHROPIC_API_KEY
    base_url: https://api.anthropic.com
  google:                            # Gemini
    api_key_env: GOOGLE_API_KEY
  mistral:
    api_key_env: MISTRAL_API_KEY
  cohere:
    api_key_env: COHERE_API_KEY
  azure_openai:
    api_key_env: AZURE_OPENAI_API_KEY
    endpoint_env: AZURE_OPENAI_ENDPOINT
    api_version: "2024-10-21"
    deployment: <deployment-name>
  aws_bedrock:
    region_env: AWS_REGION
    # credentials via standard AWS env vars / IAM role
  groq:
    api_key_env: GROQ_API_KEY
  together:
    api_key_env: TOGETHER_API_KEY
  xai:                               # Grok
    api_key_env: XAI_API_KEY

# Optional overrides shared across providers
generation:
  temperature: 0.2                   # strict/precise persona → low temp
  max_output_tokens: 1024
  timeout_seconds: 60
  retries: 2

extract_topics:
  - company information
  - booking request
  - payment information

rag:
  chunk_size_tokens: 512
  chunk_overlap_tokens: 64
  top_k: 5
  min_similarity: 0.25

email:
  max_summary_words: 100
  chunk_size_words: 4000

chat:
  session_turns: 6

server:
  host: 0.0.0.0
  port: 8000

database:
  host: localhost
  port: 3306
  name: <db_name>
  user: <user>
  # password sourced from env var, not committed
```

Secrets (DB password, HF token, provider API keys) must be read from environment variables, **not** stored in the YAML. The loader must fail fast with a clear error if the active provider's required env var is missing.

### 4.1 Supported LLM Providers

The `provider` field accepts one of the following values. Adding a new provider must only require implementing a thin adapter (see §4.2), not changing routes or RAG code.

| Value          | Backend                                         | Notes                                                     |
| -------------- | ----------------------------------------------- | --------------------------------------------------------- |
| `local_qwen`   | Local Qwen model — backend selected per host    | Default. On Mac (dev) use **MLX** (Apple Silicon-native); on Linux+NVIDIA (prod) use `transformers`/vLLM with int4/int8 quant to fit 8 GB VRAM. No external API calls. |
| `openai`       | OpenAI ChatGPT (GPT-4, GPT-4o, GPT-4o-mini, …) | Via official `openai` SDK.                                |
| `anthropic`    | Anthropic Claude (Opus, Sonnet, Haiku)         | Via official `anthropic` SDK.                             |
| `google`       | Google Gemini                                   | Via `google-generativeai` SDK.                            |
| `mistral`      | Mistral AI (Large, Medium, Small)              | Hosted.                                                   |
| `cohere`       | Cohere Command family                          | Hosted.                                                   |
| `azure_openai` | OpenAI models via Azure                        | Enterprise/compliance deployments.                        |
| `aws_bedrock`  | Anthropic / Meta / Mistral / Amazon via Bedrock | Enterprise/compliance deployments.                        |
| `groq`         | Groq-hosted open models (Llama, Mixtral)       | Low-latency inference.                                    |
| `together`     | Together AI-hosted open models                 | Broad open-model catalog.                                 |
| `xai`          | xAI Grok                                        | Hosted.                                                   |

### 4.2 Provider Abstraction

- Define a single internal interface — `LLMClient` with methods `generate(messages, **gen_kwargs)` and `embed(texts)` — implemented once per provider.
- The route handlers, summarizer, and RAG retrieval **never** import a specific SDK; they only talk to `LLMClient`.
- A factory reads `profiles.<active>.provider` and returns the correct implementation.
- `embedding_model` may belong to a different provider than the chat model (e.g. local embeddings + hosted chat); the config must allow this split.
- Behaviour under provider failure: retry per `generation.retries`, then surface a 502 with a sanitized error — never leak API keys or stack traces to clients.

### 4.3 Bootstrap Behaviour by Provider

`run_server.py` branches on the active provider:

- **`local_qwen`** → download/verify the model weights locally (existing behavior).
- **Hosted providers** → skip the model download step entirely; instead, verify that the required env vars are set and perform a lightweight connectivity check (e.g. a minimal `models.list` or 1-token completion) before marking the server ready.
- The conda env still carries the `dev_` / `prod_` prefix and includes the SDKs for the active provider (and optionally all providers in dev for easy switching).

---

## 5. Database — MySQL

- Shared database; all LLM-owned tables prefixed `llm_`.
- Minimum tables for Round 1:
  - `llm_documents` — one row per source file (path, hash, size, mtime, status).
  - `llm_chunks` — chunked text + metadata, FK to `llm_documents`.
  - `llm_embeddings` — vector column (BLOB/JSON or `VECTOR` type if MySQL 9+), FK to `llm_chunks`.
  - `llm_request_log` — audit of inbound requests (route, topic match, refused flag, latency, timestamp). No raw email bodies stored unless explicitly enabled.
- Logs do **not** go into the database — they go to text files (see §7).
- Migrations: provide a simple SQL bootstrap script run on first server start if tables are missing.

---

## 6. Deployment & Bootstrap

### 6.1 `run_server.py` responsibilities (in order)

1. Load `llm_configs.yml`; validate schema; abort with clear error on misconfiguration.
2. Determine active env (`dev` or `prod`).
3. Ensure the conda environment named **`dev_advantage_removals`** or **`prod_advantage_removals`** exists; create it if missing. On macOS, use the `osx-arm64` conda channel (miniforge recommended) — the dev env must be Apple Silicon native, not Rosetta/x86_64.
4. Install/sync dependencies into that env from the platform-appropriate requirements file (`requirements-mac.txt` on macOS, `requirements-linux.txt` on Linux). Fail fast if a CUDA wheel is requested on macOS or vice versa.
5. If the active provider is `local_qwen`, download the configured model if not already cached locally. For hosted providers, skip the download and instead verify that the required API-key env vars are set and run a lightweight connectivity check against the provider.
6. Run DB migrations if needed; verify MySQL connectivity.
7. Index/refresh the `llm_docs/` folder.
8. Start the HTTP server. In **prod** mode, use a production WSGI/ASGI server (e.g. `gunicorn` with `uvicorn` workers for FastAPI) with sensible defaults (workers = CPU count, timeouts, access logs enabled).

### 6.2 `requirements.txt`

Must at minimum include:

- **Core:** FastAPI (or Flask), uvicorn, gunicorn (prod), pyyaml, sqlalchemy, pymysql, pypdf, python-docx, openpyxl, pandas, watchdog, python-dotenv.
- **Local-model path — Mac dev (Apple Silicon):** `mlx`, `mlx-lm` (preferred, Apple-native, uses the M-series Neural Engine / unified memory efficiently). Alternative: `llama-cpp-python` built with Metal. `torch` is installed as the macOS wheel (uses MPS) only if the embedding model needs it; CUDA wheels must not be installed on macOS.
- **Local-model path — Linux prod (NVIDIA):** `transformers`, `torch` (CUDA build matching the GPU), `accelerate`, `bitsandbytes` (for int4/int8). Optionally `vllm` for faster inference.
- **Embeddings:** `sentence-transformers` works on both Mac (MPS/CPU) and CUDA.
- **Document parsers:** `pypdf`, `python-docx`, `openpyxl`, `pandas`.
- **Hosted-provider SDKs** (install those matching enabled providers): `openai`, `anthropic`, `google-generativeai`, `mistralai`, `cohere`, `boto3` (for Bedrock), `groq`, `together`.

Keep two requirements files — `requirements-mac.txt` (dev) and `requirements-linux.txt` (prod) — because the local-model stack diverges between platforms (MLX vs. CUDA transformers). `run_server.py` picks the right one based on the detected platform and the active profile.

---

## 7. Logging

- Written to text files only, never to the DB.
- Separate directories/files per environment: `logs/dev/<date>.log`, `logs/prod/<date>.log`.
- Rotating file handler (size- or time-based).
- Levels: dev → `DEBUG`; prod → `INFO` and above (plus a separate `error.log`).
- Each log line includes: timestamp, level, request ID (where applicable), route, message.

---

## 8. Non-Functional Requirements

- **Hardware targets:**
  - **Dev:** MacBook Air, Apple M5, 16 GB unified memory, macOS Tahoe 26.6. Model runs via MLX with 4-bit quantization. Qwen2.5-7B-Instruct-4bit uses ~4–5 GB of unified memory, leaving ample headroom for the OS, Python stack, embeddings, and MySQL.
  - **Prod:** Linux host with ≥ 8 GB VRAM NVIDIA GPU. Chosen Qwen variant plus quantization must fit comfortably at batch size 1.
- **Latency targets (soft, Round 1):** `/chat` < 5 s typical response; `/email` < 30 s for emails ≤ 10k words.
- **Reliability:** server must not crash on malformed input or oversized payloads — return 4xx with a clear error.
- **Observability:** `/health` reflects model-loaded, DB-reachable, and index-ready states.
- **Reproducibility:** pinned dependency versions in `requirements.txt`; model revision pinned in config.

---

## 9. Proposed Folder Structure

```
hassan_project/
├── run_server.py
├── requirements.txt
├── llm_configs.yml
├── project_structure.txt          # original, do not modify
├── project_requirements.md        # this file
├── llm_docs/                      # source documents for RAG
├── logs/
│   ├── dev/
│   └── prod/
├── app/
│   ├── __init__.py
│   ├── api/                       # FastAPI routers (email, chat, health, documents)
│   ├── rag/                       # chunking, embedding, retrieval
│   ├── llm/                       # model loader, prompt templates, personality
│   ├── summarize/                 # email summarizer (map-reduce)
│   ├── db/                        # SQLAlchemy models, migrations
│   ├── config/                    # YAML loader & validation
│   └── utils/                     # logging, file watcher, hashing
└── scripts/
    └── init_db.sql
```

---

## 10. Open Questions / Assumptions

Items the original brief leaves ambiguous — flagged for confirmation before implementation:

1. **Qwen variants** — proposed defaults, confirm before implementation:
   - Dev (Apple M5, 16 GB unified): `mlx-community/Qwen2.5-7B-Instruct-4bit` via MLX. Fallback for lower-spec Macs: `mlx-community/Qwen2.5-3B-Instruct-4bit`.
   - Prod (8 GB VRAM NVIDIA): `Qwen/Qwen2.5-7B-Instruct` with int4 (bitsandbytes/AWQ) quantization, or `Qwen/Qwen2.5-3B-Instruct` unquantized if latency matters more than quality.
2. **Default hosted-provider models** per provider (e.g. `gpt-4o-mini` for OpenAI, `claude-haiku-4-5` for Anthropic, `gemini-1.5-flash` for Google) — confirm the preferred defaults.
3. **Data-residency / privacy constraints** when hosted providers are enabled — some customers may require Azure OpenAI or Bedrock over the public APIs. Confirm which providers are actually permitted in prod.
4. **Embedding model** — not specified in the brief; recommend `BAAI/bge-small-en-v1.5` or `intfloat/e5-small-v2` for CPU/GPU friendliness. Hosted alternatives (`text-embedding-3-small`, Cohere `embed-v3`) are supported via the same provider abstraction.
5. **Vector storage in MySQL** — Round 1 may use JSON/BLOB columns with Python-side cosine similarity. If MySQL 9+ `VECTOR` type is available, prefer it.
6. **Email language(s)** — assume English for Round 1 unless told otherwise.
7. **Document ACLs** — Round 1 assumes all documents in `llm_docs/` are equally accessible to all queries.
8. **Session persistence** on `/chat` — in-memory for Round 1; move to Redis/MySQL later if required.
9. **Rate limiting & auth** — deferred; the API is assumed to sit behind a trusted gateway for Round 1.
10. **Cost controls for hosted providers** — do we need per-day token/spend caps and alerting when a hosted provider is active? Deferred unless requested.

---

## 11. Acceptance Criteria — Round 1

The round is complete when:

- `python run_server.py` on a clean machine creates the conda env, installs deps, downloads the configured Qwen model (or verifies hosted-provider connectivity), migrates the DB, indexes `llm_docs/`, and starts the server — with no manual steps.
- Switching `provider` in `llm_configs.yml` from `local_qwen` to `openai` (or `anthropic`, etc.) and restarting the server routes all LLM calls through the hosted API without any code change. Missing API-key env vars cause a clear startup error rather than a runtime crash.
- `POST /email` with a short email returns a grounded reply; with a 300k-word email it returns a reply after summarization without OOM.
- `POST /chat` returns a grounded reply and respects session context across turns.
- Both routes refuse off-topic queries politely and set `refused: true`.
- Adding a new PDF to `llm_docs/` makes its content retrievable without a restart.
- Dev and prod envs are distinguishable by conda env name, model variant, log directory, and server settings.
- All LLM tables in MySQL carry the `llm_` prefix; no logs are written to the DB.
