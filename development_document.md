# Development Document — Local LLM RAG Assistant

**Purpose:** This document is the build plan Claude follows while implementing Round 1. It locks in folder structure, phase order, and a concrete to-do list so work does not derail. Keep it updated as decisions are made.

**Companion docs:**
- `project_structure.txt` — original brief (do not modify).
- `project_requirements.md` — refined requirements, canonical source of truth for *what* to build.
- This file — *how* and *in what order* to build it.

---

## 0. Working Principles

Read these before every work session.

1. **Build the minimum that satisfies the requirement.** No speculative abstractions, no "we might need this later." Three similar lines beats a premature base class.
2. **Ask before assuming.** If a requirement is ambiguous, stop and ask the user — do not pick a direction silently. Pending questions live in §2.
3. **One phase at a time.** Finish and demo a phase before starting the next. Do not interleave phases.
4. **Prefer boring tech.** FastAPI, SQLAlchemy, pyyaml, MLX. No frameworks introduced "for flexibility."
5. **Dev target is the Mac.** Every change must run on the MacBook Air M5 (Apple Silicon, 16 GB unified memory, macOS Tahoe 26.6). Prod (Linux + NVIDIA) is a secondary target for Round 1 — scaffold it but do not block on it.
6. **No code comments unless the "why" is non-obvious.** Names carry meaning.
7. **Update the to-do list in §6 as you go.** Check items off; add newly discovered sub-tasks under the right phase; do not reorder phases without agreement.

---

## 2. Pre-flight Questions (BLOCK on these before Phase 1)

Answer these before writing any non-scaffolding code. Fill in the "Decision" column as answers arrive.

| # | Question                                                                                         | Default guess                      | Decision |
| - | ------------------------------------------------------------------------------------------------ | ---------------------------------- | -------- |
| 1 | Web framework: FastAPI or Flask?                                                                 | FastAPI                            |          |
| 2 | Conda env name prefix: `dev_<name>` — what is `<name>`?                                          | `hassan`                           | `advantage_removals` → envs: `dev_advantage_removals`, `prod_advantage_removals` |
| 3 | Is MySQL already installed and running on the dev Mac?                                           | Assume yes                         |          |
| 4 | MySQL version — 9+ (`VECTOR` type) or 8.x (JSON/BLOB + Python cosine)?                           | 8.x → JSON column                  |          |
| 5 | Should `run_server.py` create the database, or assume it exists?                                 | Create DB if missing, then tables  |          |
| 6 | File watcher for `llm_docs/` — live watch or startup-scan + manual reindex endpoint?             | Startup + manual endpoint          |          |
| 7 | Chat session storage — in-memory for Round 1?                                                    | In-memory, drops on restart        |          |
| 8 | CORS / API-key auth — configure now or defer?                                                    | Defer both                         |          |
| 9 | Sample documents in `llm_docs/` — user provides, or Claude ships a couple of test files?        | User provides                      |          |

---

## 3. Target Architecture (one-paragraph summary)

A FastAPI server exposes `/email`, `/chat`, `/health`, and `/documents/reindex`. A single `LLMClient` interface abstracts over local Qwen (MLX on Mac dev, transformers+CUDA on Linux prod) and hosted providers (OpenAI/Anthropic/etc.). Documents in `llm_docs/` are parsed, chunked, embedded, and stored in MySQL with the `llm_` table prefix. On each request the server extracts the topic, refuses if off-list, otherwise retrieves top-k chunks and asks the LLM to answer with a strict, precise persona. The email route adds a map-reduce summarization step for long inputs.

---

## 4. Folder Structure (create exactly this; no extras unless a phase requires it)

```
hassan_project/
├── run_server.py                       # single entry point (bootstrap + launch)
├── requirements-mac.txt                # dev deps (MLX, no CUDA)
├── requirements-linux.txt              # prod deps (torch CUDA, bitsandbytes)
├── llm_configs.yml                     # user-editable config
├── .env.example                        # names of required env vars (no secrets)
├── project_structure.txt               # original brief — DO NOT MODIFY
├── project_requirements.md             # refined requirements
├── development_document.md             # this file
├── llm_docs/                           # source documents (RAG inputs)
│   └── .gitkeep
├── logs/
│   ├── dev/                            # dev run logs
│   └── prod/                           # prod run logs
├── scripts/
│   └── init_db.sql                     # idempotent table creation
└── app/
    ├── __init__.py
    ├── main.py                         # FastAPI app factory, route registration
    ├── config.py                       # YAML loader + validation (pydantic)
    ├── logging_setup.py                # file handlers per env
    ├── api/
    │   ├── __init__.py
    │   ├── health.py
    │   ├── chat.py
    │   ├── email.py
    │   └── documents.py
    ├── llm/
    │   ├── __init__.py
    │   ├── client.py                   # LLMClient Protocol/ABC
    │   ├── factory.py                  # provider → implementation
    │   ├── backends/
    │   │   ├── __init__.py
    │   │   ├── mlx_qwen.py             # dev backend
    │   │   ├── transformers_qwen.py    # prod backend (stub ok for Round 1)
    │   │   └── openai_client.py        # one hosted example; others added when needed
    │   └── prompts.py                  # system prompt + templates
    ├── rag/
    │   ├── __init__.py
    │   ├── chunker.py
    │   ├── embedder.py
    │   ├── retriever.py
    │   └── indexer.py                  # scan llm_docs/, upsert into DB
    ├── summarize/
    │   ├── __init__.py
    │   └── email_summarizer.py         # map-reduce
    ├── topics/
    │   ├── __init__.py
    │   └── classifier.py               # extract topics + allow-list filter
    ├── db/
    │   ├── __init__.py
    │   ├── engine.py                   # SQLAlchemy engine/session
    │   └── models.py                   # llm_documents, llm_chunks, llm_embeddings, llm_request_log
    ├── parsers/
    │   ├── __init__.py
    │   ├── pdf.py
    │   ├── docx.py
    │   ├── spreadsheet.py              # xlsx/xls/csv
    │   └── text.py                     # txt/md
    └── utils/
        ├── __init__.py
        ├── hashing.py                  # content hash for change detection
        └── request_id.py
```

**Rules:**
- Do not add a new top-level folder without updating this document.
- Do not create a module file until the phase that owns it is in progress. Empty files = clutter.

---

## 5. Phases (strict order)

Each phase has an entry criterion, exit criterion, and the to-do items in §6. Finish a phase fully before starting the next.

**Phase 0 — Scaffolding.** Create folder structure, placeholder `llm_configs.yml`, both requirements files, empty `run_server.py` with a `--help`. Exit: `python run_server.py --help` prints usage.

**Phase 1 — Bootstrap.** Config loader, conda env detection/creation, platform-aware pip install, logging setup. Exit: running `run_server.py` on a clean Mac creates `dev_advantage_removals` conda env, installs deps, and starts a stub HTTP server with `/health` returning `{"status":"ok"}`.

**Phase 2 — Database.** MySQL connection, SQLAlchemy models, idempotent table creation. Exit: on startup all `llm_*` tables exist; `/health` also reports DB reachability.

**Phase 3 — LLM backend (MLX only).** Implement `LLMClient` + MLX Qwen backend + factory. Exit: a scratch script can call `client.generate([{"role":"user","content":"hi"}])` and get a response from Qwen2.5-7B-Instruct-4bit on the Mac.

**Phase 4 — Document ingestion.** Parsers, chunker, embedder, indexer. Exit: dropping a PDF in `llm_docs/` and calling `POST /documents/reindex` populates `llm_documents` / `llm_chunks` / `llm_embeddings`.

**Phase 5 — RAG retrieval + `/chat`.** Query embedding, top-k lookup, prompt assembly, chat route. Exit: `POST /chat` with a question grounded in a `llm_docs/` file returns a correct answer that cites the doc.

**Phase 6 — Topic allow-list.** Topic extraction + refusal path, applied to `/chat` first. Exit: off-topic chat messages return `refused: true` with a polite reason.

**Phase 7 — `/email` route with summarizer.** Map-reduce summarization, then the same topic filter + RAG. Exit: `POST /email` on a short email returns a grounded reply; on a very long email (test with 50k words to start, scale up) returns a reply after summarization without OOM.

**Phase 8 — Hosted provider (OpenAI only to start).** Add `openai_client.py` backend. Exit: flipping `provider: openai` in YAML and restarting routes all LLM calls through OpenAI with no code change.

**Phase 9 — Prod packaging (lightweight for Round 1).** Linux requirements path verified, gunicorn+uvicorn command for prod, logs routed to `logs/prod/`. Exit: config switched to `active: prod` on a Linux box starts under gunicorn (full prod validation deferred to Round 2).

Phases 8 and 9 are Round-1-nice-to-have; confirm with user whether to include them in Round 1 or push to Round 2 after Phase 7 lands.

---

## 6. To-Do List

Format: `[ ]` todo, `[~]` in progress, `[x]` done. Keep nesting shallow. Add sub-items when a task reveals complexity — but if a task sprouts more than ~5 sub-items, pause and reconsider whether the scope is right.

### Phase 0 — Scaffolding
- [x] Create folder tree per §4 (empty `__init__.py` files only where imports are needed)
- [x] Write `.env.example` listing: `MYSQL_PASSWORD`, `OPENAI_API_KEY` (and other provider keys as placeholders)
- [x] Write minimal `llm_configs.yml` using the schema from `project_requirements.md` §4, dev profile pointed at `mlx-community/Qwen2.5-7B-Instruct-4bit`
- [x] Write `requirements-mac.txt` (fastapi, uvicorn, pyyaml, pydantic, sqlalchemy, pymysql, mlx, mlx-lm, sentence-transformers, pypdf, python-docx, openpyxl, pandas, python-dotenv)
- [x] Write `requirements-linux.txt` (same core + transformers, torch CUDA, accelerate, bitsandbytes; no mlx)
- [x] Write `run_server.py` with `argparse` → `--help` works

### Phase 1 — Bootstrap
- [x] `app/config.py` loads `llm_configs.yml`, validates with pydantic, exposes typed config object; fails fast on missing env vars for the active provider
- [x] `run_server.py` detects whether current interpreter is already inside the target conda env; if not, creates `dev_advantage_removals` / `prod_advantage_removals` and re-execs inside it
- [x] Platform check: macOS → installs `requirements-mac.txt`; Linux → installs `requirements-linux.txt`; errors clearly on anything else
- [x] `app/logging_setup.py` configures rotating file handler to `logs/<env>/YYYY-MM-DD.log` (dev=DEBUG, prod=INFO + separate error.log)
- [x] `app/main.py` creates FastAPI app and registers `/health` (stub returns `{"status":"ok"}`)
- [x] `run_server.py` launches uvicorn (dev) / gunicorn+uvicorn (prod) using host/port from config

### Phase 2 — Database
- [x] `scripts/init_db.sql` — `CREATE TABLE IF NOT EXISTS` for `llm_documents`, `llm_chunks`, `llm_embeddings`, `llm_request_log`
- [x] `app/db/engine.py` — SQLAlchemy engine + session factory using config + `MYSQL_PASSWORD` env
- [x] `app/db/models.py` — SQLAlchemy ORM for the four tables; embeddings stored as JSON column (confirm MySQL version in question #4)
- [x] Bootstrap: on server start, run `init_db.sql`; add DB ping to `/health`
- [x] Confirm with user whether `run_server.py` should also create the database itself (question #5)

### Phase 3 — LLM backend (MLX)
- [x] `app/llm/client.py` — `LLMClient` Protocol with `generate(messages, **kwargs) -> str` and `embed(texts) -> list[list[float]]`
- [x] `app/llm/backends/mlx_qwen.py` — loads Qwen2.5-7B-Instruct-4bit via `mlx_lm`, implements `generate`; downloads on first run
- [x] Embeddings via `sentence-transformers` BAAI/bge-small-en-v1.5 (CPU/MPS); may live in a separate `EmbeddingClient` to keep backends clean
- [x] `app/llm/factory.py` — reads `profiles.<active>.provider` + `backend`, returns the right client
- [x] `app/llm/prompts.py` — system prompt enforcing strict/precise persona per `project_requirements.md` §3.6
- [x] Smoke test script: load client, run a one-shot generation, print output

### Phase 4 — Document ingestion
- [x] `app/parsers/*.py` — one function per file type returning plain text; no OCR
- [x] `app/rag/chunker.py` — token-based chunking per config (default 512 tokens, 64 overlap)
- [x] `app/rag/embedder.py` — wraps the embedding model, batches inputs
- [x] `app/utils/hashing.py` — SHA256 of file bytes
- [x] `app/rag/indexer.py` — scans `llm_docs/`, diffs against `llm_documents` by hash, upserts new/changed, purges deleted
- [x] `app/api/documents.py` — `POST /documents/reindex` triggers the indexer
- [x] Call `indexer.scan()` once on server startup
- [x] Confirm whether live file-watching is needed (question #6); if yes, add `watchdog` observer in a background thread

### Phase 5 — RAG + /chat
- [x] `app/rag/retriever.py` — embed the query, fetch all vectors from `llm_embeddings`, compute cosine in Python (or `VECTOR` type if MySQL 9+), return top-k above `min_similarity`
- [x] `app/api/chat.py` — `POST /chat` → retrieve → compose prompt (system + retrieved context + user msg + rolling session) → `client.generate` → respond
- [x] In-memory session store keyed by `session_id`, last N turns (default 6)
- [x] Log request to `llm_request_log` (route, latency, topics, refused flag)

### Phase 6 — Topic allow-list
- [x] `app/topics/classifier.py` — small LLM call: "which of these topics (if any) does the query match?" returns a list
- [x] On empty match → return `refused: true` with polite message; do not invoke RAG or main LLM
- [x] Apply to `/chat` first; `/email` reuses in Phase 7

### Phase 7 — /email + summarizer
- [x] `app/summarize/email_summarizer.py` — if email length > `chunk_size_words`, split and summarize each chunk, then summarize the summaries ("map-reduce"); enforce ≤ 100 words final
- [x] `app/api/email.py` — summarize → classify topics → refuse-or-retrieve-and-generate → respond
- [x] Test with a short email, a 10k-word email, and a 50k-word email (scale from there)

### Phase 8 — Hosted provider (OpenAI first) — confirm scope
- [x] `app/llm/backends/openai_client.py` — implements `LLMClient` via `openai` SDK, model + base_url from config
- [x] Startup connectivity check when active provider is hosted
- [x] Flip `provider: openai` in dev YAML and verify both routes still work

### Phase 9 — Prod packaging — confirm scope
- [x] Document the Linux-side bootstrap path (`requirements-linux.txt`, CUDA torch wheel pin)
- [x] Confirm gunicorn command: `gunicorn app.main:app -k uvicorn.workers.UvicornWorker -w <N> --bind host:port`
- [x] Prod log path writes to `logs/prod/`

---

## 7. Guardrails — Do Not Do

- Do not introduce Celery, Redis, Kafka, Docker, Kubernetes, Alembic, or a message queue in Round 1.
- Do not build a plugin system, a generic "ModelRegistry," or multi-tenant support.
- Do not add authentication, rate limiting, or request signing unless question #8 says to.
- Do not write OpenAPI-schema customization beyond what FastAPI gives for free.
- Do not add retries/circuit breakers beyond the simple `retries: 2` in config.
- Do not pre-create empty backend files for providers that are not in-scope for Phase 3 or 8.
- Do not write unit tests for every function — write them for the chunker, hashing, topic classifier, and summarizer. Routes get one happy-path integration test each.
- Do not refactor code "while you're in there." Open an issue in this doc under §8 instead.

---

## 7a. Known Future Scope (Round 2+)

Captured so Round 1 does not paint us into a corner — **not** a reason to build anything now.

- **Round 2 introduces agents.** The LLM will route different task types to different data sources (e.g. structured DB queries, specific documents, external APIs). The user will specify the routing rules at the start of Round 2.
- **What this means for Round 1 design choices:**
  - Keep the `LLMClient` interface narrow and provider-agnostic — Round 2 agents will reuse it.
  - Keep retrieval behind a function boundary (`retriever.search(query) -> chunks`) so an agent layer can later call it as one of several tools instead of the only tool.
  - Keep the topic classifier output structured (list of matched topics) — Round 2 may use it as the input signal for agent routing.
- **What NOT to do now:** no agent framework (LangChain agents, LlamaIndex agents, custom tool-dispatch loop), no "tool registry," no plan/act loop. We will design the agent layer in Round 2 once the actual routing rules are known.

## 8. Running Log — Discovered Issues / Deferred Items

Append entries as `YYYY-MM-DD — <note>`. Use this instead of silent scope creep.

- (empty)

---

## 9. Definition of Done — Round 1

Mirrors `project_requirements.md` §11, restated here for convenience:

- [x] `python run_server.py` on the dev Mac creates the conda env, installs deps, downloads the MLX Qwen model, migrates the DB, indexes `llm_docs/`, and starts the server with no manual steps.
- [x] `POST /email` returns a grounded reply for a short email and for a long email (summarized first).
- [x] `POST /chat` returns a grounded reply and respects session context.
- [x] Both routes refuse off-topic queries politely with `refused: true`.
- [x] Adding a new PDF to `llm_docs/` + calling `/documents/reindex` makes its content retrievable without restart.
- [x] All LLM tables carry the `llm_` prefix; no logs go to the DB.
- [x] Switching `provider` to `openai` and restarting routes traffic through the hosted API with no code change (Phase 8, if in scope).
