# Advantage Removals — Local LLM RAG Assistant

Round 1 build. A local Qwen (or hosted provider) RAG assistant exposed over HTTP for email replies and website chat.

See `project_structure.txt`, `project_requirements.md`, and `development_document.md` for the full specification and build plan.

---

## Prerequisites on the dev machine (MacBook Air M5)

Install these once before the first run:

1. **Miniforge** (Apple Silicon-native conda).
   ```bash
   brew install miniforge
   conda init zsh && exec $SHELL -l
   ```

2. **MySQL** (local).
   ```bash
   brew install mysql
   brew services start mysql
   # Set a password for root (or create a user), then:
   #   export MYSQL_PASSWORD='your-password'
   ```

3. **A `.env` file** at the project root (copy from `.env.example`):
   ```bash
   cp .env.example .env
   # Edit .env and set MYSQL_PASSWORD (and OPENAI_API_KEY if using OpenAI).
   ```

## Run the server

```bash
python3 run_server.py
```

That single command:

1. Creates the conda env `dev_advantage_removals` if missing.
2. Installs deps from `requirements-mac.txt`.
3. Re-executes inside that env.
4. Creates the MySQL database `advantage_removals` if missing.
5. Applies `scripts/init_db.sql` to create the `llm_*` tables.
6. Scans `llm_docs/` and indexes any files (first run: embeddings model is downloaded).
7. On first real LLM call: the MLX Qwen2.5-7B-Instruct-4bit model is downloaded (~4 GB).
8. Starts uvicorn on `0.0.0.0:8000` (dev) with auto-reload.

Hit the server:

```bash
curl -s localhost:8000/health | jq
curl -s -X POST localhost:8000/documents/reindex | jq
curl -s -X POST localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"text":"What is your booking process?","session_id":"s1"}' | jq
curl -s -X POST localhost:8000/email \
  -H 'content-type: application/json' \
  -d '{"text":"Hi, I need to book a removal for next Thursday from Manchester to London..."}' | jq
```

Or use the end-to-end smoke test:

```bash
python3 smoke_test.py
```

## Switch LLM backend

Edit `llm_configs.yml` → `env.profiles.dev`:

| Goal                                        | Setting                                                     |
| ------------------------------------------- | ----------------------------------------------------------- |
| Real local Qwen on the Mac                  | `provider: local_qwen`, `backend: mlx`                      |
| Fast pipeline smoke test (no model download)| `provider: local_qwen`, `backend: mock`                     |
| Hosted OpenAI (ChatGPT)                     | `provider: openai`, `backend: null`, `model: gpt-4o-mini`   |

Restart `run_server.py` after any change. Other providers (Anthropic, Gemini, Mistral, Cohere, Azure, Bedrock, Groq, Together, xAI) are declared in the config but their SDK adapters land in Round 2 if needed.

## Run in prod

Edit `llm_configs.yml` → `env.active: prod`. On a Linux + NVIDIA host, `python3 run_server.py` builds `prod_advantage_removals` from `requirements-linux.txt` and launches `gunicorn` with `uvicorn.workers.UvicornWorker`. Logs go to `logs/prod/`.

## Add documents

Drop files into `llm_docs/` (PDF, DOC/DOCX, XLSX/XLS, CSV, TXT, MD). Then call `POST /documents/reindex` — or restart the server. Files that haven't changed are skipped (SHA256 diff). Deleted files are purged from the DB on next scan.

## Logs

- Dev: `logs/dev/app.log` (rotated nightly, 14 days retention).
- Prod: `logs/prod/app.log` + `logs/prod/error.log`.
