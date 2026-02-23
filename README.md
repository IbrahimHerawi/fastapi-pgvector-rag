# fastapi-pgvector-rag

`fastapi-pgvector-rag` is a local Retrieval-Augmented Generation (RAG) service that combines:

- FastAPI for HTTP endpoints
- PostgreSQL + `pgvector` for persistent document, chunk, and embedding storage
- Ollama for both embedding generation and answer generation
- A background worker that asynchronously ingests documents

It is designed so you can run everything on a single machine with Docker Compose, then interact with it through simple API calls.

## Project overview

This project exposes a versioned API (`/api/v1`) for document ingestion and grounded question-answering:

- `POST /documents`: stores a document and creates an ingestion job
- `GET /documents/{document_id}/status`: lets clients poll ingestion progress (`pending`, `processing`, `done`, `failed`)
- `POST /ask`: embeds the question, retrieves similar chunks from `pgvector`, generates a response with Ollama, and returns answer + sources
- `GET /health`: basic health endpoint for service checks

The ingestion and query flow is:

1. A client submits a document (`title`, `source`, `content`).
2. The API saves the document and queues an ingestion job.
3. The worker claims pending jobs, chunks document text, requests embeddings from Ollama, and stores vectors in Postgres.
4. A user asks a question.
5. The API embeds the question, runs vector similarity retrieval over stored chunks, and sends retrieved context to the generation model.
6. The API returns the final answer with source metadata and logs query details.

## Core features

- Async ingestion pipeline:
  Documents are accepted quickly, then processed in the background so write latency stays predictable.
- Vector retrieval in Postgres:
  Embeddings are stored in `pgvector`, enabling cosine-similarity search without external vector infrastructure.
- Source-grounded answers:
  `/ask` returns source snippets and chunk metadata to help users verify where answers came from.
- Query logging:
  Questions, answers, models used, retrieval chunk IDs, and latency are persisted for observability/debugging.
- Configurable runtime:
  Model names, chunking limits, retrieval `top_k`, and timeouts are controlled by environment variables.
- Local-first developer workflow:
  Docker Compose brings up dependencies; Alembic manages schema migrations for both main and test databases.
- Optional API key auth:
  If `API_KEY` is configured, endpoints require `X-API-Key`; otherwise local development is frictionless.
- Test-friendly setup:
  Dedicated test database URL + migration flow is documented so tests can run against a real migrated schema.

## Prerequisites

- Docker Desktop (with `docker compose`)
- Python 3.13
- [`uv`](https://docs.astral.sh/uv/)

## Docker Compose quickstart

1. Create your local env file:

```powershell
Copy-Item .env.example .env
```

2. Install Python deps (needed for Alembic and tests):

```powershell
uv sync --dev
```

3. Start Postgres and Ollama:

```powershell
docker compose up -d db ollama
```

4. Ensure the test DB exists (`rag_test`, safe to run repeatedly):

```powershell
docker compose exec -T db sh /docker-entrypoint-initdb.d/20-init-test-db.sh
```

5. Run migrations for the main app DB (`rag`):

```powershell
uv run python -m alembic -x dburl=main upgrade head
```

6. Start API + worker:

```powershell
docker compose up -d --build api worker
```

7. Verify API health:

```powershell
curl.exe -sS http://127.0.0.1:8000/api/v1/health
```

## Pull Ollama models

Pull the models configured in `.env` (defaults shown here):

```powershell
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull llama3.1:8b
```

If you changed `OLLAMA_EMBED_MODEL` or `OLLAMA_CHAT_MODEL`, pull those names instead.

## Runnable demo script

Run the full flow (create document -> wait for ingestion -> ask question):

```powershell
uv run python scripts/demo/run_demo.py
```

Optional flags:

```powershell
uv run python scripts/demo/run_demo.py --api-key "your-key" --question "What does this project use for vector search?"
```

## Demo: create document -> poll status -> ask (curl)

Use this if you want explicit API calls instead of the demo script.
If `API_KEY` is set in `.env`, add `-H "X-API-Key: <value>"` to each request.

```powershell
$BaseUrl = "http://127.0.0.1:8000/api/v1"

$createPayload = @'
{
  "title": "RAG quickstart note",
  "source": "readme-demo",
  "content": "pgvector stores embeddings in Postgres. The worker chunks and embeds documents using Ollama."
}
'@

$create = curl.exe -sS -X POST "$BaseUrl/documents" `
  -H "Content-Type: application/json" `
  --data "$createPayload" | ConvertFrom-Json

$create | ConvertTo-Json
$documentId = $create.document_id
```

```powershell
do {
  Start-Sleep -Seconds 1
  $status = curl.exe -sS "$BaseUrl/documents/$documentId/status" | ConvertFrom-Json
  $status | ConvertTo-Json
} while ($status.status -in @("pending", "processing"))
```

```powershell
$askPayload = @'
{
  "question": "How are embeddings generated in this project?"
}
'@

curl.exe -sS -X POST "$BaseUrl/ask" `
  -H "Content-Type: application/json" `
  --data "$askPayload"
```

## Run tests

### Test database setup (exact commands)

1. Start DB and ensure the test DB exists:

```powershell
docker compose up -d db
docker compose exec -T db sh /docker-entrypoint-initdb.d/20-init-test-db.sh
```

2. Set `TEST_DATABASE_URL` in your shell:

```powershell
$env:TEST_DATABASE_URL="postgresql+psycopg://postgres:postgres@127.0.0.1:55432/rag_test"
```

3. Run Alembic migrations against the test DB:

```powershell
uv run python -m alembic -x dburl=test upgrade head
```

4. Run tests:

```powershell
uv run pytest -q
```

## Notes

- `docker-compose.yml` is strictly `.env`-driven (no Compose fallback defaults).
- Postgres is published on `127.0.0.1:${POSTGRES_HOST_PORT}`.
- Host tooling uses `DATABASE_URL` / `TEST_DATABASE_URL`.
- Container services use `DOCKER_DATABASE_URL` and `DOCKER_OLLAMA_BASE_URL`.
- Runtime settings auto-load `.env`.
- Alembic supports `-x dburl=main|test|<full-url>|<env-var-name>`.
