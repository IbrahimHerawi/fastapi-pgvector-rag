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

1. If this is your first clone and `.env` does not exist yet, create it from the example:

```powershell
Copy-Item .env.example .env
```

2. Install host-side Python deps (used by Alembic, the demo script, and tests):

```powershell
uv sync --dev
```

3. Start Postgres and Ollama:

```powershell
docker compose up -d db ollama
```

4. Wait until Postgres is healthy before running init or migration commands:

```powershell
docker compose ps
```

Continue when the `db` service shows `healthy`.

5. Ensure the test DB exists (`rag_test`, safe to run repeatedly):

```powershell
docker compose exec -T db sh /docker-entrypoint-initdb.d/20-init-test-db.sh
```

6. Run migrations for the main app DB (`rag`):

```powershell
uv run python -m alembic -x dburl=main upgrade head
```

7. Build the local app image and start the API + worker:

```powershell
docker compose up -d --build api worker
```

8. Verify container state:

```powershell
docker compose ps
```

You should see `db` as `healthy` and `api`, `worker`, and `ollama` as `Up`.

9. Verify API health:

```powershell
curl.exe -sS http://127.0.0.1:8000/api/v1/health
```

The `/ask` flow will not work until the Ollama models in the next section are pulled successfully.

## Pull Ollama models

Pull the models configured in `.env` (defaults shown here). Run these after the `ollama` container is up; the first pull can take a while.

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

## Run tests

### Test database setup (exact commands)

1. Start the DB:

```powershell
docker compose up -d db
```

2. Wait until Postgres is healthy:

```powershell
docker compose ps
```

Continue when the `db` service shows `healthy`.

3. Ensure the test DB exists:

```powershell
docker compose exec -T db sh /docker-entrypoint-initdb.d/20-init-test-db.sh
```

4. Run Alembic migrations against the test DB configured in `.env`:

```powershell
uv run python -m alembic -x dburl=test upgrade head
```

5. Run tests:

```powershell
uv run pytest -q
```

## Notes

- `docker-compose.yml` is strictly `.env`-driven (no Compose fallback defaults).
- Postgres is published on `127.0.0.1:${POSTGRES_HOST_PORT}`.
- Host tooling uses `DATABASE_URL` / `TEST_DATABASE_URL`.
- Container services use `DOCKER_DATABASE_URL` and `DOCKER_OLLAMA_BASE_URL`.
- If `/ask` returns `external_service_unavailable` on slower hardware, increase `REQUEST_TIMEOUT_S` in `.env` (default is `120`).
- Runtime settings auto-load `.env`.
- Alembic supports `-x dburl=main|test|<full-url>|<env-var-name>`.
