# fastapi-pgvector-rag

## Local setup (practical standard)

1. Create your local env file:

```powershell
Copy-Item .env.example .env
```

2. Start the shared Postgres instance:

```powershell
docker compose up -d db ollama
```

3. Ensure `rag_test` exists in the same `db` container (safe to run repeatedly):

```powershell
docker compose exec -T db sh /docker-entrypoint-initdb.d/20-init-test-db.sh
```

4. Run migrations for app DB (`rag`):

```powershell
.\.venv\Scripts\python.exe -m alembic -x dburl=main upgrade head
```

5. Run migrations for test DB (`rag_test`):

```powershell
.\.venv\Scripts\python.exe -m alembic -x dburl=test upgrade head
```

6. Run all tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Notes:

- `docker-compose.yml` is now strictly `.env`-driven (no fallback defaults in Compose).
- Postgres is published on `127.0.0.1:${POSTGRES_HOST_PORT}`.
- Host tooling (Alembic/pytest) uses `DATABASE_URL` and `TEST_DATABASE_URL` from `.env`.
- Container services use `DOCKER_DATABASE_URL` so `api` and `worker` still connect to `db` on the Compose network.
- For pytest runs, you do not need to export `APP_ENV=test`; test fixtures set test mode automatically.
- Runtime settings auto-load `.env`.
- Alembic accepts `-x dburl=main|test|<full-url>|<env-var-name>` and chooses target DB in `alembic/env.py`.
