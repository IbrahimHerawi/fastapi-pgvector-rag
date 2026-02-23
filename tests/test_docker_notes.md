# Docker verification notes

Manual verification steps for Task 05:

1. `docker compose build`
2. `docker compose up -d db ollama`
3. `docker compose up api`

Expected result:
- Build completes successfully.
- API container starts and serves on `http://localhost:8000`.
- Ollama is reachable only on `127.0.0.1:11434`.
