# fastapi-pgvector-rag

## DB test prerequisites

Set test env vars (example values are in `.env.test.example`):

- `APP_ENV=test`
- `TEST_DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag_test`

Then run:

```powershell
python -m pytest tests/test_db_engine.py -q
```
