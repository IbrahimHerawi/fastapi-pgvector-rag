"""Data model package for rag_api."""

try:
    from rag_api.models.schema import Base, Chunk, Document, IngestionJob, QueryLog
except ModuleNotFoundError as exc:
    # Keep package importable in environments where DB dependencies are not installed yet.
    if exc.name and exc.name.startswith("sqlalchemy"):
        Base = None  # type: ignore[assignment]
        Document = None  # type: ignore[assignment]
        Chunk = None  # type: ignore[assignment]
        QueryLog = None  # type: ignore[assignment]
        IngestionJob = None  # type: ignore[assignment]
    else:
        raise

__all__ = [
    "Base",
    "Chunk",
    "Document",
    "IngestionJob",
    "QueryLog",
]
