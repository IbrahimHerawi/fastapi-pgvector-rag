import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import UniqueConstraint

from rag_api.models import Base


def _column_names(table_name: str) -> set[str]:
    table = Base.metadata.tables[table_name]
    return set(table.columns.keys())


def _has_unique_constraint(table_name: str, column_names: tuple[str, ...]) -> bool:
    table = Base.metadata.tables[table_name]
    for constraint in table.constraints:
        if not isinstance(constraint, UniqueConstraint):
            continue
        if tuple(col.name for col in constraint.columns) == column_names:
            return True
    return False


def test_expected_tables_exist_in_metadata() -> None:
    assert Base is not None

    expected_tables = {"documents", "chunks", "query_logs", "ingestion_jobs"}
    assert expected_tables.issubset(set(Base.metadata.tables.keys()))


def test_expected_columns_exist_in_metadata() -> None:
    assert Base is not None

    assert {
        "id",
        "title",
        "source",
        "content",
        "content_sha256",
        "created_at",
    }.issubset(_column_names("documents"))

    assert {
        "id",
        "document_id",
        "chunk_index",
        "start_char",
        "end_char",
        "text",
        "embedding",
        "created_at",
    }.issubset(_column_names("chunks"))

    assert {
        "id",
        "question",
        "answer",
        "retrieved_chunk_ids",
        "models",
        "latency_ms",
        "created_at",
    }.issubset(_column_names("query_logs"))

    assert {
        "id",
        "document_id",
        "status",
        "error",
        "created_at",
        "updated_at",
    }.issubset(_column_names("ingestion_jobs"))


def test_expected_unique_constraints_exist() -> None:
    assert Base is not None

    assert _has_unique_constraint("ingestion_jobs", ("document_id",))
    assert _has_unique_constraint("chunks", ("document_id", "chunk_index"))
