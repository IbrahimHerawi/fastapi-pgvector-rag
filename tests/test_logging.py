import logging

import pytest

from rag_api.core.logging import (
    clear_request_id,
    configure_logging,
    reset_request_id,
    set_request_id,
)


@pytest.fixture(autouse=True)
def clear_request_context() -> None:
    clear_request_id()
    yield
    clear_request_id()


def test_structured_log_contains_level_message_and_request_id(caplog: pytest.LogCaptureFixture) -> None:
    configure_logging(level=logging.INFO)
    logger = logging.getLogger("rag_api.test.logging")
    caplog.set_level(logging.INFO, logger=logger.name)

    token = set_request_id("req-123")
    try:
        logger.info("hello structured logging")
    finally:
        reset_request_id(token)

    assert "level=INFO" in caplog.text
    assert "request_id=req-123" in caplog.text
    assert 'message="hello structured logging"' in caplog.text


def test_structured_log_has_request_id_key_without_context(caplog: pytest.LogCaptureFixture) -> None:
    configure_logging(level=logging.INFO)
    logger = logging.getLogger("rag_api.test.logging.default")
    caplog.set_level(logging.INFO, logger=logger.name)

    logger.info("no request context")

    assert "level=INFO" in caplog.text
    assert "request_id=-" in caplog.text
    assert 'message="no request context"' in caplog.text
