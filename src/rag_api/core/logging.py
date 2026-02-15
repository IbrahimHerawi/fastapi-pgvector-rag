"""Logging placeholders for rag_api."""

import logging


def configure_logging(level: int | str = logging.INFO) -> None:
    logging.basicConfig(level=level)
