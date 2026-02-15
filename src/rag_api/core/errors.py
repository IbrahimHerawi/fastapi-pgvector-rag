"""Custom error hierarchy for rag_api."""


class RagAPIError(Exception):
    """Base exception for rag_api."""


class ConfigurationError(RagAPIError):
    """Raised when required configuration is missing or invalid."""
