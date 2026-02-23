"""LLM-specific logging helpers for structured logging."""

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def log_llm_invocation(
    model: str,
    prompt_summary: str,
    latency_ms: float,
    attempt: int,
    tokens: Optional[Dict[str, int]] = None,
    correlation_id: Optional[str] = None
) -> None:
    """Log an LLM invocation with structured fields."""
    tokens = tokens or {}
    logger.info(
        "LLM invocation",
        extra={
            "correlation_id": correlation_id,
            "llm": {
                "provider": "ollama",
                "model": model,
                "prompt_summary": prompt_summary,
                "tokens": tokens,
                "latency_ms": latency_ms,
                "attempt": attempt
            }
        }
    )


def log_llm_error(
    error_type: str,
    message: str,
    correlation_id: Optional[str] = None,
    attempt: int = 1,
    stack_trace: Optional[str] = None
) -> None:
    """Log an LLM error with structured fields."""
    logger.error(
        f"LLM error: {error_type}",
        extra={
            "correlation_id": correlation_id,
            "error": {
                "type": error_type,
                "message": message,
                "stack_trace": stack_trace,
                "attempt": attempt
            }
        }
    )


def log_llm_retry(
    model: str,
    attempt: int,
    max_attempts: int,
    reason: str,
    correlation_id: Optional[str] = None
) -> None:
    """Log LLM retry attempt."""
    logger.warning(
        f"LLM retry: {reason}",
        extra={
            "correlation_id": correlation_id,
            "llm": {
                "model": model,
                "attempt": attempt,
                "max_attempts": max_attempts
            },
            "reason": reason
        }
    )


def log_rag_cache_stats(
    cache_hit: bool,
    query: str,
    correlation_id: Optional[str] = None
) -> None:
    """Log RAG cache statistics."""
    logger.debug(
        f"RAG cache: {'hit' if cache_hit else 'miss'}",
        extra={
            "correlation_id": correlation_id,
            "rag": {
                "cache_hit": cache_hit,
                "query": query
            }
        }
    )


def log_document_retrieval(
    query: str,
    document_count: int,
    latency_ms: float,
    correlation_id: Optional[str] = None
) -> None:
    """Log document retrieval from vectorstore."""
    logger.debug(
        "Document retrieval",
        extra={
            "correlation_id": correlation_id,
            "rag": {
                "query": query,
                "document_count": document_count,
                "latency_ms": latency_ms
            }
        }
    )
