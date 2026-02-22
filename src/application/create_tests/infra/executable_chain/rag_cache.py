"""
RAG Cache Manager for caching vectorstore retrieval and embedding results.

Improves performance by:
1. Caching retrieved context documents (similarity search results)
2. Reducing redundant vectorstore queries
3. Avoiding re-embedding similar prompts
"""
import hashlib
from typing import Dict, List, Any, Optional
from langchain_core.documents import Document


class RAGCache:
    """Cache for RAG retrieval results."""

    def __init__(self, max_cache_size: int = 100):
        """Initialize RAG cache.

        Args:
            max_cache_size: Maximum number of cached queries (default: 100)
        """
        self._context_cache: Dict[str, List[Document]] = {}
        self._max_size = max_cache_size
        self._access_count: Dict[str, int] = {}

    def _get_cache_key(self, question: str) -> str:
        """Generate cache key from question using hash.

        Args:
            question: The user question/prompt

        Returns:
            str: Hash-based cache key
        """
        return hashlib.md5(question.encode()).hexdigest()

    def get(self, question: str) -> Optional[List[Document]]:
        """Get cached context for a question.

        Args:
            question: The user question/prompt

        Returns:
            List of cached documents or None if not cached
        """
        cache_key = self._get_cache_key(question)
        if cache_key in self._context_cache:
            self._access_count[cache_key] = self._access_count.get(cache_key, 0) + 1
            return self._context_cache[cache_key]
        return None

    def set(self, question: str, documents: List[Document]) -> None:
        """Cache retrieved context documents.

        Args:
            question: The user question/prompt
            documents: Retrieved documents from vectorstore
        """
        cache_key = self._get_cache_key(question)

        # Evict least recently used if cache is full
        if len(self._context_cache) >= self._max_size and cache_key not in self._context_cache:
            self._evict_lru()

        self._context_cache[cache_key] = documents
        self._access_count[cache_key] = 0

    def _evict_lru(self) -> None:
        """Evict least recently used cache entry."""
        if not self._context_cache:
            return

        lru_key = min(self._access_count, key=self._access_count.get)
        del self._context_cache[lru_key]
        del self._access_count[lru_key]

    def clear(self) -> None:
        """Clear entire cache."""
        self._context_cache.clear()
        self._access_count.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache size, hits count, etc.
        """
        return {
            "cache_size": len(self._context_cache),
            "max_size": self._max_size,
            "total_accesses": sum(self._access_count.values())
        }
