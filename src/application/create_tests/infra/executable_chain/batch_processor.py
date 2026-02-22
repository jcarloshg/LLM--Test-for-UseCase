"""Batch processor for executing multiple test cases concurrently.

Handles async execution of test cases with rate limiting and progress tracking.
"""
import asyncio
import time
from typing import List, Callable, Any, Optional, TypeVar, Generic

T = TypeVar('T')


class BatchProcessor(Generic[T]):
    """Generic batch processor for concurrent async operations."""

    def __init__(self, max_concurrent: int = 3, timeout: Optional[float] = None):
        """Initialize batch processor.

        Args:
            max_concurrent: Maximum concurrent operations (default: 3)
            timeout: Timeout per operation in seconds (default: None)
        """
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self._executed = 0
        self._failed = 0
        self._total_time = 0.0

    async def process(
        self,
        items: List[T],
        async_func: Callable[[T], Any],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Any]:
        """Process items concurrently.

        Args:
            items: List of items to process
            async_func: Async function to apply to each item
            progress_callback: Optional callback(current, total) for progress tracking

        Returns:
            List of results in the same order as input items
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        start_time = time.time()

        async def process_with_semaphore(item: T, index: int) -> tuple[int, Any]:
            async with semaphore:
                try:
                    if self.timeout:
                        result = await asyncio.wait_for(
                            async_func(item), timeout=self.timeout
                        )
                    else:
                        result = await async_func(item)

                    self._executed += 1
                    if progress_callback:
                        progress_callback(self._executed, len(items))

                    return index, result
                except Exception as e:
                    self._failed += 1
                    if progress_callback:
                        progress_callback(self._executed + self._failed, len(items))
                    raise

        try:
            # Execute all items concurrently
            tasks = [
                process_with_semaphore(item, idx) for idx, item in enumerate(items)
            ]
            results_with_indices = await asyncio.gather(*tasks, return_exceptions=True)

            self._total_time = time.time() - start_time

            # Reconstruct results in original order
            results_dict = {}
            for item in results_with_indices:
                if isinstance(item, Exception):
                    raise item
                idx, result = item
                results_dict[idx] = result

            return [results_dict[i] for i in range(len(items))]

        except Exception as e:
            self._total_time = time.time() - start_time
            raise e

    def get_stats(self) -> dict:
        """Get processing statistics.

        Returns:
            Dict with execution stats (executed, failed, total_time, throughput)
        """
        throughput = self._executed / self._total_time if self._total_time > 0 else 0
        return {
            "executed": self._executed,
            "failed": self._failed,
            "total_time_seconds": round(self._total_time, 2),
            "throughput_items_per_second": round(throughput, 2),
            "max_concurrent": self.max_concurrent,
        }
