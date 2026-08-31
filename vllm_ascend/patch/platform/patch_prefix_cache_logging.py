# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import functools
import time
from collections.abc import Callable
from typing import Any

from vllm.logger import logger
from vllm.v1.core.kv_cache_manager import KVCacheManager


def _time_prefix_cache_lookup(
    method: Callable[..., tuple[Any, ...]],
) -> Callable[..., tuple[Any, ...]]:
    """Log the local prefix-cache hit length and lookup latency."""

    @functools.wraps(method)
    def wrapped(
        self: KVCacheManager,
        request: Any,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, ...]:
        lookup_start = time.perf_counter()
        result = method(self, request, *args, **kwargs)
        lookup_time_ms = (time.perf_counter() - lookup_start) * 1000
        if self.enable_caching:
            logger.info(
                "Local prefix cache lookup: request_id=%s, hit_length=%d tokens, query_time=%.3f ms",
                request.request_id,
                result[1],
                lookup_time_ms,
            )
        return result

    setattr(wrapped, "_vllm_ascend_prefix_cache_logging_patched", True)
    return wrapped


if not getattr(
    KVCacheManager.get_computed_blocks,
    "_vllm_ascend_prefix_cache_logging_patched",
    False,
):
    KVCacheManager.get_computed_blocks = _time_prefix_cache_lookup(KVCacheManager.get_computed_blocks)
