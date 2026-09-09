#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PD-disaggregated P-node DP queue stats.

When a prefill-node request is admitted into the running queue, log this DP
rank's running/waiting counts, the sum of running request lengths, and the
sum of tokens that need to be pulled from the KV pool.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from vllm.logger import logger

_KV_POOL_LOAD_TOKENS_ATTR = "_kv_pool_load_tokens"
_INSTALLED_ATTR = "_prefill_dp_queue_stats_installed"
_CONNECTOR_WRAPPED_ATTR = "_prefill_dp_stats_wrapped"


@dataclass(frozen=True)
class PrefillDPQueueStats:
    dp_rank: int
    dp_size: int
    num_running: int
    num_waiting: int
    running_tokens: int
    kv_pool_load_tokens: int
    admitted_request_id: str | None = None
    admitted_count: int = 0


def is_pd_disagg_prefill_node(vllm_config: Any) -> bool:
    """True on PD-disaggregated P nodes (kv_role='kv_producer')."""
    kv_cfg = getattr(vllm_config, "kv_transfer_config", None)
    if kv_cfg is None:
        return False
    return getattr(kv_cfg, "kv_role", None) == "kv_producer"


def _queue_len(queue: Any) -> int:
    if queue is None:
        return 0
    try:
        return len(queue)
    except TypeError:
        return 0


def _iter_queue(queue: Any) -> Iterable[Any]:
    if queue is None:
        return ()
    try:
        return iter(queue)
    except TypeError:
        return ()


def request_token_length(request: Any) -> int:
    """Prefer current sequence length, then prompt length."""
    num_tokens = getattr(request, "num_tokens", None)
    if isinstance(num_tokens, int):
        return num_tokens
    num_prompt_tokens = getattr(request, "num_prompt_tokens", None)
    if isinstance(num_prompt_tokens, int):
        return num_prompt_tokens
    return 0


def _kv_pool_connectors(scheduler: Any) -> list[Any]:
    connector = getattr(scheduler, "connector", None)
    if connector is None:
        return []
    connectors = getattr(connector, "_connectors", None)
    if connectors:
        return list(connectors)
    return [connector]


def lookup_kv_pool_load_tokens(scheduler: Any, request: Any) -> int:
    """Tokens this request still needs to load from the KV pool."""
    request_id = getattr(request, "request_id", None)
    if request_id is None:
        return 0
    total = 0
    for connector in _kv_pool_connectors(scheduler):
        connector_scheduler = getattr(connector, "connector_scheduler", None)
        if connector_scheduler is None:
            continue
        load_specs = getattr(connector_scheduler, "load_specs", None)
        if not load_specs:
            continue
        spec = load_specs.get(request_id)
        if spec is None:
            continue
        kvpool_cached = int(getattr(spec, "kvpool_cached_tokens", 0) or 0)
        vllm_cached = int(getattr(spec, "vllm_cached_tokens", 0) or 0)
        total += max(kvpool_cached - vllm_cached, 0)
    return total


def _live_request_ids(scheduler: Any) -> set[str]:
    live_ids: set[str] = set()
    for request in getattr(scheduler, "running", ()) or ():
        request_id = getattr(request, "request_id", None)
        if request_id is not None:
            live_ids.add(request_id)
    for queue_name in ("waiting", "skipped_waiting"):
        for request in _iter_queue(getattr(scheduler, queue_name, None)):
            request_id = getattr(request, "request_id", None)
            if request_id is not None:
                live_ids.add(request_id)
    return live_ids


def _prune_kv_pool_cache(scheduler: Any) -> None:
    cache = getattr(scheduler, _KV_POOL_LOAD_TOKENS_ATTR, None)
    if not cache:
        return
    live_ids = _live_request_ids(scheduler)
    stale_ids = [request_id for request_id in cache if request_id not in live_ids]
    for request_id in stale_ids:
        cache.pop(request_id, None)


def record_kv_pool_load_tokens(scheduler: Any, request: Any, num_tokens: int | None = None) -> None:
    cache = getattr(scheduler, _KV_POOL_LOAD_TOKENS_ATTR, None)
    if cache is None:
        cache = {}
        setattr(scheduler, _KV_POOL_LOAD_TOKENS_ATTR, cache)
    request_id = getattr(request, "request_id", None)
    if request_id is None:
        return
    if num_tokens is None:
        num_tokens = lookup_kv_pool_load_tokens(scheduler, request)
    if num_tokens:
        cache[request_id] = int(num_tokens)


def collect_prefill_dp_queue_stats(
    scheduler: Any,
    admitted_request_id: str | None = None,
    admitted_count: int = 0,
) -> PrefillDPQueueStats:
    parallel_config = getattr(getattr(scheduler, "vllm_config", None), "parallel_config", None)
    dp_rank = int(getattr(parallel_config, "data_parallel_rank", 0) or 0)
    dp_size = int(getattr(parallel_config, "data_parallel_size", 1) or 1)
    running = list(getattr(scheduler, "running", ()) or ())
    num_waiting = _queue_len(getattr(scheduler, "waiting", None)) + _queue_len(
        getattr(scheduler, "skipped_waiting", None)
    )
    running_tokens = sum(request_token_length(request) for request in running)
    cache = getattr(scheduler, _KV_POOL_LOAD_TOKENS_ATTR, {}) or {}
    kv_pool_load_tokens = 0
    counted_ids: set[str] = set()
    for request in running:
        request_id = getattr(request, "request_id", None)
        if request_id is None:
            continue
        counted_ids.add(request_id)
        kv_pool_load_tokens += int(cache.get(request_id, 0) or lookup_kv_pool_load_tokens(scheduler, request))
    # Async KV-pool loads sit in skipped_waiting (WAITING_FOR_REMOTE_KVS)
    # until they join the running queue.
    for request in _iter_queue(getattr(scheduler, "skipped_waiting", None)):
        request_id = getattr(request, "request_id", None)
        if request_id is None or request_id in counted_ids:
            continue
        counted_ids.add(request_id)
        kv_pool_load_tokens += int(cache.get(request_id, 0) or lookup_kv_pool_load_tokens(scheduler, request))
    return PrefillDPQueueStats(
        dp_rank=dp_rank,
        dp_size=dp_size,
        num_running=len(running),
        num_waiting=num_waiting,
        running_tokens=running_tokens,
        kv_pool_load_tokens=kv_pool_load_tokens,
        admitted_request_id=admitted_request_id,
        admitted_count=admitted_count,
    )


def log_prefill_dp_queue_stats(
    scheduler: Any,
    admitted_request: Any | None = None,
    admitted_count: int = 0,
) -> PrefillDPQueueStats:
    admitted_request_id = None if admitted_request is None else getattr(admitted_request, "request_id", None)
    stats = collect_prefill_dp_queue_stats(
        scheduler,
        admitted_request_id=admitted_request_id,
        admitted_count=admitted_count,
    )
    logger.info(
        "P-node DP queue stats: dp_rank=%d/%d admitted=%d last_req=%s "
        "running=%d waiting=%d running_tokens=%d kv_pool_load_tokens=%d",
        stats.dp_rank,
        stats.dp_size,
        stats.admitted_count,
        stats.admitted_request_id or "-",
        stats.num_running,
        stats.num_waiting,
        stats.running_tokens,
        stats.kv_pool_load_tokens,
    )
    return stats


def _wrap_connector_matched_tokens(scheduler: Any) -> None:
    connector = getattr(scheduler, "connector", None)
    if connector is None or getattr(connector, _CONNECTOR_WRAPPED_ATTR, False):
        return
    original = getattr(connector, "get_num_new_matched_tokens", None)
    if not callable(original):
        return

    def wrapped(request, num_computed_tokens):
        result = original(request, num_computed_tokens)
        ext_tokens = result[0] if isinstance(result, tuple) else result
        if ext_tokens:
            record_kv_pool_load_tokens(scheduler, request, int(ext_tokens))
        return result

    connector.get_num_new_matched_tokens = wrapped
    setattr(connector, _CONNECTOR_WRAPPED_ATTR, True)


def _running_request_ids(scheduler: Any) -> set[str]:
    ids: set[str] = set()
    for request in getattr(scheduler, "running", ()) or ():
        request_id = getattr(request, "request_id", None)
        if request_id is not None:
            ids.add(request_id)
    return ids


def install_prefill_dp_queue_stats(scheduler: Any) -> bool:
    """Hook running-queue admission on a PD-disaggregated P-node scheduler."""
    if getattr(scheduler, _INSTALLED_ATTR, False):
        return False
    vllm_config = getattr(scheduler, "vllm_config", None)
    if not is_pd_disagg_prefill_node(vllm_config):
        return False

    setattr(scheduler, _KV_POOL_LOAD_TOKENS_ATTR, {})
    _wrap_connector_matched_tokens(scheduler)
    original_schedule = scheduler.schedule

    def wrapped_schedule(*args, **kwargs):
        # Connector may be attached after __init__ on some scheduler paths.
        _wrap_connector_matched_tokens(scheduler)
        before_ids = _running_request_ids(scheduler)
        result = original_schedule(*args, **kwargs)
        newly_admitted = [
            request
            for request in getattr(scheduler, "running", ()) or ()
            if getattr(request, "request_id", None) not in before_ids
        ]
        if not newly_admitted:
            return result
        for request in newly_admitted:
            record_kv_pool_load_tokens(scheduler, request)
        _prune_kv_pool_cache(scheduler)
        log_prefill_dp_queue_stats(
            scheduler,
            admitted_request=newly_admitted[-1],
            admitted_count=len(newly_admitted),
        )
        return result

    scheduler.schedule = wrapped_schedule
    setattr(scheduler, _INSTALLED_ATTR, True)
    logger.info(
        "Enabled P-node DP queue stats logging on dp_rank=%s",
        getattr(getattr(vllm_config, "parallel_config", None), "data_parallel_rank", 0),
    )
    return True
