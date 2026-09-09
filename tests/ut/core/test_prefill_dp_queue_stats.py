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
"""Unit tests for PD-disaggregated P-node DP queue stats logging."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm_ascend.core.prefill_dp_queue_stats import (
    collect_prefill_dp_queue_stats,
    install_prefill_dp_queue_stats,
    is_pd_disagg_prefill_node,
    log_prefill_dp_queue_stats,
    lookup_kv_pool_load_tokens,
    record_kv_pool_load_tokens,
    request_token_length,
)


def _vllm_config(kv_role="kv_producer", dp_rank=1, dp_size=4):
    return SimpleNamespace(
        kv_transfer_config=None if kv_role is None else SimpleNamespace(kv_role=kv_role),
        parallel_config=SimpleNamespace(data_parallel_rank=dp_rank, data_parallel_size=dp_size),
    )


def _request(request_id: str, num_tokens: int, num_prompt_tokens: int | None = None):
    return SimpleNamespace(
        request_id=request_id,
        num_tokens=num_tokens,
        num_prompt_tokens=num_prompt_tokens if num_prompt_tokens is not None else num_tokens,
    )


def _load_spec(vllm_cached_tokens: int, kvpool_cached_tokens: int):
    return SimpleNamespace(
        vllm_cached_tokens=vllm_cached_tokens,
        kvpool_cached_tokens=kvpool_cached_tokens,
    )


def _scheduler(
    *,
    kv_role="kv_producer",
    dp_rank=1,
    dp_size=4,
    running=None,
    waiting=None,
    skipped_waiting=None,
    load_specs=None,
):
    connector_scheduler = SimpleNamespace(load_specs=load_specs or {})
    connector = SimpleNamespace(
        connector_scheduler=connector_scheduler,
        get_num_new_matched_tokens=MagicMock(return_value=(0, False)),
    )
    scheduler = SimpleNamespace(
        vllm_config=_vllm_config(kv_role=kv_role, dp_rank=dp_rank, dp_size=dp_size),
        running=list(running or []),
        waiting=list(waiting or []),
        skipped_waiting=list(skipped_waiting or []),
        connector=connector,
        schedule=MagicMock(return_value="scheduled"),
    )
    return scheduler


def test_is_pd_disagg_prefill_node_only_kv_producer():
    assert is_pd_disagg_prefill_node(_vllm_config("kv_producer"))
    assert not is_pd_disagg_prefill_node(_vllm_config("kv_consumer"))
    assert not is_pd_disagg_prefill_node(_vllm_config("kv_both"))
    assert not is_pd_disagg_prefill_node(_vllm_config(None))
    assert not is_pd_disagg_prefill_node(SimpleNamespace())


def test_request_token_length_prefers_num_tokens():
    assert request_token_length(_request("r0", 128, 64)) == 128
    assert request_token_length(SimpleNamespace(num_prompt_tokens=32)) == 32
    assert request_token_length(SimpleNamespace()) == 0


def test_lookup_kv_pool_load_tokens_from_load_spec():
    request = _request("r1", 100)
    scheduler = _scheduler(
        load_specs={"r1": _load_spec(vllm_cached_tokens=16, kvpool_cached_tokens=80)},
    )
    assert lookup_kv_pool_load_tokens(scheduler, request) == 64


def test_lookup_kv_pool_load_tokens_multi_connector():
    request = _request("r1", 100)
    connector_a = SimpleNamespace(
        connector_scheduler=SimpleNamespace(load_specs={"r1": _load_spec(8, 24)}),
    )
    connector_b = SimpleNamespace(
        connector_scheduler=SimpleNamespace(load_specs={"r1": _load_spec(0, 10)}),
    )
    scheduler = SimpleNamespace(connector=SimpleNamespace(_connectors=[connector_a, connector_b]))
    assert lookup_kv_pool_load_tokens(scheduler, request) == 26


def test_collect_stats_counts_running_waiting_and_kv_pool_tokens():
    running = [_request("r0", 128), _request("r1", 256)]
    waiting = [_request("w0", 64), _request("w1", 32)]
    skipped = [_request("s0", 16)]
    scheduler = _scheduler(
        dp_rank=2,
        dp_size=8,
        running=running,
        waiting=waiting,
        skipped_waiting=skipped,
        load_specs={
            "r0": _load_spec(0, 40),
            "r1": _load_spec(8, 8),
            "s0": _load_spec(0, 12),
        },
    )
    record_kv_pool_load_tokens(scheduler, running[0], 40)
    record_kv_pool_load_tokens(scheduler, skipped[0], 12)
    stats = collect_prefill_dp_queue_stats(scheduler, admitted_request_id="r1")
    assert stats.dp_rank == 2
    assert stats.dp_size == 8
    assert stats.num_running == 2
    assert stats.num_waiting == 3
    assert stats.running_tokens == 384
    assert stats.kv_pool_load_tokens == 52
    assert stats.admitted_request_id == "r1"


def test_install_skipped_on_non_prefill_nodes():
    scheduler = _scheduler(kv_role="kv_consumer")
    assert install_prefill_dp_queue_stats(scheduler) is False
    assert not getattr(scheduler, "_prefill_dp_queue_stats_installed", False)


def test_install_is_idempotent_on_prefill_node():
    scheduler = _scheduler()
    assert install_prefill_dp_queue_stats(scheduler) is True
    assert install_prefill_dp_queue_stats(scheduler) is False


def test_install_logs_dp_stats_when_request_joins_running():
    scheduler = _scheduler(
        dp_rank=0,
        dp_size=2,
        waiting=[_request("w0", 32)],
        load_specs={"new": _load_spec(4, 36)},
    )
    original_schedule = scheduler.schedule

    def admit_one(_self=scheduler):
        _self.running.append(_request("new", 100))
        return "ok"

    original_schedule.side_effect = admit_one
    assert install_prefill_dp_queue_stats(scheduler) is True

    with patch("vllm_ascend.core.prefill_dp_queue_stats.logger") as mock_logger:
        assert scheduler.schedule() == "ok"

    assert [req.request_id for req in scheduler.running] == ["new"]
    stats_call = next(
        call for call in mock_logger.info.call_args_list if call.args and "P-node DP queue stats:" in call.args[0]
    )
    assert stats_call.args[1:] == (0, 2, 1, "new", 1, 1, 100, 32)


def test_matched_tokens_are_cached_before_running_admission():
    scheduler = _scheduler()
    original = scheduler.connector.get_num_new_matched_tokens
    original.return_value = (48, True)
    assert install_prefill_dp_queue_stats(scheduler) is True

    request = _request("async-kv", 80)
    ext_tokens, load_async = scheduler.connector.get_num_new_matched_tokens(request, 0)
    assert ext_tokens == 48
    assert load_async is True
    assert scheduler._kv_pool_load_tokens["async-kv"] == 48


def test_log_prefill_dp_queue_stats_format():
    scheduler = _scheduler(running=[_request("r0", 10)])
    with patch("vllm_ascend.core.prefill_dp_queue_stats.logger") as mock_logger:
        stats = log_prefill_dp_queue_stats(scheduler, admitted_request=_request("r0", 10), admitted_count=1)
    assert stats.num_running == 1
    assert stats.admitted_count == 1
    mock_logger.info.assert_called_once()
    assert "P-node DP queue stats:" in mock_logger.info.call_args.args[0]


def test_no_log_when_running_queue_unchanged():
    scheduler = _scheduler(running=[_request("r0", 10)])
    original_schedule = scheduler.schedule
    original_schedule.return_value = "idle"
    assert install_prefill_dp_queue_stats(scheduler) is True
    with patch("vllm_ascend.core.prefill_dp_queue_stats.logger") as mock_logger:
        assert scheduler.schedule() == "idle"
    assert not any(call.args and "P-node DP queue stats:" in call.args[0] for call in mock_logger.info.call_args_list)
