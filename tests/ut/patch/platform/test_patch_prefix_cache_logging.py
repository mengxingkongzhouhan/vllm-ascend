# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_ascend.patch.platform import patch_prefix_cache_logging


def test_prefix_cache_lookup_logs_hit_length_and_query_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_method = MagicMock(return_value=("blocks", 128, 64))
    timed_method = patch_prefix_cache_logging._time_prefix_cache_lookup(original_method)
    perf_counter = MagicMock(side_effect=[10.0, 10.00125])
    monkeypatch.setattr(patch_prefix_cache_logging.time, "perf_counter", perf_counter)
    logger_info = MagicMock()
    monkeypatch.setattr(patch_prefix_cache_logging.logger, "info", logger_info)
    request = SimpleNamespace(request_id="request-1")

    result = timed_method(SimpleNamespace(enable_caching=True), request)

    assert result == ("blocks", 128, 64)
    logger_info.assert_called_once_with(
        "Local prefix cache lookup: request_id=%s, hit_length=%d tokens, query_time=%.3f ms",
        "request-1",
        128,
        pytest.approx(1.25),
    )


def test_prefix_cache_lookup_preserves_query_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    error = RuntimeError("lookup failed")
    original_method = MagicMock(side_effect=error)
    timed_method = patch_prefix_cache_logging._time_prefix_cache_lookup(original_method)
    logger_info = MagicMock()
    monkeypatch.setattr(patch_prefix_cache_logging.logger, "info", logger_info)

    with pytest.raises(RuntimeError, match="lookup failed"):
        timed_method(
            SimpleNamespace(enable_caching=True),
            SimpleNamespace(request_id="request-2"),
        )

    logger_info.assert_not_called()


def test_prefix_cache_lookup_does_not_log_when_caching_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timed_method = patch_prefix_cache_logging._time_prefix_cache_lookup(
        MagicMock(return_value=("blocks", 0))
    )
    logger_info = MagicMock()
    monkeypatch.setattr(patch_prefix_cache_logging.logger, "info", logger_info)

    timed_method(SimpleNamespace(enable_caching=False), SimpleNamespace(request_id="request-3"))

    logger_info.assert_not_called()
