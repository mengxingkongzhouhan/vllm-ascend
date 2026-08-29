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
# This file is a part of the vllm-ascend project.

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "benchmarks" / "scripts" / "two_tier_prefix_bench.py"


def _load_script() -> ModuleType:
    """Import the benchmark script by path; it is a CLI, not an installed package."""
    sys.modules.setdefault("aiohttp", MagicMock())
    spec = importlib.util.spec_from_file_location("two_tier_prefix_bench", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations through sys.modules, so register the
    # module before executing it.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load_script()


def make_args(**overrides) -> argparse.Namespace:
    # A per-engine cache of 64947 tokens holds about 6.9 prefixes of 8192
    # tokens once one in-flight request is accounted for.
    args = argparse.Namespace(
        prefix_len=8192,
        suffix_len=128,
        num_prompts=600,
        hot_prefixes=4,
        warm_prefixes=64,
        hot_fraction=0.5,
        num_suffixes=8,
        concurrency=1,
        engines=1,
        kv_cache_tokens=64947,
        block_size=128,
        sticky_routing=False,
        seed=1000,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class FakeTokenizer:
    """One token per whitespace word, round-tripping content like a real one."""

    def encode(self, text, add_special_tokens=False):
        self._words = text.split()
        return list(range(len(self._words)))

    def decode(self, ids):
        return " ".join(self._words[index] for index in ids)


class CollapsingTokenizer:
    """Pathological tokenizer whose decode discards the input content."""

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))

    def decode(self, ids):
        return " ".join(f"t{index}" for index in ids)


@pytest.mark.parametrize(
    ("value", "multiple", "expected"),
    [(30856, 128, 30976), (30720, 128, 30720), (1, 128, 128)],
)
def test_align_up_rounds_to_whole_blocks(value, multiple, expected):
    assert bench.align_up(value, multiple) == expected


def test_workable_plan_has_no_problems():
    plan = bench.plan_capacity(make_args())

    assert plan.problems == []
    assert plan.request_tokens == 8320
    assert plan.hot_per_engine == pytest.approx(4.0)
    assert plan.warm_slots == pytest.approx(plan.prefix_slots - 4.0)


def test_prefix_longer_than_the_free_cache_is_rejected():
    # The reported case: 30720-token prefixes against a 64947-token cache leave
    # room for barely one prefix, so no ordering can keep two tiers alive.
    plan = bench.plan_capacity(make_args(prefix_len=30720, hot_prefixes=10, warm_prefixes=0, hot_fraction=1.0))

    assert plan.problems
    assert "for the warm tier" in plan.problems[0]


def test_concurrency_that_consumes_the_cache_is_rejected():
    plan = bench.plan_capacity(make_args(prefix_len=30720, concurrency=32, engines=16))

    assert plan.problems
    assert "no prefix can stay cached" in plan.problems[0]


def test_warm_set_that_fits_locally_is_rejected_with_a_target():
    plan = bench.plan_capacity(make_args(warm_prefixes=8))

    assert plan.problems
    assert "Raise --warm-prefixes to at least" in plan.problems[0]


def test_phase_too_short_to_cover_the_warm_set_is_rejected():
    plan = bench.plan_capacity(make_args(num_prompts=64, warm_prefixes=64))

    assert plan.problems
    assert "Raise --num-prompts to at least" in plan.problems[-1]


def test_sticky_routing_divides_the_tiers_across_engines():
    shared = bench.plan_capacity(make_args(engines=16, warm_prefixes=384, hot_prefixes=16))
    sticky = bench.plan_capacity(make_args(engines=16, warm_prefixes=384, hot_prefixes=16, sticky_routing=True))

    # Without sticky routing every engine eventually sees every prefix; with it
    # each engine only sees its share, which is what must overflow its cache.
    assert shared.hot_per_engine == pytest.approx(16.0)
    assert sticky.hot_per_engine == pytest.approx(1.0)
    assert sticky.warm_per_engine == pytest.approx(24.0)


def _sequence(**overrides):
    args = make_args(**overrides)
    hot = [f"HOT{index}" for index in range(args.hot_prefixes)]
    warm = [f"WARM{index}" for index in range(args.warm_prefixes)]
    suffixes = [f"S{index}" for index in range(args.num_suffixes)]
    return args, hot, warm, bench.build_sequence(hot, warm, suffixes, args)


def test_sequence_respects_the_requested_tier_mix():
    args, _, _, sequence = _sequence()

    assert len(sequence) == args.num_prompts
    hot_share = sum(1 for request in sequence if request.tier == "hot") / len(sequence)
    assert hot_share == pytest.approx(args.hot_fraction, abs=0.1)


def test_warm_tier_is_cycled_so_reuse_distance_is_maximal():
    _, _, warm, sequence = _sequence()
    warm_order = [request.prefix for request in sequence if request.tier == "warm"]

    # A full cycle before any repeat means the distance between two uses of a
    # warm prefix is the size of the warm set, maximising eviction in between.
    assert len(set(warm_order[: len(warm)])) == len(warm)
    first, second = warm_order.index(warm[0]), warm_order.index(warm[0], 1)
    assert second - first == len(warm)


def test_sequence_is_reproducible_for_a_fixed_seed():
    _, _, _, first = _sequence()
    _, _, _, second = _sequence()

    assert [(request.tier, request.prefix) for request in first] == [
        (request.tier, request.prefix) for request in second
    ]


def test_prefixes_are_distinct_and_reproducible():
    texts = bench.build_prefix_texts(FakeTokenizer(), 5, 32, seed=7)

    assert len(set(texts)) == 5
    assert all(len(text.split()) == 32 for text in texts)
    assert texts == bench.build_prefix_texts(FakeTokenizer(), 5, 32, seed=7)


def test_indistinct_prefixes_fail_loudly():
    # Prefixes that collapse into one would silently turn the whole workload
    # into a single-prefix run, so this must not pass quietly.
    with pytest.raises(RuntimeError, match="distinct prefixes"):
        bench.build_prefix_texts(CollapsingTokenizer(), 5, 32, seed=7)
