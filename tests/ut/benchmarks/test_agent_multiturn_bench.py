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
import asyncio

import pytest

from tests.ut.benchmarks.loader import FakeTokenizer, load_bench_script

bench = load_bench_script("agent_multiturn_bench")


def make_args(**overrides) -> argparse.Namespace:
    args = argparse.Namespace(
        host="127.0.0.1",
        port=8000,
        endpoint="/v1/chat/completions",
        model="test-model",
        request_timeout=60.0,
        system_len=1024,
        session_context_len=4096,
        user_len=256,
        output_len=256,
        turns=6,
        sessions=768,
        local_session_fraction=0.25,
        num_user_messages=32,
        concurrency=16,
        engines=16,
        kv_cache_tokens=64947,
        max_model_len=40960,
        block_size=128,
        sticky_routing=True,
        seed=1000,
        save_dataset=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_prompt_grows_with_each_turn():
    args = make_args()

    first = bench.turn_prompt_tokens(args, 0)
    second = bench.turn_prompt_tokens(args, 1)

    # Turn one carries the preamble, the session context and one user message.
    assert first == 1024 + 4096 + 256
    # Each later turn adds the previous reply and the new user message.
    assert second - first == args.user_len + args.output_len


def test_workable_plan_has_no_problems():
    plan = bench.plan_capacity(make_args())

    assert plan.problems == []
    assert plan.resident_sessions > 1
    assert plan.local_per_engine == pytest.approx(12.0)
    assert plan.pooled_per_engine == pytest.approx(36.0)


def test_conversation_outgrowing_max_model_len_is_rejected():
    plan = bench.plan_capacity(make_args(turns=200))

    assert plan.problems
    assert "past --max-model-len" in plan.problems[0]


def test_conversation_too_large_for_the_cache_is_rejected():
    plan = bench.plan_capacity(make_args(session_context_len=30000, max_model_len=131072))

    assert plan.problems
    assert "not even a local session can stay cached" in plan.problems[0]


def test_concurrency_beyond_what_the_cache_holds_is_rejected():
    # Only the in-flight working set has to stay resident for the local tier to
    # hit, so this fires on concurrency rather than on the size of the tier.
    plan = bench.plan_capacity(make_args(concurrency=96))

    assert plan.problems
    assert "evict each other instead of hitting locally" in plan.problems[0]


def test_local_tier_alone_leaves_nothing_for_the_pool():
    plan = bench.plan_capacity(make_args(local_session_fraction=1.0))

    assert plan.problems
    assert "Raise --sessions to at least" in plan.problems[-1]


def test_pooled_tier_that_survives_between_turns_is_rejected():
    plan = bench.plan_capacity(make_args(sessions=16))

    assert plan.problems
    assert "Raise --sessions to at least" in plan.problems[-1]


def test_sticky_routing_divides_sessions_across_engines():
    shared = bench.plan_capacity(make_args(sticky_routing=False))
    sticky = bench.plan_capacity(make_args(sticky_routing=True))

    assert shared.pooled_per_engine == pytest.approx(576.0)
    assert sticky.pooled_per_engine == pytest.approx(36.0)


def test_sessions_open_with_the_shared_preamble_and_own_context():
    args = make_args(sessions=4, local_session_fraction=0.5)
    contexts = [f"CTX{index}" for index in range(args.sessions)]

    sessions = bench.build_sessions(args, "SYSTEM", contexts)

    assert [session.tier for session in sessions] == [
        bench.LOCAL_TIER,
        bench.LOCAL_TIER,
        bench.POOLED_TIER,
        bench.POOLED_TIER,
    ]
    for index, session in enumerate(sessions):
        assert session.messages[0] == {"role": "system", "content": "SYSTEM"}
        assert session.messages[1] == {"role": "user", "content": contexts[index]}


def test_local_sessions_are_requeued_ahead_of_pooled_ones(monkeypatch):
    args = make_args(sessions=4, turns=2, concurrency=1, local_session_fraction=0.5)
    sessions = bench.build_sessions(args, "SYSTEM", [f"CTX{index}" for index in range(4)])
    served: list[tuple[int, int]] = []

    async def fake_send_chat(http, url, payload, timeout):
        return bench.ChatResult(ok=True, ttft=0.01, latency=0.02, text="REPLY")

    monkeypatch.setattr(bench, "send_chat", fake_send_chat)
    runner = bench.Runner(args, sessions, ["ASK"])

    original_serve = runner._serve_turn

    async def tracking_serve(http, session):
        served.append((session.session_id, session.turns_done))
        await original_serve(http, session)

    runner._serve_turn = tracking_serve
    asyncio.run(runner.run())

    # Every session completes all of its turns.
    assert sorted(served) == [(index, turn) for index in range(4) for turn in range(2)]
    # A local session's second turn is served immediately after its first, while
    # a pooled session waits behind everything else that is queued.
    local_gap = served.index((0, 1)) - served.index((0, 0))
    pooled_gap = served.index((2, 1)) - served.index((2, 0))
    assert local_gap < pooled_gap


def test_turns_accumulate_the_reply_and_the_next_question(monkeypatch):
    args = make_args(sessions=1, turns=3, concurrency=1, local_session_fraction=1.0)
    sessions = bench.build_sessions(args, "SYSTEM", ["CTX"])
    prompt_lengths: list[int] = []

    async def fake_send_chat(http, url, payload, timeout):
        prompt_lengths.append(len(payload["messages"]))
        return bench.ChatResult(ok=True, ttft=0.01, latency=0.02, text="REPLY")

    monkeypatch.setattr(bench, "send_chat", fake_send_chat)
    runner = bench.Runner(args, sessions, ["ASK"])
    asyncio.run(runner.run())

    # system + user, then + assistant + user each turn.
    assert prompt_lengths == [2, 4, 6]
    assert sessions[0].messages[2] == {"role": "assistant", "content": "REPLY"}


def test_failed_turns_are_counted_and_do_not_extend_the_conversation(monkeypatch):
    args = make_args(sessions=1, turns=3, concurrency=1, local_session_fraction=1.0)
    sessions = bench.build_sessions(args, "SYSTEM", ["CTX"])

    async def fake_send_chat(http, url, payload, timeout):
        return bench.ChatResult(ok=False, error="HTTP 500")

    monkeypatch.setattr(bench, "send_chat", fake_send_chat)
    runner = bench.Runner(args, sessions, ["ASK"])
    asyncio.run(runner.run())

    assert runner.stats[bench.LOCAL_TIER].failures == 1
    assert sessions[0].turns_done == 0


def test_user_messages_are_distinct_and_reproducible():
    texts = bench.build_distinct_texts(FakeTokenizer(), 4, 16, seed=3, label="ask")

    assert len(set(texts)) == 4
    assert texts == bench.build_distinct_texts(FakeTokenizer(), 4, 16, seed=3, label="ask")
