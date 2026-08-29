#!/usr/bin/env python3
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#
"""Agent-style multi-turn workload for prefix cache and KV pool measurement.

A multi-turn conversation is the natural shape for exercising both cache tiers,
because every turn after the first re-sends the whole conversation so far. Where
that prefix is found depends only on how many other sessions were served in
between:

* a session whose next turn follows immediately finds its context still in the
  engine's local cache, and contributes to the local prefix cache hit rate;
* a session that waits behind every other active session has its context
  evicted in the meantime and has to read it back from the KV pool, which is
  what a shared pool exists for.

The workload therefore splits sessions into two tiers and controls the gap
directly, rather than hoping a think-time delay produces the right one. Three
levels of reuse result, matching a real agent deployment:

1. a system prompt and tool schema shared by every session, always locally hot;
2. each session's own growing conversation, local or pooled by tier;
3. the new user message of each turn, always a miss.

Assistant replies are fed back into the next turn verbatim, so the next prompt
really does carry the previous one as a byte-identical prefix, exactly as an
agent loop would.

Latency is reported per tier and per turn index, since the point of the caches
is the effect on TTFT: locally hit, pool hit, and cold turns should separate
clearly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prefix_cache_bench_common import (
    ChatResult,  # noqa: E402
    LatencyStats,  # noqa: E402
    align_up,  # noqa: E402
    build_distinct_texts,  # noqa: E402
    report_group,  # noqa: E402
    send_chat,  # noqa: E402
)

# How many times more pooled sessions than fit in the cache, so that a pooled
# session's context is reliably gone by the time its next turn arrives.
POOLED_OVERFLOW_FACTOR = 3

LOCAL_TIER = "local"
POOLED_TIER = "pooled"


@dataclass
class Session:
    session_id: int
    tier: str
    messages: list[dict] = field(default_factory=list)
    turns_done: int = 0


@dataclass
class Plan:
    """Per-engine KV cache budget for the requested conversation shape."""

    first_turn_tokens: int
    final_turn_tokens: int
    cache_budget_tokens: int
    resident_sessions: float
    local_per_engine: float
    pooled_per_engine: float
    problems: list[str] = field(default_factory=list)


def turn_prompt_tokens(args: argparse.Namespace, turn_index: int) -> int:
    """Prompt length at a given turn, as the conversation accumulates."""
    grown = turn_index * (args.user_len + args.output_len)
    return args.system_len + args.session_context_len + args.user_len + grown


def plan_capacity(args: argparse.Namespace) -> Plan:
    """Check that local sessions can stay resident while pooled ones cannot."""
    first_turn = align_up(turn_prompt_tokens(args, 0), args.block_size)
    final_turn = align_up(turn_prompt_tokens(args, args.turns - 1), args.block_size)

    # A session's footprint keeps growing, so size it on the last turn.
    running_reserve = args.concurrency / args.engines * final_turn
    cache_budget = args.kv_cache_tokens - running_reserve
    resident_sessions = cache_budget / final_turn if final_turn else 0.0

    local_sessions = round(args.sessions * args.local_session_fraction)
    pooled_sessions = args.sessions - local_sessions
    divisor = args.engines if args.sticky_routing else 1
    local_per_engine = local_sessions / divisor
    pooled_per_engine = pooled_sessions / divisor

    # A local session is requeued at the front, so its next turn arrives after
    # only the other in-flight turns. What has to stay resident for it to hit
    # locally is that working set, not the whole local tier.
    local_active = min(local_per_engine, args.concurrency / args.engines)

    problems = []
    if final_turn > args.max_model_len:
        problems.append(
            f"the last turn reaches {final_turn:,} tokens, past --max-model-len {args.max_model_len:,}. "
            f"Reduce --turns, --user-len, --output-len or --session-context-len."
        )
    if resident_sessions < 1:
        problems.append(
            f"one engine has room for {resident_sessions:.2f} conversations of {final_turn:,} tokens while "
            f"{args.concurrency} requests are in flight, so not even a local session can stay cached. "
            f"Lower --concurrency or --turns, or give the engine more KV cache."
        )
    elif local_active > resident_sessions:
        problems.append(
            f"{local_active:.2f} conversations per engine are in flight but only {resident_sessions:.2f} fit, "
            f"so they evict each other instead of hitting locally. Lower --concurrency or --turns."
        )
    elif pooled_per_engine < POOLED_OVERFLOW_FACTOR * resident_sessions:
        needed = int(POOLED_OVERFLOW_FACTOR * resident_sessions * divisor / max(1 - args.local_session_fraction, 1e-9))
        problems.append(
            f"{pooled_per_engine:.2f} pooled sessions per engine fit in {resident_sessions:.2f} slots, so their "
            f"context survives until the next turn and the pool is never read. Raise --sessions to at "
            f"least {needed}."
        )

    return Plan(
        first_turn_tokens=first_turn,
        final_turn_tokens=final_turn,
        cache_budget_tokens=int(cache_budget),
        resident_sessions=resident_sessions,
        local_per_engine=local_per_engine,
        pooled_per_engine=pooled_per_engine,
        problems=problems,
    )


def report_plan(plan: Plan, args: argparse.Namespace) -> None:
    rows = [
        ("turn 1 prompt", f"{plan.first_turn_tokens:,} tokens"),
        (f"turn {args.turns} prompt", f"{plan.final_turn_tokens:,} tokens"),
        ("KV cache", f"{args.kv_cache_tokens:,} tokens"),
        ("free for conversations", f"{plan.cache_budget_tokens:,} tokens = {plan.resident_sessions:.2f} sessions"),
        ("local sessions / engine", f"{plan.local_per_engine:.2f} (expected to stay resident)"),
        ("pooled sessions / engine", f"{plan.pooled_per_engine:.2f} (expected to be evicted between turns)"),
    ]
    width = max(len(label) for label, _ in rows)
    print("Capacity plan (per engine)")
    for label, value in rows:
        print(f"  {label:<{width}} : {value}")
    for problem in plan.problems:
        print(f"  PROBLEM: {problem}")


def build_sessions(args: argparse.Namespace, system_prompt: str, contexts: list[str]) -> list[Session]:
    """Create sessions, each opening with the shared preamble and its own context."""
    local_sessions = round(args.sessions * args.local_session_fraction)
    sessions = []
    for index in range(args.sessions):
        tier = LOCAL_TIER if index < local_sessions else POOLED_TIER
        session = Session(session_id=index, tier=tier)
        session.messages.append({"role": "system", "content": system_prompt})
        session.messages.append({"role": "user", "content": contexts[index]})
        sessions.append(session)
    return sessions


class Runner:
    """Serves sessions turn by turn, controlling each tier's reuse distance."""

    def __init__(self, args: argparse.Namespace, sessions: list[Session], user_messages: list[str]):
        self.args = args
        self.user_messages = user_messages
        self.url = f"http://{args.host}:{args.port}{args.endpoint}"
        self.queue: deque[Session] = deque(sessions)
        self.lock = asyncio.Lock()
        self.stats = {
            LOCAL_TIER: LatencyStats(f"{LOCAL_TIER} sessions"),
            POOLED_TIER: LatencyStats(f"{POOLED_TIER} sessions"),
        }
        self.per_turn: dict[int, LatencyStats] = {}
        self.transcript: list[dict] = []

    async def _next_session(self) -> Session | None:
        async with self.lock:
            return self.queue.popleft() if self.queue else None

    async def _requeue(self, session: Session) -> None:
        """Put a session back where its tier's reuse distance requires.

        Front means the next turn is served almost immediately and finds the
        context locally. Back means every other active session is served first,
        which is what evicts the context and forces a pool read.
        """
        async with self.lock:
            if session.tier == LOCAL_TIER:
                self.queue.appendleft(session)
            else:
                self.queue.append(session)

    async def _serve_turn(self, http: aiohttp.ClientSession, session: Session) -> None:
        turn_index = session.turns_done
        payload = {
            "model": self.args.model,
            "messages": session.messages,
            "max_tokens": self.args.output_len,
            "ignore_eos": True,
            "stream": True,
            "temperature": 0.0,
        }
        result: ChatResult = await send_chat(http, self.url, payload, self.args.request_timeout)

        self.stats[session.tier].add(result)
        self.per_turn.setdefault(turn_index, LatencyStats(f"turn {turn_index + 1}")).add(result)
        if not result.ok:
            print(f"  session {session.session_id} turn {turn_index + 1} failed: {result.error}")
            return

        if self.args.save_dataset:
            self.transcript.append(
                {
                    "session": session.session_id,
                    "tier": session.tier,
                    "turn": turn_index + 1,
                    "messages": session.messages,
                }
            )
        # Feeding the real reply back is what makes the next prompt an exact
        # extension of this one, so the server can match it as a prefix.
        session.messages = session.messages + [{"role": "assistant", "content": result.text}]
        session.turns_done += 1
        if session.turns_done < self.args.turns:
            session.messages.append(
                {"role": "user", "content": self.user_messages[turn_index % len(self.user_messages)]}
            )
            await self._requeue(session)

    async def _worker(self, http: aiohttp.ClientSession) -> None:
        while True:
            session = await self._next_session()
            if session is None:
                return
            await self._serve_turn(http, session)

    async def run(self) -> None:
        connector = aiohttp.TCPConnector(limit=0)
        async with aiohttp.ClientSession(connector=connector) as http:
            await asyncio.gather(*(self._worker(http) for _ in range(self.args.concurrency)))

    def report(self) -> None:
        print("\nBy tier")
        for tier in (LOCAL_TIER, POOLED_TIER):
            report_group(self.stats[tier])
        print("\nBy turn index (turn 1 is cold, later turns should be faster)")
        for turn_index in sorted(self.per_turn):
            report_group(self.per_turn[turn_index])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--endpoint", default="/v1/chat/completions")
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--tokenizer", required=True, help="path or id of the tokenizer, for exact lengths")

    parser.add_argument("--system-len", type=int, default=1024, help="shared system and tool preamble tokens")
    parser.add_argument("--session-context-len", type=int, default=4096, help="per-session opening context tokens")
    parser.add_argument("--user-len", type=int, default=256, help="tokens of each follow-up user message")
    parser.add_argument("--output-len", type=int, default=256, help="tokens the model generates per turn")
    parser.add_argument("--turns", type=int, default=6, help="turns per session")
    # Defaults suit a single engine at low concurrency; the capacity plan reports
    # what to raise them to once --engines and --sticky-routing spread the
    # sessions out.
    parser.add_argument("--sessions", type=int, default=64, help="total conversations")
    parser.add_argument(
        "--local-session-fraction",
        type=float,
        default=0.25,
        help="share of sessions served back to back, so they hit the local cache",
    )
    parser.add_argument("--num-user-messages", type=int, default=32, help="distinct follow-up messages cycled")

    parser.add_argument("--concurrency", type=int, default=4, help="turns in flight across the cluster")
    parser.add_argument("--engines", type=int, default=1, help="prefill engines the sessions are spread over")
    parser.add_argument(
        "--kv-cache-tokens", type=int, required=True, help="per-engine 'GPU KV cache size' from the startup log"
    )
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--sticky-routing", action="store_true", help="a session always reaches the same engine")

    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--save-dataset", help="write the conversations as jsonl")
    parser.add_argument("--force", action="store_true", help="run even if the capacity plan reports problems")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    plan = plan_capacity(args)
    report_plan(plan, args)
    if plan.problems and not args.force:
        print("\nRefusing to run: the plan above cannot produce both hit rates. Pass --force to run anyway.")
        return 1

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    # One shared preamble for every session, so it is always locally hot.
    system_prompt = build_distinct_texts(tokenizer, 1, args.system_len, args.seed, "system")[0]
    contexts = build_distinct_texts(tokenizer, args.sessions, args.session_context_len, args.seed + 1, "context")
    user_messages = build_distinct_texts(tokenizer, args.num_user_messages, args.user_len, args.seed + 2, "ask")

    sessions = build_sessions(args, system_prompt, contexts)
    local = sum(1 for session in sessions if session.tier == LOCAL_TIER)
    print(f"\n{len(sessions)} sessions ({local} local, {len(sessions) - local} pooled) x {args.turns} turns")

    runner = Runner(args, sessions, user_messages)
    await runner.run()
    runner.report()

    if args.save_dataset:
        with open(args.save_dataset, "w") as file:
            for record in runner.transcript:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(runner.transcript)} turns to {args.save_dataset}")

    print(
        "\nRead the hit rates from the server log over this run:\n"
        "  'Prefix cache hit rate'          -> shared preamble plus the local tier\n"
        "  'External prefix cache hit rate' -> the pooled tier\n"
        "  'kvpool hit tokens ... local prefix cache hit tokens ... need to load'\n"
        "     pooled sessions should show a non-zero 'need to load' from turn 2 on."
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run(parse_args())))
