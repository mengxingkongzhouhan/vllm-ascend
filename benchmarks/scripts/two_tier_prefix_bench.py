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
"""Drive a workload that keeps both prefix cache hit rates high at once.

The local prefix cache is consulted before the KV pool, and the pool is only
credited with what it adds on top. A workload therefore cannot push both hit
rates up with a single tier of prefixes: whatever the local cache holds, the
pool contributes nothing for, and whatever the local cache never holds gives no
local hits. ``vllm bench serve --dataset-name prefix_repetition`` offers one
tier, which is why it reports one rate or the other but not both.

This script uses two tiers instead:

* **hot** prefixes are re-requested often enough to stay resident in every
  engine's local cache, and produce the local hit rate;
* **warm** prefixes are cycled round-robin through a set far larger than the
  cache, so each one is evicted before its next use and has to be re-read from
  the pool, producing the external hit rate.

It also seeds the pool before measuring. A pool starts empty, so the first use
of a prefix can only ever be a miss; without a seeding pass the measured window
mixes cold misses with real hits, which is why a first run can show no external
hits at all while a second run does.

Before sending anything it checks the plan against the KV cache budget and
refuses combinations that cannot work, since with long prefixes a cache often
holds only one or two of them and no request ordering can fix that.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prefix_cache_bench_common import (
    LatencyStats,  # noqa: E402
    align_up,  # noqa: E402
    build_distinct_texts,  # noqa: E402
    report_group,  # noqa: E402
    send_chat,  # noqa: E402
)

# Requests whose prompt is a large fraction of the cache leave little room for
# cached prefixes, so the planner insists on at least this many free prefix
# slots beyond the resident hot set for the warm tier to stream through.
MIN_WARM_SLOTS = 1

# How many times larger the per-engine warm set must be than the slots it can
# occupy, so that a warm prefix is reliably evicted before its next use.
WARM_OVERFLOW_FACTOR = 3


@dataclass
class Plan:
    """Per-engine KV cache budget for the requested workload."""

    request_tokens: int
    prefix_tokens: int
    cache_budget_tokens: int
    prefix_slots: float
    hot_per_engine: float
    warm_per_engine: float
    warm_slots: float
    problems: list[str] = field(default_factory=list)


def plan_capacity(args: argparse.Namespace) -> Plan:
    """Work out whether the two tiers can coexist in one engine's cache."""
    prefix_tokens = align_up(args.prefix_len, args.block_size)
    request_tokens = align_up(args.prefix_len + args.suffix_len, args.block_size)

    # Running requests allocate from the same cache as the cached prefixes, so
    # only what is left over can hold prefixes.
    running_reserve = args.concurrency / args.engines * request_tokens
    cache_budget = args.kv_cache_tokens - running_reserve
    prefix_slots = cache_budget / prefix_tokens

    # With sticky routing a prefix only ever reaches one engine, so each engine
    # sees its share; otherwise every engine eventually sees all of them.
    divisor = args.engines if args.sticky_routing else 1
    hot_per_engine = args.hot_prefixes / divisor
    warm_per_engine = args.warm_prefixes / divisor
    warm_slots = prefix_slots - hot_per_engine

    problems = []
    if prefix_slots < 1:
        problems.append(
            f"one engine has room for {prefix_slots:.2f} prefixes while {args.concurrency} concurrent "
            f"requests are in flight, so no prefix can stay cached. Lower --concurrency, lower "
            f"--prefix-len, or give the engine more KV cache."
        )
    elif warm_slots < MIN_WARM_SLOTS:
        problems.append(
            f"the hot set needs {hot_per_engine:.2f} of the {prefix_slots:.2f} prefix slots per engine, "
            f"leaving {warm_slots:.2f} for the warm tier. Reduce --hot-prefixes or --prefix-len so at "
            f"least {MIN_WARM_SLOTS} slot is free, otherwise hot and warm prefixes evict each other and "
            f"both hit rates collapse."
        )
    elif warm_per_engine < WARM_OVERFLOW_FACTOR * warm_slots:
        needed = int(WARM_OVERFLOW_FACTOR * warm_slots * divisor) + 1
        problems.append(
            f"each engine sees {warm_per_engine:.2f} warm prefixes but has {warm_slots:.2f} free slots, "
            f"so warm prefixes survive until their next use and the pool is never read. Raise "
            f"--warm-prefixes to at least {needed}."
        )

    # The seeding pass stores every warm prefix, so one use each in the measured
    # phase already reads the pool; but a phase that does not get through the
    # whole warm set reports a rate over an unrepresentative subset.
    warm_requests = args.num_prompts * (1 - args.hot_fraction)
    if args.warm_prefixes and warm_requests < args.warm_prefixes:
        needed = int(args.warm_prefixes / max(1 - args.hot_fraction, 1e-9)) + 1
        problems.append(
            f"the measured phase sends {warm_requests:.0f} warm requests for {args.warm_prefixes} warm "
            f"prefixes, so most are never exercised. Raise --num-prompts to at least {needed}."
        )

    return Plan(
        request_tokens=request_tokens,
        prefix_tokens=prefix_tokens,
        cache_budget_tokens=int(cache_budget),
        prefix_slots=prefix_slots,
        hot_per_engine=hot_per_engine,
        warm_per_engine=warm_per_engine,
        warm_slots=warm_slots,
        problems=problems,
    )


def report_plan(plan: Plan, args: argparse.Namespace) -> None:
    print("Capacity plan (per engine)")
    print(f"  prompt                : {plan.request_tokens:,} tokens ({plan.request_tokens // args.block_size} blocks)")
    print(f"  KV cache              : {args.kv_cache_tokens:,} tokens")
    print(f"  free for prefixes     : {plan.cache_budget_tokens:,} tokens = {plan.prefix_slots:.2f} prefixes")
    print(f"  hot prefixes / engine : {plan.hot_per_engine:.2f} (expected to stay resident)")
    print(f"  warm prefixes / engine: {plan.warm_per_engine:.2f} for {plan.warm_slots:.2f} free slots")
    for problem in plan.problems:
        print(f"  PROBLEM: {problem}")


@dataclass
class Request:
    tier: str
    prefix: str
    suffix: str


def build_sequence(hot: list[str], warm: list[str], suffixes: list[str], args: argparse.Namespace) -> list[Request]:
    """Interleave the tiers, cycling warm prefixes so reuse distance stays maximal.

    The warm tier is walked round-robin rather than sampled, which makes the
    distance between two uses of a warm prefix exactly the size of the warm set
    and therefore as likely as possible to have been evicted in between.
    """
    rng = random.Random(args.seed)
    sequence: list[Request] = []
    warm_cursor = 0
    for index in range(args.num_prompts):
        suffix = suffixes[index % len(suffixes)]
        if warm and rng.random() >= args.hot_fraction:
            sequence.append(Request("warm", warm[warm_cursor % len(warm)], suffix))
            warm_cursor += 1
        else:
            sequence.append(Request("hot", rng.choice(hot), suffix))
    return sequence


async def run_phase(name: str, requests: list[Request], args: argparse.Namespace) -> None:
    """Send one phase of the workload under a fixed in-flight limit."""
    url = f"http://{args.host}:{args.port}{args.endpoint}"
    semaphore = asyncio.Semaphore(args.concurrency)
    print(f"\nPhase '{name}': {len(requests)} requests at concurrency {args.concurrency}")
    stats = {tier: LatencyStats(f"{tier} requests") for tier in ("hot", "warm")}

    async def worker(request: Request) -> None:
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": f"{request.prefix} {request.suffix}"}],
            "max_tokens": args.output_len,
            "ignore_eos": True,
            "stream": True,
            "temperature": 0.0,
        }
        async with semaphore:
            stats[request.tier].add(await send_chat(session, url, payload, args.request_timeout))

    started = time.perf_counter()
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        await asyncio.gather(*(worker(request) for request in requests))
    elapsed = time.perf_counter() - started

    done = sum(len(group.latencies) for group in stats.values())
    print(f"  completed {done}/{len(requests)} in {elapsed:.1f} s ({done / max(elapsed, 1e-9):.2f} req/s)")
    for tier in ("hot", "warm"):
        if stats[tier].latencies or stats[tier].failures:
            report_group(stats[tier])


def save_dataset(path: str, sequence: list[Request]) -> None:
    with open(path, "w") as file:
        for request in sequence:
            file.write(json.dumps({"prompt": f"{request.prefix} {request.suffix}"}) + "\n")
    print(f"Wrote {len(sequence)} prompts to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--endpoint", default="/v1/chat/completions")
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--tokenizer", required=True, help="path or id of the tokenizer, for exact prefix lengths")

    parser.add_argument("--prefix-len", type=int, default=8192, help="tokens per shared prefix")
    parser.add_argument("--suffix-len", type=int, default=128, help="unique tokens appended per request")
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--num-prompts", type=int, default=2000, help="requests in the measured phase")
    # Defaults suit a single engine; the capacity plan reports what to raise
    # them to once --engines and --sticky-routing spread the prefixes out.
    parser.add_argument("--hot-prefixes", type=int, default=4, help="prefixes that should stay locally cached")
    parser.add_argument("--warm-prefixes", type=int, default=64, help="prefixes that should be served by the pool")
    parser.add_argument("--hot-fraction", type=float, default=0.5, help="share of requests drawn from the hot tier")
    parser.add_argument("--num-suffixes", type=int, default=64, help="distinct suffixes cycled across requests")

    parser.add_argument("--concurrency", type=int, default=16, help="requests in flight across the cluster")
    parser.add_argument("--engines", type=int, default=1, help="prefill engines the requests are spread over")
    parser.add_argument(
        "--kv-cache-tokens", type=int, required=True, help="per-engine 'GPU KV cache size' from startup"
    )
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--sticky-routing", action="store_true", help="a prefix always reaches the same engine")

    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--save-dataset", help="also write the measured phase prompts as jsonl")
    parser.add_argument("--skip-seeding", action="store_true", help="pool is already populated from an earlier run")
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
    hot = build_distinct_texts(tokenizer, args.hot_prefixes, args.prefix_len, args.seed, "hot")
    warm = build_distinct_texts(tokenizer, args.warm_prefixes, args.prefix_len, args.seed + 1, "warm")
    suffixes = build_distinct_texts(tokenizer, args.num_suffixes, args.suffix_len, args.seed + 2, "ask")

    if not args.skip_seeding:
        # Every prefix is stored on first use, so send each one once before
        # measuring. Skipping this mixes unavoidable cold misses into the
        # measured window and can hide the external hits entirely.
        seeding = [Request("warm", prefix, suffixes[0]) for prefix in warm]
        seeding += [Request("hot", prefix, suffixes[0]) for prefix in hot]
        await run_phase("seed pool and local caches", seeding, args)

    sequence = build_sequence(hot, warm, suffixes, args)
    hot_count = sum(1 for request in sequence if request.tier == "hot")
    print(f"\nMeasured phase mix: {hot_count} hot, {len(sequence) - hot_count} warm")
    if args.save_dataset:
        save_dataset(args.save_dataset, sequence)
    await run_phase("measure", sequence, args)

    print(
        "\nRead the two hit rates from the server log over this phase only:\n"
        "  'Prefix cache hit rate'          -> driven by the hot tier\n"
        "  'External prefix cache hit rate' -> driven by the warm tier\n"
        "  'kvpool hit tokens ... local prefix cache hit tokens ... need to load'\n"
        "     warm requests should show a non-zero 'need to load'."
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run(parse_args())))
