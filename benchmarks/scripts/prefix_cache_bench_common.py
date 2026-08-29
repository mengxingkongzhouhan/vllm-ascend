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
"""Helpers shared by the prefix cache benchmark drivers in this directory."""

from __future__ import annotations

import json
import random
import statistics
import time
from dataclasses import dataclass, field

import aiohttp

# Percentiles reported for latency metrics.
REPORTED_PERCENTILES = (50, 99)


def align_up(value: int, multiple: int) -> int:
    """Round up to a whole number of KV cache blocks."""
    return -(-value // multiple) * multiple


def build_distinct_texts(tokenizer, count: int, num_tokens: int, seed: int, label: str = "text") -> list[str]:
    """Build `count` distinct texts, each close to `num_tokens` tokens long.

    Exactness is best effort. What matters for cache hits is that every request
    reusing a text sends a byte-identical string, which holds because each text
    is built once here and reused verbatim.
    """
    rng = random.Random(seed)
    texts = []
    for index in range(count):
        words = [f"{label}{index:06d}"] + [str(rng.randint(100000, 999999)) for _ in range(num_tokens)]
        token_ids = tokenizer.encode(" ".join(words), add_special_tokens=False)
        while len(token_ids) < num_tokens:
            words.append(str(rng.randint(100000, 999999)))
            token_ids = tokenizer.encode(" ".join(words), add_special_tokens=False)
        texts.append(tokenizer.decode(token_ids[:num_tokens]))
    if len(set(texts)) != count:
        raise RuntimeError(
            f"built {len(set(texts))} distinct {label}s out of {count} requested. They must all differ, "
            "otherwise requests share content and the workload collapses into one case."
        )
    return texts


@dataclass
class ChatResult:
    ok: bool
    ttft: float = 0.0
    latency: float = 0.0
    text: str = ""
    error: str = ""


async def send_chat(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    timeout: float,
) -> ChatResult:
    """Post one streaming chat completion, returning TTFT and the full reply.

    The reply text is returned because a multi-turn workload has to append the
    model's actual output to the next turn: only then does the next prompt carry
    the previous one as a byte-identical prefix.
    """
    started = time.perf_counter()
    ttft = 0.0
    pieces: list[str] = []
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            if response.status != 200:
                body = await response.text()
                return ChatResult(ok=False, error=f"HTTP {response.status}: {body[:200]}")
            async for raw_line in response.content:
                line = raw_line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[len(b"data:") :].strip()
                if data == b"[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0].get("delta", {})
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                piece = delta.get("content")
                if piece:
                    if not ttft:
                        ttft = time.perf_counter() - started
                    pieces.append(piece)
    except (aiohttp.ClientError, TimeoutError) as error:
        # asyncio.TimeoutError is an alias of TimeoutError from Python 3.11.
        return ChatResult(ok=False, error=f"{type(error).__name__}: {error}")
    return ChatResult(ok=True, ttft=ttft, latency=time.perf_counter() - started, text="".join(pieces))


@dataclass
class LatencyStats:
    """Latency samples for one labelled group of requests."""

    label: str
    ttfts: list[float] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)
    failures: int = 0

    def add(self, result: ChatResult) -> None:
        if not result.ok:
            self.failures += 1
            return
        if result.ttft:
            self.ttfts.append(result.ttft)
        self.latencies.append(result.latency)


def format_percentiles(values: list[float]) -> str:
    if not values:
        return "no sample"
    if len(values) == 1:
        return f"mean={values[0]:.3f}s"
    quantiles = statistics.quantiles(values, n=100)
    tail = ", ".join(f"p{percentile}={quantiles[percentile - 1]:.3f}s" for percentile in REPORTED_PERCENTILES)
    return f"mean={statistics.fmean(values):.3f}s, {tail}"


def report_group(stats: LatencyStats) -> None:
    total = len(stats.latencies) + stats.failures
    print(f"  {stats.label}: {len(stats.latencies)}/{total} ok")
    if stats.failures:
        print(f"    failures : {stats.failures}")
    if stats.latencies:
        print(f"    TTFT     : {format_percentiles(stats.ttfts)}")
        print(f"    latency  : {format_percentiles(stats.latencies)}")
