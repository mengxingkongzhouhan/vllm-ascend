# vLLM Ascend Benchmarks

## Introduction

This document outlines the benchmarking methodology for vllm-ascend, aimed at evaluating the performance under a variety of workloads. The primary goal is to help developers assess whether their pull requests improve or degrade vllm-ascend's performance.

## Overview

**Benchmarking Coverage**: We measure latency, throughput, and fixed-QPS serving on the Atlas800I A2 (see [quick_start](../docs/source/quick_start.md) to learn more supported devices list), with different models(coming soon).

- Latency tests
    - Input length: 32 tokens.
    - Output length: 128 tokens.
    - Batch size: fixed (8).
    - Models: Qwen2.5-7B-Instruct, Qwen3-8B.
    - Evaluation metrics: end-to-end latency (mean, median, p99).

- Throughput tests
    - Input length: randomly sample 200 prompts from ShareGPT dataset (with fixed random seed).
    - Output length: the corresponding output length of these 200 prompts.
    - Batch size: dynamically determined by vllm to achieve maximum throughput.
    - Models: Qwen2.5-VL-7B-Instruct, Qwen2.5-7B-Instruct, Qwen3-8B.
    - Evaluation metrics: throughput.
- Serving tests
    - Input length: randomly sample 200 prompts from ShareGPT dataset (with fixed random seed).
    - Output length: the corresponding output length of these 200 prompts.
    - Batch size: dynamically determined by vllm and the arrival pattern of the requests.
    - **Average QPS (query per second)**: 1, 4, 16 and inf. QPS = inf means all requests come at once. For other QPS values, the arrival time of each query is determined using a random Poisson process (with fixed random seed).
    - Models: Qwen2.5-VL-7B-Instruct, Qwen2.5-7B-Instruct, Qwen3-8B.
    - Evaluation metrics: throughput, TTFT (time to the first token, with mean, median and p99), ITL (inter-token latency, with mean, median and p99).

**Benchmarking Duration**: about 800 seconds for single model.

## Quick Use

### Prerequisites

Before running the benchmarks, ensure the following:

- vllm and vllm-ascend are installed and properly set up in an NPU environment, as these scripts are specifically designed for NPU devices.

- Install necessary dependencies for benchmarks:
  
  ```shell
  pip install -r benchmarks/requirements-bench.txt
  ```
  
- For performance benchmark, it is recommended to set the [load-format](https://github.com/vllm-project/vllm-ascend/blob/5897dc5bbe321ca90c26225d0d70bff24061d04b/benchmarks/tests/latency-tests.json#L7) as `dummy`, It will construct random weights based on the passed model without downloading the weights from internet, which can greatly reduce the benchmark time.
- If you want to run a customized benchmark, feel free to add your own models and parameters in the [JSON](https://github.com/vllm-project/vllm-ascend/tree/main/benchmarks/tests), let's take `Qwen2.5-VL-7B-Instruct`as an example:

  ```json
  [
  {
    "test_name": "serving_qwen2_5vl_7B_tp1",
    "qps_list": [
      1,
      4,
      16,
      "inf"
    ],
    "server_parameters": {
      "model": "Qwen/Qwen2.5-VL-7B-Instruct",
      "tensor_parallel_size": 1,
      "swap_space": 16,
      "disable_log_stats": "",
      "disable_log_requests": "",
      "trust_remote_code": "",
      "max_model_len": 16384
    },
    "client_parameters": {
      "model": "Qwen/Qwen2.5-VL-7B-Instruct",
      "backend": "openai-chat",
      "dataset_name": "hf",
      "hf_split": "train",
      "endpoint": "/v1/chat/completions",
      "dataset_path": "lmarena-ai/vision-arena-bench-v0.1",
      "num_prompts": 200
    }
  }
  ]
  ```
  
this Json will be structured and parsed into server parameters and client parameters by the benchmark script. This configuration defines a test case named `serving_qwen2_5vl_7B_tp1`, designed to evaluate the performance of the `Qwen/Qwen2.5-VL-7B-Instruct` model under different request rates. The test includes both server and client parameters, for more parameters details, see vllm benchmark [cli](https://github.com/vllm-project/vllm/tree/main/vllm/benchmarks).

- **Test Overview**
    - Test Name: serving_qwen2_5vl_7B_tp1

    - Queries Per Second (QPS): The test is run at four different QPS levels: 1, 4, 16, and inf (infinite load, typically used for stress testing).

- Server Parameters
    - Model: Qwen/Qwen2.5-VL-7B-Instruct

    - Tensor Parallelism: 1 (no model parallelism is used; the model runs on a single device or node)

    - Swap Space: 16 GB (used to handle memory overflow by swapping to disk)

    - disable_log_stats: disables logging of performance statistics.

    - disable_log_requests: disables logging of individual requests.

    - Trust Remote Code: enabled (allows execution of model-specific custom code)

    - Max Model Length: 16,384 tokens (maximum context length supported by the model)

- Client Parameters

    - Model: Qwen/Qwen2.5-VL-7B-Instruct (same as the server)

    - Backend: openai-chat (suggests the client uses the OpenAI-compatible chat API format)

    - Dataset Source: Hugging Face (hf)

    - Dataset Split: train

    - Endpoint: /v1/chat/completions (the REST API endpoint to which chat requests are sent)

    - Dataset Path: lmarena-ai/vision-arena-bench-v0.1 (the benchmark dataset used for evaluation, hosted on Hugging Face)

    - Number of Prompts: 200 (the total number of prompts used during the test)

### Run benchmarks

#### Use benchmark script

The provided scripts automatically execute performance tests for serving, throughput, and latency. To start the benchmarking process, run command in the vllm-ascend root directory:

```shell
bash benchmarks/scripts/run-performance-benchmarks.sh
```

Once the script completes, you can find the results in the benchmarks/results folder. The output files may resemble the following:

```shell
.
|-- serving_qwen2_5_7Bvl_tp1_qps_1.json
|-- serving_qwen2_5_7Bvl_tp1_qps_16.json
|-- serving_qwen2_5_7Bvl_tp1_qps_4.json
|-- serving_qwen2_5_7Bvl_tp1_qps_inf.json
|-- throughput_qwen2_5_7Bvl_tp1.json
```

These files contain detailed benchmarking results for further analysis.

#### Use benchmark cli

For more flexible and customized use, benchmark cli is also provided to run online/offline benchmarks
Similarly, let's take `Qwen2.5-VL-7B-Instruct` benchmark as an example:

##### Online serving

1. Launch the server:

    ```shell
    vllm serve Qwen2.5-VL-7B-Instruct --max-model-len 16789
    ```

2. Running performance tests using cli
  
    ```shell
    vllm bench serve --model Qwen2.5-VL-7B-Instruct\
    --endpoint-type "openai-chat" --dataset-name hf \
    --hf-split train --endpoint "/v1/chat/completions" \
    --dataset-path "lmarena-ai/vision-arena-bench-v0.1" \
    --num-prompts 200 \
    --request-rate 16
    ```

##### Offline

- **Throughput**

  ```shell
  vllm bench throughput --output-json results/throughput_qwen2_5_7B_tp1.json \
  --model Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 1 --load-format dummy \
  --dataset-path /github/home/.cache/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 200 --backend vllm
  ```

- **Latency**
  
  ```shell
  vllm bench latency --output-json results/latency_qwen2_5_7B_tp1.json \
  --model Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 1 \
  --load-format dummy --num-iters-warmup 5 --num-iters 15
  ```

### Exercising the local prefix cache and the KV pool together

The local prefix cache is consulted before the KV pool, and the pool is only credited with the tokens it adds on top. A workload built from one set of repeated prefixes therefore drives one hit rate or the other but not both: whatever the local cache holds, the pool contributes nothing for. `vllm bench serve --dataset-name prefix_repetition` offers exactly one such set, which is why it commonly reports a high `Prefix cache hit rate` next to `External prefix cache hit rate: 0.0%`.

`benchmarks/scripts/two_tier_prefix_bench.py` drives a two-tier workload instead. A small **hot** set of prefixes is re-requested often enough to stay resident locally and produces the local hit rate, while a much larger **warm** set is cycled round-robin so each prefix is evicted before its next use and has to be re-read from the pool. It also seeds the pool before measuring, because a pool starts empty and the first use of a prefix can only be a miss — the usual reason a first run shows no external hits while a second run does.

Before sending anything, the script checks the plan against the KV cache budget and refuses combinations that cannot work, since a long prefix often leaves room for only one or two of them and no request ordering can fix that. Pass the per-engine `GPU KV cache size` printed at startup, the number of prefill engines, and whether the router pins a prefix to one engine:

```shell
python benchmarks/scripts/two_tier_prefix_bench.py \
  --host $BENCH_HOST --port $BENCH_PORT \
  --model qwen3 --tokenizer $TOKENIZER_PATH \
  --kv-cache-tokens 64947 --engines 16 --sticky-routing \
  --prefix-len 8192 --suffix-len 128 --output-len 128 \
  --hot-prefixes 16 --warm-prefixes 384 --hot-fraction 0.5 \
  --num-prompts 8000 --concurrency 16
```

The capacity plan it prints first shows how the budget is spent, and each rejection names the argument to change:

```text
Capacity plan (per engine)
  prompt                : 8,320 tokens (65 blocks)
  KV cache              : 64,947 tokens
  free for prefixes     : 56,627 tokens = 6.91 prefixes
  hot prefixes / engine : 1.00 (expected to stay resident)
  warm prefixes / engine: 24.00 for 5.91 free slots
```

Read both rates from the server log over the measured phase only. `Prefix cache hit rate` is driven by the hot tier and `External prefix cache hit rate` by the warm tier; warm requests should show a non-zero `need to load` in the `kvpool hit tokens ... local prefix cache hit tokens ... need to load` lines. See [FAQ 29](../docs/source/faqs.md) for how to interpret a rate that stays at zero.
