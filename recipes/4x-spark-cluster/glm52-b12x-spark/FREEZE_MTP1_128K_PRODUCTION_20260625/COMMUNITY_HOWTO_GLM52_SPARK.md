# Running GLM-5.2 NVFP4 MTP at 128K on 4x DGX Spark

This is a reproducible field note for serving GLM-5.2 on four NVIDIA DGX Spark nodes. The result is a working 128K-context OpenAI-compatible vLLM endpoint using a Spark-specific port of the B12X / dark-devotion GLM stack.

## Result

Validated baseline:

```text
Model: GLM-5.2 NVFP4 MTP hybrid
Hardware: 4x DGX Spark, one GB10 GPU per host
GPU arch: SM121
Executor: Ray-backed multi-node vLLM
Parallelism: TP4 / PP1 / DCP4
Speculation: MTP1
Context: 131,072 tokens
KV: fp8
Attention: B12X sparse MLA
MoE: flashinfer_cutlass
CUDA graphs: enabled, max capture size 4
Endpoint: OpenAI-compatible /v1
```

Measured batch-one codegen decode:

```text
no-MTP hot:              10.427 tok/s
MTP3 hot:                12.698 tok/s
MTP1 hot:                15.176 tok/s
MTP1 baked-image hot:    14.460 tok/s
MTP1 audit run:          14.471 tok/s, speculative acceptance 0.747
```

MTP1 is the current best-known serving point for this 128K Spark setup.

## Why Spark needs its own recipe

The upstream RTX PRO 6000-style recipe assumes a different topology: many GPUs in one host, PCIe-oriented custom allreduce options, x86 paths, and SM120 defaults. A DGX Spark cluster is different:

```text
4 hosts
1 GB10 GPU per host
SM121
RoCE/NCCL across nodes
unified/system memory pressure matters
```

The practical consequences are:

- Do not enable B12X PCIe allreduce across Spark hosts.
- Do use NCCL/RoCE over the high-speed Spark interface.
- Use SM121 build/runtime flags.
- Keep Ray as small as possible if using Ray-backed vLLM.
- Put all memory-critical env vars on every worker container, not just the head.
- Use DCP to avoid replicated MLA KV cache at 128K.
- Use explicit KV bytes instead of trusting one startup memory snapshot.

## Hardware and network assumptions

The recipe assumes the four Spark nodes are reachable on the high-speed network:

```text
head:    192.168.100.1
worker1: 192.168.100.2
worker2: 192.168.100.3
worker3: 192.168.100.4
iface:   enP2p1s0f0np0
```

The important network settings are:

```text
NCCL_SOCKET_IFNAME=enP2p1s0f0np0
GLOO_SOCKET_IFNAME=enP2p1s0f0np0
NCCL_IB_DISABLE=0
```

Do not copy the RTX single-host habit of disabling IB here. On Spark, that costs real decode performance.

## Model and image

Model directory:

```text
/var/tmp/models/Mapika/GLM-5.2-NVFP4-MTP-hybrid
```

The MTP hybrid directory matters. The base checkpoint does not contain the `model.layers.78.*` tensors needed by vLLM's MTP loader.

Promoted image:

```text
glm-darkdevotion-b12x:20260625-arm64-mtp1-trim
```

This image is derived from the Spark-compatible dark-devotion/B12X vLLM image and bakes in:

- local KV diagnostics used during bring-up;
- a post-weight-load memory trim hook;
- the vLLM files needed by the validated recipe.

Build it with:

```bash
./build-glm52-spark-overlay-image.sh
```

## Launch

```bash
cd /home/matt/code/blackwell-llm-docker/recipes/4x-spark-cluster/glm52-b12x-spark
./launch-glm52-mtp1-dcp4-128k.sh
```

The launcher starts a slim Ray cluster and then starts vLLM on the head node.

Ray is intentionally constrained:

```text
--object-store-memory=134217728
--object-spilling-directory=/var/tmp/ray-spill
--include-dashboard=false
--include-log-monitor=false
--disable-usage-stats
--num-cpus=1
--num-gpus=1
```

Ray may still start a small head-side `dashboard.py --modules-to-load=UsageStatsHead --disable-frontend` helper. That is not the full dashboard/frontend and has been treated as a known Ray behavior.

## Promoted env

The current baseline is captured in `glm52-mtp1-dcp4-128k.env`:

```text
IMAGE=glm-darkdevotion-b12x:20260625-arm64-mtp1-trim
MODEL_DIR=/var/tmp/models/Mapika/GLM-5.2-NVFP4-MTP-hybrid
PROFILE=custom
SERVED_MODEL_NAME=glm52-mtp1-dcp4-128k
TP_SIZE=4
PP_SIZE=1
DCP_SIZE=4
DCP_COMM_BACKEND=ag_rs
DCP_KV_CACHE_INTERLEAVE_SIZE=1
MAX_MODEL_LEN=131072
MAX_NUM_SEQS=1
MAX_NUM_BATCHED_TOKENS=1024
MAX_CUDAGRAPH_CAPTURE_SIZE=4
GPU_MEMORY_UTILIZATION=0.916
KV_CACHE_MEMORY_BYTES=1810000000
KV_CACHE_DTYPE=fp8
ENFORCE_EAGER=0
ENABLE_MTP=1
NUM_SPECULATIVE_TOKENS=1
ATTENTION_BACKEND=B12X_MLA_SPARSE
MOE_BACKEND=flashinfer_cutlass
VLLM_DCP_GLOBAL_TOPK=1
VLLM_DCP_SHARD_DRAFT=1
VLLM_KZ_TRIM_AFTER_LOAD=1
NCCL_IB_DISABLE=0
NCCL_SOCKET_IFNAME=enP2p1s0f0np0
VLLM_USE_B12X_MOE=0
VLLM_USE_B12X_FP8_GEMM=0
B12X_MOE_FORCE_A16=0
B12X_W4A16_TC_DECODE=0
```

Key choices:

- `DCP_SIZE=4` is what makes 128K context feasible on four Sparks.
- `KV_CACHE_MEMORY_BYTES=1810000000` avoids startup-profile variance at the memory edge.
- `NUM_SPECULATIVE_TOKENS=1` outperformed MTP3 at 128K.
- `MAX_NUM_BATCHED_TOKENS=1024` outperformed 2048 in the 128K MTP path.
- `flashinfer_cutlass` MoE beat the B12X MoE/W4A16 path on this workload.

## Validate the deployment

Run:

```bash
./validate-glm52-mtp1-dcp4-128k.sh
./smoke-glm52-quality.sh
MAX_TOKENS=512 ./bench-glm52-mtp.sh
```

Expected evidence:

- `/v1/models` reports `max_model_len=131072`.
- The API process includes `--tensor-parallel-size 4` and `--decode-context-parallel-size 4`.
- The API process includes `--kv-cache-memory-bytes 1810000000`.
- The speculative config includes MTP with one speculative token.
- Each Ray node exposes `object_store_memory=134217728`.
- `VLLM_KZ_TRIM_AFTER_LOAD completed ... malloc_trim=1` appears on all four ranks.
- Chat quality smoke passes without `ignore_eos`.
- MTP Prometheus counters advance during decode.

## MTP benchmark

Example 512-token result:

```text
completion_tokens=512
elapsed_s=35.688
client_completion_tps=14.346
metric_generation_delta=512
metric_generation_tps=14.346
spec_drafts_delta=295
spec_draft_tokens_delta=295
spec_accepted_tokens_delta=217
spec_acceptance_ratio=0.736
```

Example 256-token audit result:

```text
completion_tokens=256
elapsed_s=17.691
client_completion_tps=14.471
metric_generation_delta=256
metric_generation_tps=14.471
spec_drafts_delta=146
spec_draft_tokens_delta=146
spec_accepted_tokens_delta=109
spec_acceptance_ratio=0.747
```

These Prometheus deltas prove speculative decode is actually active. It is not enough to see `--speculative-config` in the process args.

## Quality and thinking mode

GLM-5.2 is a reasoning model. For concise answer/code generation, use no-thinking chat-template kwargs:

```json
{
  "chat_template_kwargs": {
    "enable_thinking": false,
    "thinking": false
  }
}
```

Without that, short prompts may return reasoning text in `content` before the final answer. That is not the same as DCP/top-k incoherence.

## Lessons learned

- MTU tuning was not the main performance lever.
- RDMA/NCCL over the Spark high-speed path was material.
- Ray overhead matters because Spark memory is system memory, not isolated GPU VRAM.
- DCP4 is the long-context lever; without it, 128K is not realistic in this envelope.
- MTP3 was not best at 128K. MTP1 was faster and had better effective behavior.
- BF16 KV was desirable but not the best viable 128K serving point on this hardware/model pair.
- B12X sparse MLA is required for this stack, but B12X MoE/W4A16 was not the fastest measured MoE path here.
- Explicit KV allocation made the final 128K path repeatable.
- A small post-load trim hook is useful at this memory edge.

## Caveats

- This is a batch-one, single-request decode-oriented configuration.
- It uses fp8 KV. BF16 KV remains preferable on principle, but it did not fit the 128K target with this model footprint and runtime overhead.
- The recipe is validated for the named image/model combination. Changing the checkpoint, vLLM fork, CUDA/NCCL stack, or B12X backend should be treated as a new experiment.
- Server-side `REASONING_PARSER=glm45` is wired but not part of the validated baseline because no-thinking chat requests already produce clean answer/code traffic.

## Follow-up: MTP2 was not a clear win

The same 128K/DCP4 setup was tested with `num_speculative_tokens=2`. It reached the same reported KV capacity as MTP1: `132,096` tokens, or `1.01x` concurrency for `131,072` tokens.

Throughput was effectively tied rather than better: hot 512-token decode measured `14.44 tok/s`, versus the earlier MTP1 512-token result of `14.35 tok/s` and best MTP1 hot sample of `15.18 tok/s`. Logs showed weak second-token speculative acceptance, so MTP1 remains the recommended public baseline.
