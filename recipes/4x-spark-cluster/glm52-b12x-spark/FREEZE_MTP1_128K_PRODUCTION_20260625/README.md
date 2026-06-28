# GLM-5.2 B12X on 4x DGX Spark

This is the Spark-specific 4-node GLM-5.2 recipe. It is not the upstream single-host RTX PRO 6000 recipe copied verbatim: Spark is four hosts with one GB10 GPU each, SM121, and RoCE/NCCL between nodes.

The promoted production baseline is:

```bash
./launch-glm52-mtp1-dcp4-128k.sh
```

As of 2026-06-25, this launches a validated 128K-context GLM-5.2 NVFP4 MTP hybrid deployment with TP4/DCP4/MTP1, B12X sparse MLA attention, `flashinfer_cutlass` MoE, fp8 KV, CUDA graphs, and explicit KV allocation.

For the longer community-facing write-up, see [COMMUNITY_HOWTO_GLM52_SPARK.md](COMMUNITY_HOWTO_GLM52_SPARK.md).

## Current best-known config

Source env: `glm52-mtp1-dcp4-128k.env`

```text
IMAGE=glm-darkdevotion-b12x:20260625-arm64-mtp1-trim
MODEL_DIR=/var/tmp/models/Mapika/GLM-5.2-NVFP4-MTP-hybrid
SERVED_MODEL_NAME=glm52-mtp1-dcp4-128k
TP_SIZE=4
PP_SIZE=1
DCP_SIZE=4
DCP_COMM_BACKEND=ag_rs
MAX_MODEL_LEN=131072
MAX_NUM_SEQS=1
MAX_NUM_BATCHED_TOKENS=1024
MAX_CUDAGRAPH_CAPTURE_SIZE=4
GPU_MEMORY_UTILIZATION=0.916
KV_CACHE_MEMORY_BYTES=1810000000
KV_CACHE_DTYPE=fp8
ENABLE_MTP=1
NUM_SPECULATIVE_TOKENS=1
ATTENTION_BACKEND=B12X_MLA_SPARSE
MOE_BACKEND=flashinfer_cutlass
VLLM_DCP_GLOBAL_TOPK=1
VLLM_DCP_SHARD_DRAFT=1
VLLM_KZ_TRIM_AFTER_LOAD=1
NCCL_IB_DISABLE=0
NCCL_SOCKET_IFNAME=enP2p1s0f0np0
```

B12X MoE/W4A16 is intentionally disabled in the promoted env because it loaded but was slower than `flashinfer_cutlass` on this four-Spark batch-one workload:

```text
VLLM_USE_B12X_MOE=0
VLLM_USE_B12X_FP8_GEMM=0
B12X_MOE_FORCE_A16=0
B12X_W4A16_TC_DECODE=0
```

## Launch

```bash
cd /home/matt/code/blackwell-llm-docker/recipes/4x-spark-cluster/glm52-b12x-spark
./launch-glm52-mtp1-dcp4-128k.sh
```

The endpoint is:

```text
http://192.168.100.1:18089/v1
```

The server log is inside the head container:

```bash
docker exec glm-dark-head tail -f /tmp/glm52-spark-custom.log
```

## Validate

Run all three checks after launch:

```bash
./validate-glm52-mtp1-dcp4-128k.sh
./smoke-glm52-quality.sh
MAX_TOKENS=512 ./bench-glm52-mtp.sh
```

What they prove:

- `validate-glm52-mtp1-dcp4-128k.sh` verifies image parity across all four nodes, slimmed Ray settings, no Ray log monitor, 128 MiB object store per node, spill directory, serve args, TP4/DCP4/MTP1, explicit KV allocation, fp8 KV, B12X sparse MLA, `flashinfer_cutlass` MoE, trim hook on all ranks, `/v1/models max_model_len=131072`, and a short decode.
- `smoke-glm52-quality.sh` verifies ordinary OpenAI chat completions without `ignore_eos` using no-thinking chat-template kwargs.
- `bench-glm52-mtp.sh` verifies throughput and actual speculative decode activity by comparing vLLM Prometheus counters before/after the request.

## Current measured performance

128K TP4/DCP4 config, batch size 1, no-thinking chat codegen:

```text
no-MTP hot:              10.427 tok/s
MTP3 hot, 1024 budget:   12.698 tok/s
MTP3 hot, 2048 budget:   11.881 tok/s
MTP1 hot:                15.176 tok/s
MTP1 baked-image hot:    14.460 tok/s
MTP1 audit run:          14.471 tok/s, acceptance 0.747
```

MTP1 is the current promoted setting. MTP2 has not yet been measured in this 128K setup.

## MTP-aware benchmark example

```bash
MAX_TOKENS=512 ./bench-glm52-mtp.sh
```

Known 512-token MTP1 result:

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

Known 256-token audit result:

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

These counters confirm MTP1 is active in vLLM, not just present in launch args.

## Quality smoke and thinking mode

GLM-5.2 is a reasoning model. If requests do not disable thinking, short deterministic prompts may return reasoning text in `content` before the final answer. That is expected model/template behavior and is distinct from DCP/top-k incoherence.

For concise answer/code traffic, send:

```json
"chat_template_kwargs": {
  "enable_thinking": false,
  "thinking": false
}
```

`smoke-glm52-quality.sh` uses those kwargs and passes exact-token, arithmetic, and code-shape checks without `ignore_eos`.

`serve.sh` also supports parser knobs for a future controlled relaunch:

```bash
REASONING_PARSER=glm45 ./launch-glm52-mtp1-dcp4-128k.sh
```

That has not been forced onto the validated live env because it requires a full model reload and is not required for no-thinking answer/code traffic.

## Spark-specific differences from upstream RTX recipe

- Use `CUTE_DSL_ARCH=sm_121a` for GB10/SM121.
- Use the high-speed Spark interface: `NCCL_SOCKET_IFNAME=enP2p1s0f0np0`.
- Keep `NCCL_IB_DISABLE=0`; Spark uses RoCE/NCCL over the interconnect.
- Disable B12X PCIe allreduce with `VLLM_ENABLE_PCIE_ALLREDUCE=0`; that path is same-host PCIe oriented and not valid across four Spark hosts.
- Use Ray for the currently operational multi-node executor path.
- Keep Ray object store small: `134217728` bytes per node.
- Disable Ray log monitor and usage stats.
- Set B12X/CuTe/VLLM sparse-indexer env on every host at container startup, not only on the head API process.
- Explicitly select `--attention-backend B12X_MLA_SPARSE`.
- Explicitly select `--moe-backend flashinfer_cutlass` for the promoted Spark path.
- For MTP, explicitly set `draft_attention_backend=B12X_MLA_SPARSE` in `--speculative-config`.
- Use explicit `--kv-cache-memory-bytes 1810000000`; relying only on `gpu_memory_utilization` left too much startup/profile variance at the memory edge.

## Image build

The promoted image bakes local vLLM diagnostics and the post-load trim hook into the image instead of runtime patching:

```bash
./build-glm52-spark-overlay-image.sh
```

Image tag:

```text
glm-darkdevotion-b12x:20260625-arm64-mtp1-trim
```

The trim hook runs after weights load and before vLLM profiles available memory:

```python
gc.collect()
torch.cuda.empty_cache()
ctypes.CDLL("libc.so.6").malloc_trim(0)
```

The validator confirms `VLLM_KZ_TRIM_AFTER_LOAD completed ... malloc_trim=1` on all four ranks.

## Historical notes

Earlier profiles are retained only as history:

- BF16/auto KV DCP4 no-MTP proved the high-context DCP path but did not remain the best 128K capacity/performance point.
- DCP2/MTP3 was a useful performance/context compromise at smaller context, but not the 128K target.
- DCP4/MTP3 reached 128K but decoded slower than MTP1 and had weaker speculative value.
- no-MTP at 128K had slightly more KV headroom but was materially slower.
- B12X MoE/W4A16 loaded but was slower than `flashinfer_cutlass` on this Spark batch-one workload.

## MTP2 follow-up result, 2026-06-25

MTP2 was tested with the same DCP4/128K recipe as MTP1, changing only `NUM_SPECULATIVE_TOKENS=2` and serving as `glm52-mtp2-dcp4-128k`.

Capacity matched MTP1: vLLM reported `GPU KV cache size: 132,096 tokens` and `Maximum concurrency for 131,072 tokens per request: 1.01x`.

Performance did not clearly beat MTP1. A cold 256-token codegen request measured `7.25 tok/s`; the hot 256-token repeat measured `13.07 tok/s`; a hot 512-token request measured `14.44 tok/s`. The prior MTP1 512-token result was `14.35 tok/s`, and the best MTP1 hot sample was `15.18 tok/s`.

The observed reason is acceptance quality: MTP2 logged mean acceptance length roughly `1.7-2.1` during hot decode, with second-position acceptance often around `0.14-0.34`. That makes MTP2 roughly neutral at best on this setup. MTP1 remains the recommended default until a different draft schedule, prompt distribution, or MTP implementation shows a clear gain.
