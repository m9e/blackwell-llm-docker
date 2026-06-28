# GLM-5.2 4x DGX Spark B12X recipe

This recipe is the working 4-node DGX Spark configuration for serving GLM-5.2 NVFP4 with vLLM, B12X sparse MLA, decode-context parallelism, and MTP speculative decoding.

Main result:

```text
Hardware: 4x NVIDIA DGX Spark / GB10, one GPU per host
Model:    Mapika/GLM-5.2-NVFP4-MTP-hybrid
Runtime:  vLLM fork with dark-devotion DCP + B12X patches
Profile:  TP4 / PP1 / DCP4 / MTP1
KV:       fp8, explicit 1.81 GB/rank
Context:  131,072 requested, 132,096 token fitted KV capacity
Speed:    about 14.5-15.2 output tok/s on short-prompt codegen
Batch:    max_num_seqs=1 production profile; bs=8 queues and stays about 14.8 aggregate tok/s
```

The current live/tested environment file is:

```bash
ENV_FILE=/home/matt/code/blackwell-llm-docker/recipes/4x-spark-cluster/glm52-b12x-spark/glm52-dcp4-mtp1-128k.env \
PATCH_DIAGNOSTICS=1 \
/home/matt/code/blackwell-llm-docker/recipes/4x-spark-cluster/glm52-b12x-spark/launch-glm52-mtp3-dcp4-128k.sh
```

There is also a cleaner frozen baseline from the first production point:

```bash
/home/matt/code/blackwell-llm-docker/recipes/4x-spark-cluster/glm52-b12x-spark/launch-glm52-mtp1-dcp4-128k.sh
```

That older wrapper uses `glm52-mtp1-dcp4-128k.env` and image `glm-darkdevotion-b12x:20260625-arm64-mtp1-trim`. The newer live profile uses image `glm-darkdevotion-b12x:20260626-arm64-mtpdiag21-draftprob` and keeps extra diagnostics available. Both are the same serving shape: DCP4, 128K, MTP1.

## Files

```text
launch-ray.sh                         Ray-on-Docker cluster launcher for 4 Sparks
serve.sh                              vLLM serve wrapper with DCP, MTP, B12X, and KV knobs
glm52-dcp4-mtp1-128k.env              current 128K/MTP1 production profile
glm52-dcp4-mtp2-120k-loose.env        experimental 120K/MTP2 profile
launch-glm52-mtp3-dcp4-128k.sh        generic current launcher used by MTP1/MTP2/MTP3 env files
launch-glm52-mtp1-dcp4-128k.sh        frozen clean MTP1 launcher
COMMUNITY_HOWTO_GLM52_SPARK.md        full reproducible guide
LOCALLAMA_POST.md                     community post draft
NVIDIA_FORUM_POST.md                  forum post draft
MTP_ITERATIVE_DCP_DIAGNOSIS.md        investigation notes on MTP2/MTP3 and DCP
SOULKILLER_MEMORY_PRUNE_20260627.md   memory-pruning checklist for Spark nodes
```

## Launch sequence

1. Put GLM-5.2 NVFP4 MTP hybrid weights on all four hosts at the same path.

```text
/models
```

2. Start the Ray cluster inside the vLLM container.

```bash
cd /home/matt/code/blackwell-llm-docker/recipes/4x-spark-cluster/glm52-b12x-spark
./launch-ray.sh
```

3. Start the current 128K/MTP1 serving profile.

```bash
cd /home/matt/code/blackwell-llm-docker/recipes/4x-spark-cluster/glm52-b12x-spark
ENV_FILE=$PWD/glm52-dcp4-mtp1-128k.env PATCH_DIAGNOSTICS=1 ./launch-glm52-mtp3-dcp4-128k.sh
```

4. Check readiness from the head node.

```bash
curl http://192.168.100.1:18089/v1/models
```

Expected vLLM capacity lines:

```text
GPU KV cache size: 132,096 tokens
Maximum concurrency for 131,072 tokens per request: 1.01x
```

## Ray layout

Ray is used as the vLLM distributed executor. It is intentionally slimmed down because Spark is unified memory constrained.

```text
Ray dashboard:           disabled
Ray log monitor:         disabled
Ray usage stats:         disabled
Object store memory:     128 MiB
Object spilling:         /var/tmp/ray-spill
Ray CPUs per node:       1
Ray GPUs per node:       1
Docker network:          host
Docker IPC:              host
```

The launcher pins all communication to the 100Gbps+ Spark fabric rather than Wi-Fi:

```text
NCCL_SOCKET_IFNAME=enP2p1s0f0np0
GLOO_SOCKET_IFNAME=enP2p1s0f0np0
NCCL_IB_DISABLE=0
```

## Important runtime knobs

```text
TP=4
PP=1
DCP=4
MTP=1
MAX_MODEL_LEN=131072
MAX_NUM_SEQS=1
MAX_NUM_BATCHED_TOKENS=1024
MAX_CUDAGRAPH_CAPTURE_SIZE=4
KV_CACHE_DTYPE=fp8
KV_CACHE_MEMORY_BYTES=1810000000
ATTENTION_BACKEND=B12X_MLA_SPARSE
MOE_BACKEND=flashinfer_cutlass
VLLM_DCP_GLOBAL_TOPK=1
VLLM_DCP_SHARD_DRAFT=1
VLLM_DISABLE_TP_MQ_BROADCASTER=1
VLLM_KZ_TRIM_AFTER_LOAD=1
```

## Why DCP4

GLM-5.2 uses a DeepSeek-style MLA cache. Without decode-context parallelism, the effective KV capacity on this hardware is too small for a useful 128K profile. DCP4 shards the decode context across the same four TP ranks. The production profile uses this to fit a 131,072 token context with about 1.01x concurrency.

DCP1 and DCP2 are faster or potentially cleaner for smaller contexts, but they do not provide the same context capacity at the same physical KV allocation:

```text
DCP1: smallest context capacity, least DCP communication, useful for short-context speed tests
DCP2: intermediate capacity and communication, a possible 64K-class compromise
DCP4: required for the 128K production target
```

Observed short-context signal from the investigation:

```text
DCP1 / 32K / MTP3: up to about 23.3 tok/s hot in a diagnostic run
DCP4 / 128K / MTP1: about 14.5-15.2 tok/s, production choice
DCP4 / 128K / MTP2: sometimes viable, but less stable and more memory sensitive
DCP4 / 128K / MTP3: not promoted; acceptance collapses in later speculative positions
```

## Batch behavior

The production profile is intentionally configured for one long-context request at a time:

```text
MAX_NUM_SEQS=1
```

A test with 8 simultaneous unique codegen prompts confirmed that this profile queues rather than true-batches:

```text
Requests completed:       8/8
Total completion tokens:  3916
Wall time:                264.897 s
Aggregate throughput:     14.783 completion tok/s
Prefix cache hit rate:    0.0%
Scheduler behavior:       Running 1 request, Waiting 7 then draining
```

That is expected. To optimize bs=8, use a separate lower-context profile with larger `MAX_NUM_SEQS` and a different KV budget.

## Long-prompt behavior

End-to-end throughput on summary tasks drops mostly because prefill dominates time-to-first-token. Decode after the first token remains roughly in the same band.

MTP1 TTFT diagnostic:

```text
16K prompt:  TTFT 35.476 s, post-TTFT decode 10.866 tok/s, e2e 4.336 tok/s
32K prompt:  TTFT 64.165 s, post-TTFT decode 13.430 tok/s
64K prompt:  TTFT 129.503 s, post-TTFT decode 13.316 tok/s
112K prompt: TTFT 222.568 s, post-TTFT decode 13.310 tok/s
```

## Known caveats

- MTP1 is the production choice. The checkpoint has one actual MTP layer, `model.layers.78.*`, so MTP2/MTP3 recursively reuse the same one-step predictor.
- MTP2 can work and sometimes approaches or slightly exceeds MTP1, but it is not the safe default.
- MTP3 is not production-ready in this stack. Acceptance in later draft positions collapses under DCP4.
- fp8 KV was required for this memory target. BF16 KV did not leave enough headroom for 128K on four Sparks with this model.
- `MAX_NUM_SEQS=1` is part of the 128K target. Throughput batching is a different profile.
- Do not disable IB/RDMA on Spark. `NCCL_IB_DISABLE=1` caused severe performance regressions.
