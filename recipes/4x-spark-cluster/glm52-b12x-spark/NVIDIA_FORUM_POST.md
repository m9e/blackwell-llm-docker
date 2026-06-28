# Draft: NVIDIA forum post

Title idea: Four DGX Sparks serving GLM-5.2 at 128K with vLLM DCP4, B12X sparse MLA, and MTP1

I wanted to share a working four-node DGX Spark result because Spark behaves differently from normal discrete-GPU servers and the memory pressure is severe.

Configuration:

```text
Hardware:        4x DGX Spark / GB10, one GPU per node
Network:         Spark high-speed fabric, NCCL/RDMA enabled
Interface:       enP2p1s0f0np0
Model:           GLM-5.2 NVFP4 MTP hybrid
Runtime:         vLLM fork with DCP and B12X patches
Parallelism:     TP4 / PP1 / DCP4
Spec decode:     MTP1
Attention:       B12X sparse MLA
MoE:             flashinfer_cutlass
KV cache:        fp8, explicit 1.81 GB/rank
Max model len:   131,072
Fitted capacity: 132,096 KV tokens
Decode speed:    about 14.5-15.2 output tok/s for short codegen prompts
```

The main lesson is that the OS and Ray footprint matter on Spark. The successful run required treating the nodes as dedicated inference appliances. We disabled irrelevant desktop/headless services, reduced journal footprint, drained swap before clean launches, and slimmed Ray down. Important: the desktop-service pruning disables the desktop GUI, so only do this on headless inference nodes.

Ray launch details:

```text
--include-dashboard=false
--include-log-monitor=false
--disable-usage-stats
--object-store-memory=134217728
--object-spilling-directory=/var/tmp/ray-spill
--num-cpus=1
--num-gpus=1
```

Communication details:

```text
NCCL_SOCKET_IFNAME=enP2p1s0f0np0
GLOO_SOCKET_IFNAME=enP2p1s0f0np0
NCCL_IB_DISABLE=0
```

For this cluster, disabling IB/RDMA was a major performance regression. That is worth calling out because some RTX PRO 6000 single-host guidance does not transfer to Spark.

The vLLM serving profile uses:

```text
TP=4
PP=1
DCP=4
MAX_MODEL_LEN=131072
MAX_NUM_SEQS=1
MAX_NUM_BATCHED_TOKENS=1024
MAX_CUDAGRAPH_CAPTURE_SIZE=4
KV_CACHE_DTYPE=fp8
KV_CACHE_MEMORY_BYTES=1810000000
ATTENTION_BACKEND=B12X_MLA_SPARSE
MOE_BACKEND=flashinfer_cutlass
MTP=1
VLLM_DCP_GLOBAL_TOPK=1
VLLM_DCP_SHARD_DRAFT=1
VLLM_DISABLE_TP_MQ_BROADCASTER=1
VLLM_KZ_TRIM_AFTER_LOAD=1
```

The `VLLM_DISABLE_TP_MQ_BROADCASTER=1` patch was needed because the TP/DCP message-queue broadcaster path could hang in this multi-node Ray-on-Spark startup. The process-group path worked.

Observed capacity log:

```text
GPU KV cache size: 132,096 tokens
Maximum concurrency for 131,072 tokens per request: 1.01x
```

Performance notes:

```text
Short codegen decode:  about 14.5-15.2 tok/s
Long-prompt prefill:   about 450-500 input tok/s in the 16K-112K tests
Post-TTFT decode:      about 13 tok/s at 32K-112K prompt sizes
```

The long-context summary wall-clock results are prefill-dominated. Decode after first token remains much closer to short-context speed, so do not quote blended summary wall-clock rates as decode throughput.

Concurrency note: this 128K profile uses `MAX_NUM_SEQS=1`, so concurrent requests queue. This is expected for the single-long-context target. A batch-serving variant should raise `MAX_NUM_SEQS` and re-fit the KV budget, probably by lowering max context.

What failed or was not promoted:

```text
BF16 KV at 128K: insufficient headroom
DCP4/MTP2: sometimes competitive, but memory-sensitive
DCP4/MTP3: later draft acceptance collapses
NCCL_IB_DISABLE=1: severe regression on Spark
Heavy Ray dashboard/default services: waste too much memory
```

The model checkpoint detail matters. The working setup uses the hybrid GLM-5.2 NVFP4 MTP checkpoint that contains `model.layers.78.*`. There is one actual MTP layer, so MTP1 is the production point. MTP2/MTP3 recursively reuse the same one-step predictor and are not currently the safe default.

Recipe branch:

```text
https://github.com/m9e/blackwell-llm-docker/tree/codex/glm52-spark-community-guide/recipes/4x-spark-cluster/glm52-b12x-spark
```

vLLM patch branch:

```text
https://github.com/m9e/vllm/tree/codex/glm52-spark-dcp-mtp-patches
```

I am publishing the recipe and patches so other Spark owners can reproduce or improve it.
