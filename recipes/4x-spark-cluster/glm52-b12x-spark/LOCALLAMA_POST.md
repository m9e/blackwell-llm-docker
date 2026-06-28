# Draft: LocalLLaMA post

Title idea: GLM-5.2 at 128K on 4x DGX Spark: DCP4 + B12X + MTP1

I got GLM-5.2 NVFP4 running on four DGX Sparks at 128K context. This is still a niche/hacky setup, but it is now a real serving point rather than just a proof of life.

Main result:

```text
4x DGX Spark / GB10, one GPU per node
GLM-5.2 NVFP4 MTP hybrid checkpoint
vLLM fork with DCP + B12X sparse MLA patches
TP4 / PP1 / DCP4 / MTP1
fp8 KV cache, explicit 1.81 GB/rank
131,072 max model len
132,096 fitted KV tokens
about 14.5-15.2 output tok/s on short-prompt codegen
```

Why this is interesting: the model is too large and the memory is too tight to treat Spark like normal discrete-GPU hardware. The win was combining decode-context parallelism with aggressive system/Ray memory trimming. DCP4 shards the decode context across the four TP ranks, which is what makes 128K feasible. MTP1 then recovers enough generation speed to be usable.

The setup is not just stock vLLM. It uses a patched vLLM branch with the dark-devotion DCP work, B12X sparse MLA pieces, FlashInfer/CUTLASS MoE, and a small Spark-specific fix to disable the TP/DCP message-queue broadcaster path that was hanging in multi-node Ray startup. NCCL/RDMA remains enabled over the Spark fabric. Disabling IB was a major perf regression on this cluster.

The Ray setup is intentionally tiny:

```text
Dashboard disabled
Log monitor disabled
Usage stats disabled
Object store 128 MiB
Object spilling to /var/tmp/ray-spill
1 CPU and 1 GPU advertised per node
host networking and host IPC
```

The OS also matters. I disabled irrelevant headless-node services like cups, avahi, bluetooth, ModemManager, colord, fwupd, packagekit, desktop portal/pipewire pieces, etc. Important: this disables the desktop GUI; only do this on headless inference nodes. On Spark unified memory, a few GB of random Linux/userland overhead can be the difference between fitting and failing.

Some measured numbers:

```text
Short codegen, MTP1: about 14.5-15.2 tok/s
16K summary e2e:     about 4.7 tok/s
32K summary e2e:     about 3.0 tok/s
64K summary e2e:     about 1.7 tok/s
112K summary e2e:    about 1.0 tok/s
```

The long-summary e2e numbers are mostly prefill. In a TTFT run, post-first-token decode stayed around 13 tok/s even at 64K/112K prompt sizes.

Important caveat on batching: the 128K profile is `MAX_NUM_SEQS=1`. I tested 8 simultaneous unique codegen prompts and it queued rather than true-batched:

```text
8/8 completed
3916 completion tokens
264.897 seconds wall
14.783 aggregate tok/s
prefix cache hit rate 0.0%
```

So this is a single-long-context profile, not a batch-serving profile. For bs=8 I would build a separate lower-context config with higher `MAX_NUM_SEQS`.

What did not work:

```text
BF16 KV at 128K: did not fit with enough headroom
DCP4/MTP3: later speculative positions collapsed in acceptance
DCP4/MTP2: sometimes competitive, but not stable enough to make default
NCCL_IB_DISABLE=1: bad idea on Spark, throughput regressed hard
Stock container assumptions: not enough for this stack
```

A key practical detail: use the hybrid checkpoint that actually contains `model.layers.78.*`. The base GLM checkpoint can advertise MTP metadata without the real MTP layer. This setup has exactly one MTP layer, so MTP1 is the clean production point. MTP2/MTP3 recursively reuse the same one-step predictor and are research territory.

The full guide and scripts are in the repo recipe:

```text
https://github.com/m9e/blackwell-llm-docker/tree/codex/glm52-spark-community-guide/recipes/4x-spark-cluster/glm52-b12x-spark
```

The vLLM patch branch is:

```text
https://github.com/m9e/vllm/tree/codex/glm52-spark-dcp-mtp-patches
```

I will keep polishing the branches/docs, but the current production recommendation is simple:

```text
DCP4 / 128K / MTP1
B12X sparse MLA
flashinfer_cutlass MoE
fp8 KV
Ray slimmed down
IB/RDMA enabled
```
