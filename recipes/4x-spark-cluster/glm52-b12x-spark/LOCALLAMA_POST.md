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

Some measured numbers, split the way they should be read:

```text
Short codegen decode, MTP1: about 14.5-15.2 tok/s
Long-prompt prefill:         about 450-500 input tok/s in the 16K-112K tests
Post-TTFT decode:            about 13 tok/s at 32K-112K prompt sizes
```

The summary wall-clock rates look much lower only if prefill/TTFT is blended into generation time. I would not quote those as decode throughput.

Important caveat on concurrency: the 128K profile is `MAX_NUM_SEQS=1`, so concurrent requests queue. This is a single-long-context recipe, not a batch-serving recipe. A batch-oriented variant should raise `MAX_NUM_SEQS` and re-fit the KV budget, probably by lowering max context. Exercise left to the reader.

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
https://github.com/m9e/blackwell-llm-docker/tree/main/recipes/4x-spark-cluster/glm52-b12x-spark
```

The vLLM patch branch is:

```text
https://github.com/m9e/vllm
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
