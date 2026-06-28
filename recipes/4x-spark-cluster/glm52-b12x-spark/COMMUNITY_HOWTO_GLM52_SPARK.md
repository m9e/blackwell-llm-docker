# Serving GLM-5.2 NVFP4 at 128K on four DGX Sparks

This guide documents a working 4x DGX Spark setup for GLM-5.2 NVFP4 with 128K context. The result is not a generic vLLM recipe. It depends on a patched vLLM fork, B12X sparse MLA support, careful Ray trimming, system memory pruning, and a specific DCP/MTP tradeoff.

## Result

```text
Hardware:      4x NVIDIA DGX Spark / GB10, one GPU per node
Interconnect:  Spark high-speed fabric via enP2p1s0f0np0, NCCL IB enabled
Model:         Mapika/GLM-5.2-NVFP4-MTP-hybrid
Runtime:       vLLM fork with DCP + B12X + Spark patches
Serving shape: TP4 / PP1 / DCP4 / MTP1
Attention:     B12X sparse MLA
MoE:           flashinfer_cutlass
KV:            fp8, explicit 1.81 GB/rank
Context:       131,072 requested, 132,096 fitted KV capacity
Speed:         about 14.5-15.2 output tok/s on short-prompt codegen
```

The important point is that DCP4 makes the 128K cache fit. MTP1 then recovers enough decode throughput to make the result usable.

## Hardware and OS assumptions

Each Spark has one GB10 GPU with unified memory. The limiting resource is not just GPU memory in the discrete-GPU sense. The OS, Ray, Docker, vLLM, CUDA graphs, model weights, KV cache, and random desktop services all fight over the same memory pool.

Use the high-speed Spark interface for NCCL and Gloo. On this cluster that interface is:

```text
enP2p1s0f0np0
```

The working NCCL posture is:

```text
NCCL_SOCKET_IFNAME=enP2p1s0f0np0
GLOO_SOCKET_IFNAME=enP2p1s0f0np0
NCCL_IB_DISABLE=0
```

Do not copy the RTX PRO 6000 single-host advice to Spark blindly. Disabling IB is appropriate for some pure PCIe setups, but it is wrong here.

## Repositories

The final recipe lives in:

```text
blackwell-llm-docker/recipes/4x-spark-cluster/glm52-b12x-spark
```

The key runtime dependency is a vLLM fork carrying DCP and B12X changes. The final writeup should link the published branch after it is pushed.

Historical references that informed this setup:

```text
local-inference-lab/rtx6kpro GLM-5.2 recipe
lukealonso/b12x
local-inference-lab/vllm dark-devotion branch
spark-vllm-docker Spark container work
```

## Model checkpoint

Use the hybrid checkpoint with a real MTP layer:

```text
/var/tmp/models/Mapika/GLM-5.2-NVFP4-MTP-hybrid
```

It must contain:

```text
model.layers.78.*
```

The base GLM checkpoint advertises MTP metadata but lacks the actual layer 78 weights. That can lead to failed or misleading MTP tests. Earlier no-MTP experiments temporarily removed layer 78 because vLLM can load it if it exists even when not declared. The production MTP1 setup requires it.

The checkpoint has one actual MTP layer, not three. MTP1 is therefore the natural production point. MTP2/MTP3 recursively reuse the same one-step predictor and are treated as experimental.

## Build/runtime image

The working image family is an ARM64/SM121 build with vLLM, B12X, FlashInfer/CUTLASS, and Spark patches. The current live profile used:

```text
glm-darkdevotion-b12x:20260626-arm64-mtpdiag21-draftprob
```

A clean frozen baseline also exists:

```text
glm-darkdevotion-b12x:20260625-arm64-mtp1-trim
```

Both are DCP4/MTP1/128K capable. The later image keeps diagnostic code paths available for MTP/DCP work.

This is a containerized Ray/vLLM deployment. Ray is not started on the host OS. `launch-ray.sh` starts one Docker container per Spark and runs `ray start` inside those containers.

Do not assume a stock NGC vLLM image is sufficient. Some newer NVIDIA vLLM containers dropped Ray, while this recipe requires the Ray CLI and Python package inside the serving image because vLLM is launched with the Ray distributed executor. If you rebuild from a stock image, install a vLLM-compatible Ray package into the image before using this launcher. The lightweight path is the base `ray` package, not `ray[default]`, because the dashboard is disabled at runtime.

The overlay helper in this recipe is:

```bash
cd recipes/4x-spark-cluster/glm52-b12x-spark
BASE_IMAGE=glm-darkdevotion-b12x:20260624-arm64-mtpfix6 \
IMAGE=glm-darkdevotion-b12x:20260626-arm64-mtpdiag21-draftprob \
VLLM_SOURCE_DIR=<path-to-your-vllm-fork> \
./build-glm52-spark-overlay-image.sh
```

That script does not create the entire stack from scratch. It copies the patched vLLM Python files from `VLLM_SOURCE_DIR` into `/usr/local/lib/python3.12/dist-packages/vllm/...`, builds the named image locally, and then builds the same image on the worker hosts over SSH. The base image must already contain the compiled vLLM/B12X/CUDA/FlashInfer pieces and Ray.

Key build/runtime requirements:

```text
Native ARM64 build
SM121 target support
B12X sparse MLA support
FlashInfer/CUTLASS MoE support
DCP global top-k support
DCP draft/index KV sharding support
Spark-safe DCP broadcaster behavior
CUDA graph support preserved
```

The one small Spark-specific patch that mattered operationally disables the TP/DCP message-queue broadcaster when requested:

```text
VLLM_DISABLE_TP_MQ_BROADCASTER=1
```

That avoids hangs in the multi-node Ray-on-Spark path before model load. The normal process-group path is sufficient for this setup.

## System memory pruning

This work only fit reliably after treating every node as an inference appliance.

Disable services that are irrelevant on headless Spark inference nodes. Warning: the desktop-service commands below intentionally disable the desktop GUI. Do not run them on a Spark you still use as an interactive desktop:

```bash
sudo systemctl disable --now cups cups-browsed avahi-daemon bluetooth ModemManager colord fwupd packagekit apport upower udisks2 switcheroo-control rtkit-daemon snapd || true
```

For desktop/user services, disable where present:

```bash
systemctl --user disable --now pipewire pipewire-pulse wireplumber xdg-desktop-portal xdg-desktop-portal-gnome xdg-document-portal snapd-desktop-integration || true
```

Keep the services that actually matter:

```text
NetworkManager
wpa_supplicant if needed for management
sshd
docker/containerd
systemd-resolved
NVIDIA persistence/RDMA pieces
```

Reduce journal footprint and swap aggressiveness:

```bash
sudo journalctl --vacuum-size=256M
printf 'vm.swappiness=1\n' | sudo tee /etc/sysctl.d/99-spark-inference.conf
sudo sysctl --system
```

Before clean launches, drain swap and drop caches if safe:

```bash
sudo swapoff -a && sudo swapon -a
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

The most useful diagnostic is not just `nvidia-smi`. Also inspect:

```bash
cat /proc/meminfo
```

Look for `MemAvailable`, `SwapFree`, `SwapCached`, and whether one node is materially worse than the others. In this cluster, `soulkiller` was initially worse than the head node until desktop/background services were pruned.

## Ray setup

Ray runs inside the Docker containers created by `launch-ray.sh`; it is not a host-level Ray cluster. The container image named by `IMAGE` must already have `ray`, `vllm`, B12X, CUDA, and FlashInfer installed.

The launcher first removes stale containers, then starts a head container on `192.168.100.1` and worker containers on `192.168.100.2`, `192.168.100.3`, and `192.168.100.4`. Each container uses host networking, host IPC, all GPUs, `memlock=-1`, and a read-only model bind mount:

```text
-v /var/tmp/models/Mapika/GLM-5.2-NVFP4-MTP-hybrid:/models:ro
```

Inside those containers, Ray is slimmed down aggressively.

Head node shape:

```bash
ray start \
  --head \
  --node-ip-address=192.168.100.1 \
  --port=26379 \
  --include-dashboard=false \
  --include-log-monitor=false \
  --disable-usage-stats \
  --object-store-memory=134217728 \
  --object-spilling-directory=/var/tmp/ray-spill \
  --num-cpus=1 \
  --num-gpus=1 \
  --temp-dir=/tmp/ray-vllm-head \
  --block
```

Worker shape:

```bash
ray start \
  --address=192.168.100.1:26379 \
  --node-ip-address=<worker-ip> \
  --include-dashboard=false \
  --include-log-monitor=false \
  --disable-usage-stats \
  --object-store-memory=134217728 \
  --object-spilling-directory=/var/tmp/ray-spill \
  --num-cpus=1 \
  --num-gpus=1 \
  --temp-dir=/tmp/ray-vllm-worker \
  --block
```

The Docker launcher uses host networking and host IPC. It bind-mounts the model read-only as `/models` and sets `memlock=-1`.

Ray was kept because vLLM's Ray executor path was the working multi-node path for this fork. The memory win came from trimming Ray rather than removing it: no dashboard, no log monitor, no usage stats, 128 MiB object store, one CPU advertised per node.

## vLLM serving configuration

The production env is equivalent to:

```text
MODEL_PATH=/models
SERVED_MODEL_NAME=glm52-mtpdiag-dcp4-mtp1-128k
TP=4
PP=1
DCP=4
DCP_COMM_BACKEND=all_gather
MAX_MODEL_LEN=131072
MAX_NUM_SEQS=1
MAX_NUM_BATCHED_TOKENS=1024
MAX_CUDAGRAPH_CAPTURE_SIZE=4
GPU_MEMORY_UTILIZATION=0.918
KV_CACHE_DTYPE=fp8
KV_CACHE_MEMORY_BYTES=1810000000
ATTENTION_BACKEND=B12X_MLA_SPARSE
MOE_BACKEND=flashinfer_cutlass
ENABLE_PREFIX_CACHING=1
MTP=1
VLLM_DCP_GLOBAL_TOPK=1
VLLM_DCP_SHARD_DRAFT=1
VLLM_USE_B12X_SPARSE_INDEXER=1
VLLM_DISABLE_TP_MQ_BROADCASTER=1
VLLM_KZ_TRIM_AFTER_LOAD=1
```

The launch command is:

```bash
cd recipes/4x-spark-cluster/glm52-b12x-spark
./launch-ray.sh
ENV_FILE=$PWD/glm52-dcp4-mtp1-128k.env PATCH_DIAGNOSTICS=1 ./launch-glm52-mtp3-dcp4-128k.sh
```

Expected capacity log:

```text
GPU KV cache size: 132,096 tokens
Maximum concurrency for 131,072 tokens per request: 1.01x
```

## Why fp8 KV

BF16 KV is preferable for quality in principle, but the 128K target did not fit with enough headroom on four Sparks. fp8 KV plus DCP4 was the practical path to 128K.

This is not naive full K/V cache math. GLM-5.2's MLA-style cache is much smaller than classical attention KV, but the implementation still has index caches, draft/index groups, activation peaks, CUDA graph buffers, and allocator overhead. The actual fitted capacity is the source of truth.

## Performance notes

Short-prompt codegen with MTP1 typically lands around:

```text
14.5-15.2 completion tok/s
```

Performance should be read as prefill plus decode:

```text
Short codegen decode:   about 14.5-15.2 tok/s
12.5K translate decode: 14.489 tok/s
25K translate decode:   14.408 tok/s
50K translate decode:   14.492 tok/s
```

For long summary prompts, TTFT/prefill dominates wall time, but decode after TTFT remains much higher than the blended wall-clock number:

```text
Prompt size  TTFT / prefill     Approx prefill rate    Decode after TTFT
16K          35.476 s           about 450 tok/s        10.866 tok/s
32K          64.165 s           about 500 tok/s        13.430 tok/s
64K          129.503 s          about 494 tok/s        13.316 tok/s
112K         222.568 s          about 503 tok/s        13.310 tok/s
```

The older blended wall-clock summary rates were about 4.7, 3.0, 1.7, and 1.0 tok/s at 16K, 32K, 64K, and 112K respectively. Those are end-to-end user-wait rates for short summaries, not decode rates.

This is a single-long-context profile, not a batch-serving profile. It uses `MAX_NUM_SEQS=1`, so concurrent requests queue. To make a batch-serving variant, raise `MAX_NUM_SEQS` and re-fit the KV budget, usually by lowering `MAX_MODEL_LEN` or accepting less 128K headroom. That tradeoff is intentionally left to the reader.

## DCP1 and DCP2 tradeoff

DCP changes context capacity by sharding the sequence dimension across decode-context ranks.

Directionally, with the same physical KV pool:

```text
DCP1: about 1/4 the logical context of DCP4
DCP2: about 1/2 the logical context of DCP4
DCP4: full 128K target on this setup
```

DCP1 can be faster because it avoids the DCP communication and sharding complexity. In diagnostics, DCP1 at smaller context with MTP3 reached about 23.3 tok/s hot. That does not replace the production target because it cannot provide 128K at the same memory budget.

DCP2 is the likely compromise profile if you want a 64K-class model with less DCP overhead. It was not promoted here because the goal was maximum context at acceptable speed.

## What did not work

BF16 KV at 128K did not fit with the full model and useful graph headroom.

Disabling IB/RDMA regressed throughput badly. Spark needs the high-speed fabric path.

MTP3 under DCP4 was not viable. Later draft-token acceptance collapsed. The likely causes are a mix of recursive one-layer MTP economics and DCP-specific draft/verification behavior. The investigation found that GLM-5.2 uses the generic MTP proposer path, not the Step3.5 path that was initially suspected.

MTP2 is real but not the default. A 120K/MTP2 profile can be competitive, but it is more memory-sensitive and less consistently better than MTP1. The production-safe recommendation remains DCP4/128K/MTP1.

Naively relying on `nvidia-smi` is misleading on Spark. Use `/proc/meminfo`, Ray process state, swap state, and vLLM's own memory breakdowns.

## Quality caveat

The result is a working inference configuration, not a full model-quality certification. We used forced decode and measurement prompts during diagnostics. For production use, validate your own chat templates, EOS behavior, tool calling, long-context retrieval, and task quality.

## Operational checklist

Before launch:

```text
All four nodes can reach each other on the IB/RoCE fabric
Model path exists and is identical on every host
High-speed interface is up and selected
Unneeded desktop/system services are disabled
Swap is clean or intentionally understood
Ray spill directory exists
Docker image is present on every host
```

During launch:

```text
Ray reports four nodes
NCCL logs show IB/RDMA, not disabled IB
vLLM logs show B12X sparse MLA
vLLM logs show flashinfer_cutlass MoE
vLLM logs show DCP4 and MTP1
KV capacity is 132,096 tokens
Max concurrency for 131,072 is about 1.01x
```

After launch:

```text
Run a short unique prompt
Check prefix cache hit rate when measuring
Check post-TTFT decode separately from prefill-heavy end-to-end speed
```
