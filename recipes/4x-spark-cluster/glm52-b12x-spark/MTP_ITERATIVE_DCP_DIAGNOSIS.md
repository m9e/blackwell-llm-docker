# GLM-5.2 MTP2/MTP3 DCP4 diagnosis

Date: 2026-06-25

## Frozen production point

The known-good production point is frozen in `FREEZE_MTP1_128K_PRODUCTION_20260625`:

- TP4 / PP1 / DCP4
- MTP1
- max_model_len 131072
- fp8 KV
- B12X sparse MLA
- flashinfer_cutlass MoE
- explicit KV pool 1,810,000,000 bytes/rank
- reported KV capacity 132,096 tokens
- reported concurrency for 131,072-token request 1.01x
- observed decode around 14.5-15.2 tok/s

MTP2 reached the same capacity but did not clearly beat MTP1:

- cold 256-token run: 7.25 tok/s
- hot 256-token run: 13.07 tok/s
- hot 512-token run: 14.44 tok/s
- logs showed weak second-position acceptance, commonly ~0.14-0.34

MTP1 remains the production baseline.

## Code-level finding 1: MTP1 bypasses the suspicious path

`vllm/v1/spec_decode/step3p5.py` has a separate Step3.5 override.

For `num_speculative_tokens == 1`, it returns immediately after the first draft forward:

```python
if self.num_speculative_tokens == 1 or self.parallel_drafting:
    ...
    return draft_token_ids.view(-1, self.num_speculative_tokens)
```

MTP2/MTP3 enter the iterative branch below that return. Therefore MTP1 being healthy does not validate the later iterative path.

## Code-level finding 2: DCP local sequence lengths can go stale

The Step3.5 iterative branch mutates global metadata:

```python
common_attn_metadata.seq_lens -= num_rejected_tokens_gpu
common_attn_metadata._seq_lens_cpu = None
common_attn_metadata._num_computed_tokens_cpu = None
```

It also increments positions and relies on `_update_positions_dependent_metadata()` for later draft steps.

However, Step3.5 does not recompute `common_attn_metadata.dcp_local_seq_lens` after these mutations.

That matters because the B12X sparse MLA metadata builder explicitly prefers the DCP-local length when present:

```python
seq_lens_for_req = (
    cm.dcp_local_seq_lens
    if cm.dcp_local_seq_lens is not None
    else cm.seq_lens
)
```

So with DCP4, later draft steps can use updated global `seq_lens` / positions / slot mappings while B12X still sees old DCP-local lengths. That is exactly the kind of error that would hurt draft position 2+ while leaving MTP1 mostly unaffected.

## Code-level finding 3: Step3.5 likely misses sparse-index sharing toggles

The base MTP proposer detects `index_share_for_mtp_iteration` and has logic for sharing target `topk_indices_buffer` and toggling `set_skip_topk()` so MTP step 0 computes indices and steps 1+ reuse them.

The model-side hook exists in `vllm/model_executor/models/deepseek_mtp.py`:

```python
def set_skip_topk(self, skip: bool):
    ...
    mla_attn.skip_topk = skip
```

But `Step3p5MTPProposer.propose()` does not reference `_share_mtp_indices`, `set_skip_topk`, or `skip_topk` at all. That means MTP2/MTP3 likely rerun distributed sparse top-k on later draft steps or fail to use the intended shared-index path.

This is both a performance issue and a possible correctness/acceptance issue if later draft steps are using indices that do not correspond to the intended step-0 shared-index state.

## Current best explanation

The observed curve is most consistent with two overlapping issues:

1. Economic cost: MTP2/MTP3 do additional sequential MTP forwards, rereading MTP weights and doing DCP communication.
2. Iterative-DCP metadata bug: later draft steps likely use stale or inconsistent DCP-local sequence lengths and possibly wrong/recomputed sparse top-k metadata.

The bug hypothesis is stronger than pure overhead because MTP2/MTP3 show a sharp acceptance drop at later draft positions, while MTP1 remains healthy.

## Minimal safe patch direction

Do not modify the frozen production config. Build a new experimental image tag only.

Patch target:

- `vllm/v1/spec_decode/step3p5.py`

Patch shape:

1. Add a helper that recomputes DCP-local sequence lengths whenever Step3.5 mutates global sequence lengths or advances a draft position.

```python
def _refresh_dcp_local_seq_lens(self, common_attn_metadata):
    dcp_size = self.vllm_config.parallel_config.decode_context_parallel_size
    if dcp_size <= 1 or common_attn_metadata.dcp_local_seq_lens is None:
        return
    dcp_rank = get_dcp_group().rank_in_group
    updated = get_dcp_local_seq_lens(
        common_attn_metadata.seq_lens[:common_attn_metadata.num_reqs],
        dcp_size,
        dcp_rank,
        self.vllm_config.parallel_config.cp_kv_cache_interleave_size,
    )
    common_attn_metadata.dcp_local_seq_lens[:common_attn_metadata.num_reqs].copy_(updated)
```

Also keep `dcp_local_seq_lens_cpu` and `seq_lens_cpu_upper_bound` consistent when those fields are populated.

2. Call this helper after rejected-token removal and after `_update_positions_dependent_metadata()` advances a draft step.

3. Add Step3.5-specific sparse-index sharing:

```python
if self._share_mtp_indices and hasattr(self.model.model, "set_skip_topk"):
    self.model.model.set_skip_topk(False)  # step 0 computes
...
if self._share_mtp_indices and hasattr(self.model.model, "set_skip_topk"):
    self.model.model.set_skip_topk(True)   # steps 1+ reuse
...
finally:
    self.model.model.set_skip_topk(False)
```

4. Add temporary diagnostic logging behind an env flag, not always-on logs:

- draft step index
- global `seq_lens`
- `seq_lens_cpu_upper_bound`
- `dcp_local_seq_lens`
- expected DCP-local seq lens recomputed from global seq lens
- primary and per-group slot mapping
- skip_topk state if exposed

## Test order

1. Build experimental image, do not overwrite the production image.
2. Launch MTP2 DCP4 128K first.
3. Compare:
   - tok/s
   - mean acceptance length
   - per-position acceptance
   - accepted/drafted counts
4. Launch MTP3 DCP4 128K only after MTP2 improves or at least shows healthier position-2 acceptance.
5. If no improvement, restore MTP1 production image/config and treat MTP2/MTP3 as not worth further time until upstream Step3.5 changes land.

## 2026-06-26 iterfix1 experiment

Built `glm-darkdevotion-b12x:20260625-arm64-mtp-iterfix1` from the local dark-devotion vLLM tree. The overlay now includes `vllm/v1/spec_decode/step3p5.py` in addition to the KV diagnostics / trim files. The Step3.5 patch refreshed `dcp_local_seq_lens` after iterative-position updates and enabled the MTP model `set_skip_topk()` path for later draft iterations when index sharing is active. A CPU upper-bound decrement was removed before build because the base proposer already accounts for rejected tokens there.

MTP2 launched successfully at the full production capacity: DCP4, fp8 KV, B12X MLA sparse, FlashInfer CUTLASS MoE, `kv_cache_memory_bytes=1810000000`, `max_model_len=131072`. vLLM reported `GPU KV cache size: 132,096 tokens` and `Maximum concurrency for 131,072 tokens per request: 1.01x`.

Measured codegen prompt results:

| config | image | max tokens | elapsed | throughput | notes |
| --- | --- | ---: | ---: | ---: | --- |
| MTP2 DCP4 128K | `20260625-arm64-mtp-iterfix1` | 256 | 18.613s | 13.754 tok/s | second-position acceptance remained weak |
| MTP2 DCP4 128K | `20260625-arm64-mtp-iterfix1` | 512 | 37.250s | 13.745 tok/s | stable but below hot MTP1 |
| MTP1 DCP4 128K | `20260625-arm64-mtp1-trim` | 512 | 37.239s | 13.749 tok/s | first post-restore request, cold-ish |
| MTP1 DCP4 128K | `20260625-arm64-mtp1-trim` | 512 | 35.131s | 14.574 tok/s | hot comparison, production winner |

MTP2 speculative metrics during the 256/512 runs showed mean acceptance length around `1.74-2.00`, first-position acceptance roughly `0.57-0.78`, and second-position acceptance roughly `0.18-0.27` for the main hot windows, with one terminal short window at `0.43`. That is not enough added accepted output to pay for the second sequential MTP forward plus DCP communication. The patch did not recover MTP2/3 behavior enough to displace MTP1.

Current conclusion: the simple DCP-local sequence refresh plus later-step `skip_topk` experiment is insufficient. The strongest remaining hypotheses are either (1) MTP2/3 economics are genuinely bad at DCP4 for this model/prompt mix because draft position 2 acceptance is too low, or (2) the iterative branch still needs deeper DCP-aware slot/position/index-cache handling than this patch provides. MTP3 was intentionally not relaunched on `iterfix1` because MTP2 failed to beat MTP1.

After the experiment, production was restored to `glm52-mtp1-dcp4-128k` on `glm-darkdevotion-b12x:20260625-arm64-mtp1-trim`, with the same `132,096` token KV capacity and `1.01x` 131K concurrency.

## 2026-06-26: DCP4 MTP2 draft-replication diagnostic

Tested `VLLM_DCP_SHARD_DRAFT=0` with the `glm52-mtp2-dcp4-16k-draftrep` profile to check whether DCP-sharded draft/index KV handling was the reason MTP2 underperformed MTP1.

The launch confirmed `VLLM_DCP_GLOBAL_TOPK=1` and `VLLM_DCP_SHARD_DRAFT=0` inside the head and worker containers. It loaded the target and MTP shards and reached KV sizing, reporting a `GPU KV cache size` of `114,425` tokens and `6.98x` concurrency for `16,384` tokens. It then failed during KV cache initialization with:

```text
RuntimeError: Worker failed with error 'All drafting layers should belong to the same kv cache group'
```

Interpretation: disabling draft sharding is not currently a runnable acceptance diagnostic in this fork. The failure occurs after capacity accounting but before inference, so it gives no MTP2 acceptance data. It does show that the draft/index KV grouping path is fragile and tightly coupled to the DCP draft-sharding implementation. The next useful code-level target is the cache grouping invariant that emits `All drafting layers should belong to the same kv cache group`, plus the Step3.5 iterative DCP metadata refresh path.

## 2026-06-26: Production restore after draft-replication diagnostic

Restored frozen production profile `glm52-mtp1-dcp4-128k` after the failed draft-replication diagnostic. The server started successfully on `http://192.168.100.1:18089/v1` with `max_model_len=131072`, `decode_context_parallel_size=4`, explicit `kv_cache_memory_bytes=1810000000`, fp8 KV, `B12X_MLA_SPARSE` attention, `flashinfer_cutlass` MoE, and MTP1.

Capacity on restore remained the known-good value:

```text
GPU KV cache size: 132,096 tokens
Maximum concurrency for 131,072 tokens per request: 1.01x
Model loading took 106.93 GiB memory
init engine/profile/KV/warmup took 180.84 s, compilation 43.98 s
```

## 2026-06-26: Generic GLM MTP proposer DCP refresh patch

The previous Step3.5-specific patch did not target the active GLM-5.2 MTP path. Runtime logs show GLM uses the generic `llm_base_proposer.py` MTP path (`DeepSeekMTPModel`), not `Step3p5MTPProposer`. The actual patch therefore moved the DCP-local sequence-length refresh into `vllm/v1/spec_decode/llm_base_proposer.py` and updated the overlay builder to copy `llm_base_proposer.py` into the image.

Built image: `glm-darkdevotion-b12x:20260626-arm64-mtp-baseiterfix1`.

MTP2 DCP4 128K launched successfully with the patched generic proposer:

```text
GPU KV cache size: 132,096 tokens
Maximum concurrency for 131,072 tokens per request: 1.01x
```

512-token codegen timing:

```text
run 1: 512 tokens / 48.748s = 10.503 tok/s
run 2: 512 tokens / 33.486s = 15.290 tok/s
```

Spec-decode hot-window metrics improved materially versus the earlier MTP2 attempt. Observed windows included:

```text
Mean acceptance length: 2.01-2.33
Per-position acceptance examples: 0.740/0.274, 0.849/0.479, 0.797/0.406
Avg generation throughput windows: 15.4, 16.4, 17.0 tok/s
```

Interpretation: the generic DCP-local metadata refresh is relevant and improves MTP2. This does not yet prove MTP3, but it eliminates the earlier mistaken conclusion from the Step3.5-only patch.

## 2026-06-26 MTP3 result with generic proposer DCP refresh

Image: `glm-darkdevotion-b12x:20260626-arm64-mtp-baseiterfix1`
Config: `glm52-mtp3-dcp4-128k`, TP4/DCP4, fp8 KV, B12X MLA sparse attention, FlashInfer CUTLASS MoE, `max_model_len=131072`, `kv_cache_memory_bytes=1810000000`, `max_cudagraph_capture_size=4`.

Capacity still fits at 128K:

```text
GPU KV cache size: 132,096 tokens
Maximum concurrency for 131,072 tokens per request: 1.01x
```

512-token codegen benchmark against `/v1/completions`:

```text
run 1: 512 tokens / 52.723s = 9.711 tok/s
run 2: 512 tokens / 45.045s = 11.366 tok/s
```

Representative MTP3 acceptance windows:

```text
Mean acceptance length: 1.67, per-position: 0.458, 0.167, 0.042
Mean acceptance length: 2.05, per-position: 0.603, 0.333, 0.111
Mean acceptance length: 2.52, per-position: 0.770, 0.557, 0.197
Mean acceptance length: 2.63, per-position: 0.850, 0.533, 0.250
Mean acceptance length: 1.66, per-position: 0.532, 0.113, 0.016
Mean acceptance length: 1.70, per-position: 0.524, 0.159, 0.016
Mean acceptance length: 2.02, per-position: 0.726, 0.274, 0.016
Mean acceptance length: 2.07, per-position: 0.674, 0.326, 0.070
```

Interpretation: the generic-proposer DCP refresh patch appears to have fixed MTP2 materially, but MTP3 is still worse than MTP1/MTP2. The third draft position is usually too weak to pay for the third sequential MTP forward, and the second/third positions show enough instability that there may still be a remaining iterative metadata or index-sharing issue beyond draft step 1.

Current production comparison points:

```text
MTP1 DCP4 128K: ~14.5-15.2 tok/s production band
MTP2 DCP4 128K after baseiterfix1: hot 512-token run 15.290 tok/s; healthy second-position acceptance up to ~0.48 in best windows
MTP3 DCP4 128K after baseiterfix1: hot 512-token run 11.366 tok/s; third-position acceptance often ~0.016-0.11, best observed ~0.25
```

## 2026-06-26 MTP3 top-k step-2 recompute diagnostic

Configuration: `glm52-mtp3-dcp4-128k`, image `glm-darkdevotion-b12x:20260626-arm64-mtp-topkstep2`, `VLLM_MTP_RECOMPUTE_TOPK_FROM_STEP=2`, TP4/DCP4, `max_model_len=131072`, `kv_cache_memory_bytes=1810000000`, fp8 KV, B12X MLA sparse attention, FlashInfer CUTLASS MoE.

Launch result: healthy. vLLM reported `GPU KV cache size: 132,096 tokens` and `Maximum concurrency for 131,072 tokens per request: 1.01x`. The `/v1/models` endpoint reported `max_model_len=131072`.

Benchmark result on the standard 512-token codegen prompt:

```text
run 1: 512 tokens / 57.927 s = 8.839 tok/s
run 2: 512 tokens / 45.390 s = 11.280 tok/s
```

Spec-decoding windows after the benchmark still show weak third-position acceptance, despite recomputing sparse top-k for the third draft step:

```text
Mean acceptance length 1.74, per-position 0.571, 0.171, 0.000
Mean acceptance length 1.87, per-position 0.629, 0.210, 0.032
Mean acceptance length 2.18, per-position 0.806, 0.306, 0.065
Mean acceptance length 2.48, per-position 0.852, 0.443, 0.180
Mean acceptance length 2.17, per-position 0.741, 0.328, 0.103
Mean acceptance length 1.70, per-position 0.540, 0.159, 0.000
Mean acceptance length 2.00, per-position 0.667, 0.222, 0.111
Mean acceptance length 1.67, per-position 0.526, 0.123, 0.018
Mean acceptance length 2.31, per-position 0.796, 0.407, 0.111
```

Interpretation: recomputing sparse top-k only for draft step 2 does not explain the MTP3 cliff. It neither improves throughput over the prior MTP3 baseline nor makes third-token acceptance consistently healthy. The current best production point remains MTP1 at 128K (`~14.5-15.2 tok/s`), with MTP2 also plausible after the DCP-local metadata refresh patch (`~15.29 tok/s` hot in one 512-token run). MTP3 remains not production-worthy on this stack.

## 2026-06-26 DCP1/MTP3 32K acceptance comparison

Purpose: test whether MTP3 weakness is intrinsic/economic or caused by DCP-sharded draft/index behavior. The comparison used the same image and top-k diagnostic patch as the DCP4 control: `glm-darkdevotion-b12x:20260626-arm64-mtp-topkstep2`, `VLLM_MTP_RECOMPUTE_TOPK_FROM_STEP=1`, MTP3, fp8 KV, B12X MLA sparse attention, FlashInfer CUTLASS MoE, `max_num_batched_tokens=1024`, `max_cudagraph_capture_size=4`, explicit `kv_cache_memory_bytes=1810000000`.

DCP4 control at 128K:

```text
served model: glm52-mtp3-dcp4-128k
DCP_SIZE=4
max_model_len=131072
GPU KV cache size: 132,096 tokens
512-token codegen: 9.351 tok/s cold-ish, 11.712 tok/s hot
Representative acceptance windows:
  1.86 mean, per-position 0.622, 0.216, 0.027
  1.88 mean, per-position 0.625, 0.188, 0.062
  2.06 mean, per-position 0.762, 0.238, 0.063
  2.27 mean, per-position 0.812, 0.328, 0.125
  1.60 mean, per-position 0.508, 0.095, 0.000
  1.79 mean, per-position 0.540, 0.206, 0.048
  2.05 mean, per-position 0.688, 0.344, 0.016
```

DCP1 comparison at 32K:

```text
served model: glm52-mtp3-dcp1-32k
DCP_SIZE=1
max_model_len=32768
GPU KV cache size: 33,024 tokens
Maximum concurrency for 32,768 tokens/request: 1.01x
512-token codegen: 12.218 tok/s cold-ish, 23.287 tok/s hot
Representative acceptance windows:
  2.94 mean, per-position 0.846, 0.692, 0.404
  3.29 mean, per-position 0.872, 0.769, 0.654
  3.13 mean, per-position 0.845, 0.732, 0.549
  3.29 mean, per-position 0.947, 0.750, 0.592
  3.46 mean, per-position 0.951, 0.854, 0.659
```

Interpretation: this strongly implicates DCP-sharded draft/index behavior as the MTP3 cliff. With DCP1, MTP3 is economically excellent and third-position acceptance is often above 0.55. With DCP4, third-position acceptance usually collapses below 0.13, even with top-k recomputed from draft step 1. This points away from sparse top-k reuse as the primary cause and toward DCP-local metadata, DCP draft cache/index grouping, DCP collective/top-k behavior in the draft path, or an MTP verifier/draft path that becomes pathological under DCP4.

Local argmax note: the current runs use `draft_sample_method=probabilistic` and source default `use_local_argmax_reduction=False`; no runtime logs show local argmax enabled. Do not enable local argmax without porting/validating the known multi-step fix that passes `spec_step_idx` correctly.

Next highest-value implementation target: keep target model/cache DCP4 for 128K, but replicate layer-78 MTP draft/index cache or otherwise make the draft path DCP1/local while preserving target DCP4. The prior `VLLM_DCP_SHARD_DRAFT=0` toggle failed with a KV grouping invariant, so the generic proposer/cache-group path needs actual mixed target/draft group support rather than just flipping the env var.

## 2026-06-26 draft-replicated MLA grouping diagnostic

Patched `/home/matt/code/vllm-dark-devotion/vllm/v1/core/kv_cache_utils.py` so `dcp_replicated=True` `MLAAttentionSpec` entries are split into their own replicated KV cache group before the generic MLA grouping branch catches them. This targets GLM-5.2 MTP/indexer draft caches with `VLLM_DCP_SHARD_DRAFT=0`: target layers should remain DCP4-sharded while the MTP draft/index cache is full-context local on every DCP rank.

Built overlay image `glm-darkdevotion-b12x:20260626-arm64-draftrepmla1` from `glm-darkdevotion-b12x:20260626-arm64-mtp-topkstep2`. Started diagnostic env `glm52-mtp3-dcp4-128k-draftrepmla1.env`: TP4/DCP4/MTP3, `max_model_len=131072`, `kv_cache_dtype=fp8`, `kv_cache_memory_bytes=1900000000`, `VLLM_DCP_SHARD_DRAFT=0`, `KZ_KV_DIAG=1`. As of the first monitoring pass the launch is not deadlocked; Ray reserved all 4 GPUs and TP0 is loading checkpoint shards normally.

## 2026-06-26 MTP2 draft-replicated MLA grouping result

Configuration: `glm52-mtp2-dcp4-128k-draftrepmla1-gmu91`, image `glm-darkdevotion-b12x:20260626-arm64-draftrepmla3`, TP4/DCP4/MTP2, `max_model_len=131072`, `kv_cache_dtype=fp8`, `kv_cache_memory_bytes=1900000000`, `GPU_MEMORY_UTILIZATION=0.91`, B12X MLA sparse attention, FlashInfer CUTLASS MoE, `VLLM_DCP_SHARD_DRAFT=0`, `VLLM_DCP_GLOBAL_TOPK=1`, `VLLM_MTP_RECOMPUTE_TOPK_FROM_STEP=1`.

The prior apparent MTP2 stall was a startup free-memory guard at GMU 0.915, not a deadlock. With GMU 0.91 this run passed startup, loaded, initialized KV, and served normally. Capacity matched the MTP3 draft-replicated run:

```text
GPU KV cache size: 133,006 tokens
Maximum concurrency for 131,072 tokens per request: 1.01x
```

KV grouping on every worker is now the intended mixed target/draft shape:

```text
group 0: 99 layers, page_size_bytes=3452160, max_required_pages=512, total_max_memory_bytes=1767505920, dcp_replicated=False
group 1: model.layers.78.self_attn.indexer.k_cache, page_size_bytes=8448, max_required_pages=2048, total_max_memory_bytes=17301504, dcp_replicated=True
group 2: model.layers.78.self_attn.attn, page_size_bytes=41984, max_required_pages=2048, total_max_memory_bytes=85983232, dcp_replicated=True
```

Standard 512-token codegen benchmark:

```text
run 1: 512 tokens / 54.297 s = 9.430 tok/s
run 2: 512 tokens / 37.652 s = 13.598 tok/s
```

Representative server-side decode/acceptance windows:

```text
Avg generation throughput 14.7 tok/s; mean acceptance length 2.31; per-position 0.812, 0.500
Avg generation throughput 15.4 tok/s; mean acceptance length 2.37; per-position 0.877, 0.492
Avg generation throughput 12.8 tok/s; mean acceptance length 2.06; per-position 0.694, 0.371
Avg generation throughput 11.6 tok/s; mean acceptance length 1.98; per-position 0.707, 0.276
Avg generation throughput 13.9 tok/s; mean acceptance length 2.21; per-position 0.762, 0.444
Avg generation throughput 14.7 tok/s; mean acceptance length 2.30; per-position 0.734, 0.562
Avg generation throughput 13.6 tok/s; mean acceptance length 2.16; per-position 0.730, 0.429
```

Interpretation: replicated layer-78 draft/index KV fixes the capacity/grouping issue and MTP2 is healthy enough to serve at 128K, but it still does not reliably exceed the frozen MTP1 production band (`~14.5-15.2 tok/s`) in client-measured 512-token runs. The second draft position accepts far better than DCP4/MTP3's third position, but the extra MTP forward still seems roughly break-even to slightly negative under DCP4/128K. Since DCP1/MTP3 at 32K showed much stronger second and third position acceptance, the remaining target is still the DCP4 iterative draft path: DCP-local metadata, draft slot mapping, index/top-k consistency, or verifier/draft disagreement after the first speculative step.

## 2026-06-26 MTP3 per-group slot-mapping patch result

Patched generic MTP (`llm_base_proposer.py`) to support multiple draft KV groups similarly to Step3.5: it now records per-group block tables/slot mappings from the runner, builds per-group draft attention metadata with the correct group-specific block table, and recomputes slot mappings for non-primary draft groups during iterative draft steps. Patched `gpu_model_runner.py` to pass per-group metadata to any proposer implementing `set_per_group_attn_metadata`, not only `Step3p5MTPProposer`.

Built `glm-darkdevotion-b12x:20260626-arm64-mtpgroups1` from `glm-darkdevotion-b12x:20260626-arm64-draftrepmla3` and launched `glm52-mtp3-dcp4-128k-mtpgroups1`: TP4/DCP4/MTP3, `max_model_len=131072`, fp8 KV, B12X MLA sparse attention, FlashInfer CUTLASS MoE, `kv_cache_memory_bytes=1900000000`, `VLLM_DCP_SHARD_DRAFT=0`, `VLLM_MTP_RECOMPUTE_TOPK_FROM_STEP=1`.

Startup passed. During the long startup, py-spy showed normal progression from safetensors load into TorchInductor/Triton autotune inside `llm_base_proposer.dummy_run`, not a deadlock. Capacity remained correct:

```text
GPU KV cache size: 133,006 tokens
Maximum concurrency for 131,072 tokens per request: 1.01x
```

Standard 512-token codegen benchmark:

```text
run 1: 512 tokens / 60.280 s = 8.494 tok/s
run 2: 512 tokens / 42.214 s = 12.129 tok/s
```

Representative acceptance windows after the patch:

```text
Mean acceptance length 1.93, per-position 0.561, 0.293, 0.073
Mean acceptance length 1.69, per-position 0.564, 0.127, 0.000
Mean acceptance length 2.20, per-position 0.786, 0.286, 0.125
Mean acceptance length 2.67, per-position 0.824, 0.608, 0.235
Mean acceptance length 1.88, per-position 0.519, 0.231, 0.135
Mean acceptance length 1.89, per-position 0.649, 0.193, 0.053
Mean acceptance length 2.15, per-position 0.796, 0.315, 0.037
Mean acceptance length 2.45, per-position 0.855, 0.455, 0.145
Mean acceptance length 2.58, per-position 0.846, 0.500, 0.231
```

Interpretation: the per-group slot-mapping bug was real enough to fix, but it is not the dominant cause of the MTP3 cliff. MTP3 remains below MTP1 and below the DCP1/MTP3 behavior. The next suspected surface is that replicated draft KV groups may still be building indexer/attention metadata with DCP-local sequence lengths (`dcp_local_seq_lens`) despite the draft cache being replicated/full-context. If so, layer-78 replicated draft attention/indexer would have correct slots but an incorrectly sharded sequence-length/top-k view.

## 2026-06-26: MTP3/DCP4 `mtpgroups2` negative result

Config/image: `glm52-mtp3-dcp4-128k-mtpgroups2` on `glm-darkdevotion-b12x:20260626-arm64-mtpgroups2`.

Additional change vs `mtpgroups1`: B12X/MLA indexer DCP-local path was suppressed for replicated KV specs by gating `use_dcp_local_kv` on `not kv_cache_spec.dcp_replicated`.

Capacity stayed healthy: GPU KV cache size reported `133,006 tokens`; max concurrency for `131,072` tokens reported `1.01x`.

512-token codegen benchmark:

- run 1: `512 tokens / 63.472s = 8.067 tok/s`
- run 2: `512 tokens / 50.778s = 10.083 tok/s`

Representative server-side speculative metrics:

- mean acceptance `1.62`, per-position `0.460, 0.120, 0.040`
- mean acceptance `2.23`, per-position `0.750, 0.375, 0.107`
- mean acceptance `2.51`, per-position `0.811, 0.491, 0.208`
- mean acceptance `2.32`, per-position `0.830, 0.362, 0.128`
- mean acceptance `1.55`, per-position `0.396, 0.113, 0.038`
- mean acceptance `2.20`, per-position `0.667, 0.370, 0.167`
- mean acceptance `2.00`, per-position `0.673, 0.286, 0.041`
- mean acceptance `2.11`, per-position `0.711, 0.267, 0.133`
- mean acceptance `2.02`, per-position `0.704, 0.222, 0.093`
- mean acceptance `1.77`, per-position `0.525, 0.200, 0.050`

Interpretation: this patch did not explain the MTP3 cliff and appears worse than `mtpgroups1`. Do not treat `mtpgroups2` as a production candidate. The frozen production point remains DCP4/MTP1 at 128K, roughly `14.5-15.2 tok/s`.

## 2026-06-26: DCP4/MTP3 32K control using `mtpgroups1`

Purpose: pair against the earlier DCP1/MTP3 32K result and isolate whether the MTP3 cliff is caused by 128K max length/cache shape or by DCP4 iterative draft behavior.

Config/image: `glm52-mtp3-dcp4-32k-mtpgroups1` on `glm-darkdevotion-b12x:20260626-arm64-mtpgroups1`.

Important settings:

- `DCP_SIZE=4`
- `MAX_MODEL_LEN=32768`
- `NUM_SPECULATIVE_TOKENS=3`
- `KV_CACHE_MEMORY_BYTES=1900000000`
- `KV_CACHE_DTYPE=fp8`
- `ATTENTION_BACKEND=B12X_MLA_SPARSE`
- `MOE_BACKEND=flashinfer_cutlass`
- `VLLM_DCP_GLOBAL_TOPK=1`
- `VLLM_DCP_SHARD_DRAFT=0`
- `VLLM_MTP_RECOMPUTE_TOPK_FROM_STEP=1`

Capacity:

- GPU KV cache size: `132,517 tokens`
- Maximum concurrency for `32,768` tokens/request: `4.04x`

512-token codegen benchmark:

- run 1: `512 tokens / 64.109s = 7.986 tok/s`
- run 2: `512 tokens / 51.421s = 9.957 tok/s`

Representative speculative windows:

- mean acceptance `1.92`, per-position `0.600, 0.240, 0.080`
- mean acceptance `1.94`, per-position `0.679, 0.245, 0.019`
- mean acceptance `2.57`, per-position `0.868, 0.491, 0.208`
- mean acceptance `2.22`, per-position `0.700, 0.360, 0.160`
- mean acceptance `1.78`, per-position `0.457, 0.239, 0.087`
- mean acceptance `2.21`, per-position `0.736, 0.340, 0.132`
- mean acceptance `1.86`, per-position `0.588, 0.216, 0.059`
- mean acceptance `1.87`, per-position `0.615, 0.173, 0.077`
- mean acceptance `1.92`, per-position `0.635, 0.231, 0.058`
- mean acceptance `1.83`, per-position `0.587, 0.174, 0.065`

Interpretation: shrinking max context from 128K to 32K does not recover DCP1-like MTP3 behavior. Earlier DCP1/MTP3 32K produced much stronger acceptance, including windows around mean `3.29-3.46` with position-3 acceptance around `0.59-0.66`, and much higher hot throughput. This makes 128K cache shape/max-length an unlikely primary cause. The remaining target is DCP4-specific iterative MTP behavior: metadata/slot correctness not yet found, DCP/global-top-k semantics, or the raw cost of two extra DCP draft forwards over the network.

## 2026-06-26: benchmark hygiene issue - prefix cache contamination

The first standardized 16K matrix attempt ran `DCP1/MTP0` and `DCP1/MTP1` sequentially with vLLM prefix caching enabled. Those runs should not be used as clean A/B matrix data.

Evidence: `DCP1/MTP1` reported GPU KV cache usage around `84-87%` during the ~9.7K-token resident-context prompt even though the model reported `34,658` KV tokens of capacity. A single ~10K-token request should be roughly 29-30% of that capacity, matching the `DCP1/MTP0` long-prompt run. The inflated usage is consistent with completed prefix blocks from earlier warmup/short prompts being retained by prefix caching.

Action: `serve.sh` now exposes `ENABLE_PREFIX_CACHING=0`, which adds `--no-enable-prefix-caching`. `run-glm52-mtp-dcp-matrix.sh` now forces `ENABLE_PREFIX_CACHING=0` in generated matrix env files. Fresh matrix results should use a new result directory and should not compare against the prefix-cache-contaminated `/tmp/glm52-mtp-dcp-matrix-20260626-014704` rows except as qualitative evidence.

## 2026-06-26: DCP1 / 16K / production prefix cache / zero-context slice

Purpose: isolate baseline MTP economics without DCP sharding, while preserving production prefix-cache semantics. Each measured MTP level used a fresh deployment, so prefix-cache state could not cross-contaminate rows. The measured prompt was the fixed zero-context kanban prompt.

Results so far:

| DCP | MTP tokens | prompt | completion | elapsed | client tok/s | capacity |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| 1 | 0 | zero_ctx_kanban | 512 | 43.123s | 11.873 | 35,200 KV tokens / 2.15x @ 16K |
| 1 | 1 | zero_ctx_kanban | 512 | 32.601s | 15.705 | 34,658 KV tokens / 2.12x @ 16K |
| 1 | 2 | zero_ctx_kanban | 512 | 27.020s | 18.949 | 34,658 KV tokens / 2.12x @ 16K |

MTP2 server-side short-context windows:

- generation throughput `19.5 tok/s`, mean acceptance `2.67`, per-position `0.932, 0.740`
- generation throughput `19.7 tok/s`, mean acceptance `2.77`, per-position `0.944, 0.831`

Interpretation: under DCP1 and short-context decode, MTP is behaving economically and monotonically through MTP2: `MTP0 < MTP1 < MTP2`. This makes a generic MTP implementation failure less likely. The DCP4 MTP3 issue remains likely tied to DCP degree, long-context/resident-cache behavior, or the third draft forward's economics under distributed verification.

## 2026-06-26 DCP1/MTP3 16K control with prefix caching enabled

This run used the production-default prefix-caching setting (`ENABLE_PREFIX_CACHING=1`), `DCP=1`, `TP=4`, `max_model_len=16384`, `kv_cache_dtype=fp8`, `B12X_MLA_SPARSE`, `flashinfer_cutlass` MoE, and probabilistic Step3.5 MTP draft sampling.

Standardized zero-context results now show MTP is monotonic under `DCP=1`:

| Config | Prompt | Completion | Elapsed | tok/s |
|---|---:|---:|---:|---:|
| DCP1/MTP2/16K | 37 | 512 | 27.020s | 18.949 |
| DCP1/MTP3/16K | 37 | 512 | 23.305s | 21.970 |

DCP1/MTP3 zero-context server windows showed healthy iterative acceptance:

```text
Mean acceptance length: 2.82; per-position acceptance: 0.839, 0.565, 0.419
Mean acceptance length: 3.63; per-position acceptance: 0.985, 0.877, 0.769
Mean acceptance length: 3.45; per-position acceptance: 0.968, 0.839, 0.645
```

This is strong evidence that the Step3.5 MTP iterative path is not intrinsically broken. The DCP4/MTP3 collapse is therefore likely tied to DCP-specific state, synchronization, local/global index mapping, or token/probability gathering behavior rather than generic MTP economics alone.

The same DCP1/MTP3 run on the resident 12K prompt was much weaker:

| Config | Prompt | Completion | Elapsed | tok/s |
|---|---:|---:|---:|---:|
| DCP1/MTP3/16K resident_12k_summary | 9667 | 512 | 85.910s | 5.960 |

Resident-window acceptance collapsed after prefill:

```text
Mean acceptance length: 1.08; per-position acceptance: 0.078, 0.000, 0.000
Mean acceptance length: 1.06; per-position acceptance: 0.063, 0.000, 0.000
Mean acceptance length: 1.14; per-position acceptance: 0.140, 0.000, 0.000
Mean acceptance length: 1.16; per-position acceptance: 0.143, 0.016, 0.000
Mean acceptance length: 1.22; per-position acceptance: 0.190, 0.032, 0.000
Mean acceptance length: 1.03; per-position acceptance: 0.033, 0.000, 0.000
Mean acceptance length: 1.15; per-position acceptance: 0.145, 0.000, 0.000
```

Interpretation: zero-context DCP1 validates MTP3 and makes the DCP4 collapse suspicious, but long-context MTP acceptance is also poor even at DCP1 for this resident prompt. That means we should separate two issues:

1. DCP-specific MTP2/MTP3 correctness or synchronization collapse, visible even at 32K/zero-context for DCP4.
2. Long-context acceptance degradation, visible at DCP1/12K and not sufficient by itself to explain the DCP4 zero-context problem.

A plausible DCP correctness hypothesis is that later MTP draft steps are accidentally making rank-local choices or using rank-local target/draft probabilities/indices. Correct behavior should not be a vote across DCP ranks: draft token ids and target probabilities must be global and rank-consistent before rejection sampling. If token ids, target argmax, or sampled draft distributions differ by DCP rank, acceptance would appear as a catastrophic requirement that multiple local views accidentally agree.

## 2026-06-26: DCP4 MTP3 probability diagnostic (`mtpdiag4`)

Built `glm-darkdevotion-b12x:20260626-arm64-mtpdiag4` with an env-gated probability diagnostic in the active sampler path, `vllm/v1/sample/rejection_sampler.py`. This logs, per DCP rank, the exact draft token ids plus local target probability, draft probability, and `min(1, target_p / draft_p)` for the proposed draft tokens.

DCP4 / MTP3 / 16K launched successfully with prefix caching enabled, fp8 KV, B12X MLA sparse, and FlashInfer CUTLASS MoE. Capacity was unchanged from the prior diagnostic profile:

```text
GPU KV cache size: 132,517 tokens
Maximum concurrency for 16,384 tokens per request: 8.09x
```

The 128-token zero-context codegen request completed cold-ish at about `4.02 tok/s`. Speculative metrics during the request remained poor:

```text
Mean acceptance length: 1.26
Accepted: 6 tokens
Drafted: 69 tokens
Per-position acceptance rate: 0.217, 0.043, 0.000
Avg Draft acceptance rate: 8.7%
```

The important diagnostic result: all four DCP ranks logged identical draft tokens, target probabilities, draft probabilities, acceptance probabilities, and final output rows for the sampled cycles. Example patterns from cycles 2-5:

```text
cycle 2 draft=[304,2038,323] target_p=[0,0,0] draft_p=[0.958,0.966,0.999995] accept_p=[0,0,0]
cycle 4 draft=[323,311,264] target_p=[0.867,0,0] draft_p=[0.875,0.959,0.789] accept_p=[0.991,0,0]
cycle 5 draft=[13,323,323] target_p=[0.0759,0,0] draft_p=[0.686,0.9998,0.993] accept_p=[0.111,0,0]
```

Interpretation: this run does not support the simple hypothesis that four DCP ranks independently make different rejection decisions and the system takes the min/intersection. The ranks agree at the local probability level, not only after final output synchronization. The stronger signal is that DCP4/MTP3 draft tokens are frequently very high-probability under the draft model but effectively zero-probability under the target verifier, especially at positions 2 and 3. That points to draft/target context disagreement, stale iterative draft metadata, wrong slot/position/KV state, or sampling-constraint mismatch rather than rank-wise random acceptance intersection.

Next high-signal comparison is the same `mtpdiag4` probability diagnostic under DCP1/MTP3 at 16K with the same prompt. If DCP1 shows target probabilities aligning with draft probabilities on the same prompt, the bug remains DCP-specific but is more likely draft-context/metadata corruption than rejection synchronization.

## 2026-06-26: DCP1 MTP3 probability control (`mtpdiag4`)

Ran the same `mtpdiag4` probability diagnostic under DCP1 / MTP3 / 16K with the identical zero-context codegen prompt. Capacity matched earlier DCP1 observations:

```text
GPU KV cache size: 34,658 tokens
Maximum concurrency for 16,384 tokens per request: 2.12x
```

The 128-token request completed at `7.36 tok/s` cold-ish, materially faster than the DCP4 diagnostic run. Speculative metrics were healthy:

```text
Mean acceptance length: 3.12
Accepted: 55 tokens
Drafted: 78 tokens
Per-position acceptance rate: 0.885, 0.692, 0.538
Avg Draft acceptance rate: 70.5%
```

The probability diagnostic clearly separates DCP1 from DCP4. In DCP1, many draft cycles have target and draft probabilities aligned, including full three-token accepts:

```text
cycle 6  draft=[7082,488,3592] target_p=[1.0,0.867,1.0] draft_p=[1.0,1.0,0.480] accept_p=[1.0,0.867,1.0]
cycle 7  draft=[6850,8750,271] target_p=[1.0,0.223,1.0] draft_p=[1.0,0.182,1.0] accept_p=[1.0,1.0,1.0]
cycle 9  draft=[6850,4479,448] target_p=[1.0,1.0,0.924] draft_p=[1.0,0.999,0.999] accept_p=[1.0,1.0,0.924]
cycle 10 draft=[7082,19127,323] target_p=[1.0,1.0,1.0] draft_p=[1.0,0.818,0.981] accept_p=[1.0,1.0,1.0]
```

This makes the DCP4 result diagnostic rather than merely a prompt/model weakness. The same model, same MTP3 setting, same prompt, same top-p/temperature, and same diagnostic image produce healthy draft/target agreement under DCP1 but frequent near-zero verifier probability under DCP4.

Updated conclusion: DCP4 is corrupting or misaligning the MTP draft context/logits before rejection sampling. The rank-intersection hypothesis is mostly falsified for this run because all DCP ranks report identical local probabilities and identical final outputs. The failure is upstream of rejection synchronization: likely DCP-aware draft metadata, DCP-local sequence lengths, draft KV/cache position mapping, or sparse index/top-k state used by the MTP draft path.

The next code target should be the MTP draft forward path, not the rejection sampler. Specifically compare DCP1 vs DCP4 for the first actual request cycle:

- metadata positions and slot mappings passed to the draft model
- `dcp_local_seq_lens` and expected DCP-local seq lens
- draft model hidden-state/logit input rows for step 0 vs iterative steps
- sparse-index/top-k buffer reuse for MTP step 0/1/2
- whether the draft model uses DCP-sharded KV/index cache with the same local/global position convention as the target verifier

## 2026-06-26 - iterative MTP DCP slot-mapping patch

The DCP4/MTP3 probability diagnostics falsified the simple rank-wise rejection/intersection hypothesis: all DCP ranks logged matching draft tokens, target probabilities, draft probabilities, accept probabilities, and final output heads for the sampled cycles. The failure instead shows draft tokens with high draft probability and near-zero target probability under DCP4, while the same prompt under DCP1/MTP3 has healthy per-position acceptance.

The cross-rank `KZ_MTP_DCP_DIAG` logs identified a concrete mismatch. Draft step 0 receives DCP-sharded slot mappings such as rank-specific `[slot, -1, -1, -1]` patterns, matching the normal `BlockTable.compute_slot_mapping()` DCP behavior. Iterative draft steps 1/2 then use dense scalar slot mappings like `[165]`, `[166]` on every rank. That means non-owning DCP ranks write/read draft KV for global positions they should pad, corrupting or misaligning the iterative MTP context. A source patch was added to `vllm/v1/spec_decode/llm_base_proposer.py`: after the existing fused EAGLE position/seq-len update, it overwrites iterative draft slot mappings for non-`dcp_replicated` draft KV groups using the same DCP formula as `BlockTable.compute_slot_mapping()`.

### Correction: replicated-draft control was accidentally active

The first `mtpdiag5` launch still had `VLLM_DCP_SHARD_DRAFT=0` in the worker environment. That makes the draft KV group `dcp_replicated`, so the new DCP-sharded slot-mapping helper correctly skipped it and the logs continued to show dense scalar iterative slots. Result: coherent text, 128 tokens in 28.816s (`4.44 tok/s`), but this was not a valid test of the intended DCP-sharded draft-slot fix. Relaunching the same patched image with `VLLM_DCP_SHARD_DRAFT=1`.

## 2026-06-26 - DCP4/MTP2 and Step3.5 group-slot diagnostics

The valid `mtpdiag5` DCP4/MTP3 run with `VLLM_DCP_SHARD_DRAFT=1` improved materially versus the pre-slot-fix DCP4/MTP3 baseline, but remained far below the DCP1 control. Zero-context 128-token codegen completed in `23.716s` (`5.397 tok/s`). Server-side speculative metrics showed mean acceptance `2.19`, accepted `69`, drafted `174`, and per-position acceptance `0.776, 0.293, 0.121`.

The same patched image/config tested as DCP4/MTP2 at 16K completed 128 zero-context tokens in `22.292s` (`5.742 tok/s`). Effective per-position acceptance from counters was `45/69 = 0.652` for position 0 and `15/69 = 0.217` for position 1. This confirms the regression begins at the first iterative MTP step, not only at the third draft token.

A follow-up `mtpdiag6` image patched `Step3p5MTPProposer._update_positions_dependent_metadata()` so non-primary Step3.5 KV groups also receive DCP-sharded iterative slot mappings after the dense fallback writes. DCP4/MTP2 with that image completed 128 zero-context tokens in `24.411s` (`5.243 tok/s`), with effective acceptance `45/65 = 0.692` and `18/65 = 0.277`. The non-primary group-slot patch did not fix throughput or acceptance, so the next target is Step3.5 top-k reuse.

The active difference from the base proposer is now clear: `llm_base_proposer.py` honors `VLLM_MTP_RECOMPUTE_TOPK_FROM_STEP`, while `step3p5.py` unconditionally forced `skip_topk=True` for all iterative MTP forwards. With `VLLM_MTP_RECOMPUTE_TOPK_FROM_STEP=1`, Step3.5 should recompute sparse top-k from the first iterative draft step. A new patch changes Step3.5 to set `skip_topk = spec_step_idx < _mtp_recompute_topk_from_step` inside the iterative loop.

## 2026-06-26 - Step3.5 iterative top-k recompute result

Built `glm-darkdevotion-b12x:20260626-arm64-mtpdiag7-step3p5-recompute-topk` with a targeted Step3.5 patch: the iterative MTP loop now honors `_mtp_recompute_topk_from_step` instead of forcing `skip_topk=True` for every iterative draft forward. The test environment kept all other relevant controls constant: DCP4, TP4, 16K max model length, fp8 KV, `B12X_MLA_SPARSE`, `flashinfer_cutlass`, prefix caching enabled, `enforce_eager=False`, `VLLM_DCP_SHARD_DRAFT=1`, and `VLLM_MTP_RECOMPUTE_TOPK_FROM_STEP=1`.

Capacity stayed unchanged:

```text
GPU KV cache size: 138,752 tokens
Maximum concurrency for 16,384 tokens per request: 8.47x
```

DCP4/MTP2 with recomputed iterative top-k:

```text
128-token first probe: 28.691s, 4.461 tok/s
  accepted=68, drafts=60, per-position=42/60, 26/60

128-token hot probe: 9.704s, 13.190 tok/s
  accepted=59, drafts=68, per-position=41/68, 18/68

256-token hot probe: 16.575s, 15.445 tok/s
  accepted=137, drafts=120, per-position=91/120, 46/120
```

DCP4/MTP3 with recomputed iterative top-k:

```text
128-token first probe: 26.665s, 4.800 tok/s
  accepted=69, drafts=59, per-position=41/59, 17/59, 11/59

256-token hot probe: 21.200s, 12.075 tok/s
  accepted=130, drafts=126, per-position=80/126, 35/126, 15/126
```

Interpretation: the unconditional Step3.5 top-k reuse was a real DCP4/MTP2+ bug. Recomputing sparse top-k for iterative MTP steps removes the catastrophic DCP4 collapse and makes MTP2 competitive with the frozen DCP4/MTP1 production point. MTP3 is also fixed enough to produce nonzero third-position acceptance, but at this 16K zero-context batch-one profile it still underperforms because the third sequential MTP forward does not earn back its cost.

Current practical frontier:

```text
Production known-good: DCP4 / MTP1 / 128K / ~14.5-15.2 tok/s
Promising new candidate: DCP4 / MTP2 / 16K / 15.4 tok/s hot steady-state
Not currently attractive: DCP4 / MTP3 / 16K / 12.1 tok/s hot steady-state
```

Next production-relevance test is the same recompute-topk patch at 128K with MTP2. If that holds around or above the MTP1 production number, MTP2 becomes the likely new default. If long-context acceptance degrades materially, keep MTP1 for 128K and treat MTP2 as a short-context/high-throughput option.

## 2026-06-26 - 128K DCP4/MTP2 production-candidate result

Launched the recompute-topk patch at 128K using `glm-darkdevotion-b12x:20260626-arm64-mtpdiag7-step3p5-recompute-topk` and persisted the env as:

```text
/home/matt/code/blackwell-llm-docker/recipes/4x-spark-cluster/glm52-b12x-spark/glm52-dcp4-mtp2-128k-recompute-topk.env
```

Core config:

```text
MAX_MODEL_LEN=131072
TP_SIZE=4
DCP_SIZE=4
NUM_SPECULATIVE_TOKENS=2
KV_CACHE_DTYPE=fp8
KV_CACHE_MEMORY_BYTES=1900000000
ATTENTION_BACKEND=B12X_MLA_SPARSE
MOE_BACKEND=flashinfer_cutlass
ENABLE_PREFIX_CACHING=1
ENFORCE_EAGER=0
VLLM_DCP_SHARD_DRAFT=1
VLLM_MTP_RECOMPUTE_TOPK_FROM_STEP=1
NCCL_IB_DISABLE=0
```

Capacity:

```text
GPU KV cache size: 138,752 tokens
Maximum concurrency for 131,072 tokens per request: 1.06x
```

Zero-context codegen, same 256-token FastAPI/Pydantic/SQLite/SQLAlchemy/React prompt:

```text
run1 first request: 34.952s, 7.324 tok/s
  accepted=135, drafts=120, per-position=88/120, 47/120

run2 hot: 15.859s, 16.142 tok/s
  accepted=141, drafts=114, per-position=89/114, 52/114
```

Resident-context summary/analysis at about 11.4K prompt tokens:

```text
short summary run:
  prompt_tokens=11,425, completion_tokens=87
  32.551s end-to-end, 2.673 tok/s
  stopped naturally; not a decode-saturation measurement
  accepted=43, drafts=43, per-position=31/43, 12/43

forced 256-token analysis, cold/prefill-heavy:
  prompt_tokens=11,447, completion_tokens=256
  43.278s end-to-end, 5.915 tok/s
  accepted=107, drafts=148, per-position=73/148, 34/148

same forced analysis repeated with prefix cache:
  prompt_tokens=11,447, completion_tokens=256
  21.682s end-to-end, 11.807 tok/s
  accepted=110, drafts=147, per-position=75/147, 35/147
```

Interpretation: 128K DCP4/MTP2 with iterative top-k recompute is now the best observed production candidate. It preserves 128K capacity with `1.06x` concurrency, beats the frozen MTP1 zero-context production number on hot codegen (`16.1 tok/s` vs roughly `14.5-15.2 tok/s`), and remains usable at 11K resident context. Long-context MTP2 acceptance is lower than zero-context, especially at position 1, but it is not catastrophically broken.

Current recommendation:

```text
Promote candidate for production testing:
  DCP4 / MTP2 / 128K / recompute iterative top-k from step 1

Keep as rollback:
  DCP4 / MTP1 / 128K frozen known-good

Do not promote yet:
  DCP4 / MTP3, because third-position acceptance still does not amortize the extra forward
```

## 2026-06-26 mtpdiag8: DCP4/MTP3 recompute top-k only for iterative step 1

Image: `glm-darkdevotion-b12x:20260626-arm64-mtpdiag8-recompute-topk-window`
Model: `/var/tmp/models/Mapika/GLM-5.2-NVFP4-MTP-hybrid`
Served name: `glm52-mtpdiag8-dcp4-mtp3-16k-recompute-step1-only`
Runtime shape: TP4 / PP1 / DCP4 / MTP3, 16K max model len, fp8 KV, B12X MLA sparse attention, flashinfer_cutlass MoE, prefix caching enabled.

Patch under test: `VLLM_MTP_RECOMPUTE_TOPK_FROM_STEP=1` plus `VLLM_MTP_RECOMPUTE_TOPK_UNTIL_STEP=1`, so Step3.5 recomputes sparse top-k only for iterative draft step 1, then reuses top-k for draft step 2+.

Capacity from startup:

```text
GPU KV cache size: 138,752 tokens
Maximum concurrency for 16,384 tokens per request: 8.47x
```

Cold-ish 128-token codegen probe:

```text
completion_tokens: 128
elapsed: 24.837 s
tok/s: 5.154
drafts: 61
draft_tokens: 183
accepted_total: 66
accepted_per_pos: 41/61, 15/61, 10/61
```

Hot 256-token codegen probe:

```text
completion_tokens: 256
elapsed: 19.649 s
tok/s: 13.029
drafts: 118
draft_tokens: 354
accepted_total: 137
accepted_per_pos: 85/118, 39/118, 13/118
```

Interpretation: this is a small improvement over the prior DCP4/MTP3 hot 16K row (~12.08 tok/s), but still materially worse than DCP4/MTP2 with full iterative top-k recompute (~15.45 tok/s at 16K and ~16.14 tok/s at 128K). Third-position acceptance remains very low (~11% on the hot 256-token probe), so recomputing only step 1 does not make MTP3 viable. The strongest current explanation remains economic/architectural: this checkpoint exposes one MTP layer (`num_nextn_predict_layers=1`), so MTP2/MTP3 reuse the same one-step predictor; DCP4 then pays extra sparse top-k/DCP communication for later draft positions whose acceptance does not amortize the additional forward(s).

## 2026-06-26 DCP1/MTP3 control: MTP3 works when DCP is removed

Image: `glm-darkdevotion-b12x:20260626-arm64-mtpdiag8-recompute-topk-window`
Model: `/var/tmp/models/Mapika/GLM-5.2-NVFP4-MTP-hybrid`
Served name: `glm52-mtpdiag8-dcp1-mtp3-16k-full-recompute`
Runtime shape: TP4 / PP1 / DCP1 / MTP3, 16K max model len, fp8 KV, B12X MLA sparse attention, flashinfer_cutlass MoE, prefix caching enabled, full iterative top-k recompute from step 1.
Env snapshot saved as `glm52-dcp1-mtp3-16k-full-recompute.env`.

Startup/capacity:

```text
vLLM warning: Enabling num_speculative_tokens > 1 will run multiple times of forward on same MTP layer, which may result in lower acceptance rate
Target weight load: 924.52 s
GPU KV cache size: 34,688 tokens
Maximum concurrency for 16,384 tokens per request: 2.12x
init engine/profile/create KV/warmup: 166.31 s, compilation 45.31 s
```

Cold-ish 128-token codegen probe:

```text
completion_tokens: 128
elapsed: 19.398 s
tok/s: 6.599
drafts: 37
draft_tokens: 111
accepted_total: 91
accepted_per_pos: 36/37, 30/37, 25/37
```

Hot 256-token codegen probe:

```text
completion_tokens: 256
elapsed: 11.389 s
tok/s: 22.477
drafts: 82
draft_tokens: 246
accepted_total: 174
accepted_per_pos: 70/82, 56/82, 48/82
```

Interpretation: MTP3 is not intrinsically bad for this one-layer-MTP checkpoint. DCP1/MTP3 is much faster than the DCP4/MTP2 production candidate on the same 16K zero-context probe, and its third speculative position remains useful (~58.5% acceptance on the hot run). The DCP4/MTP3 failure is therefore DCP-specific: either DCP-local accept/reject decisions are being combined too conservatively, DCP-specific sparse/index/KV work makes the third step uneconomic, or both.

The acceptance ratios line up suspiciously with an all-DCP-ranks-must-accept/min-local-accepted-length effect. DCP1 hot acceptance was approximately:

```text
position 0: 70/82 = 0.854
position 1: 56/82 = 0.683
position 2: 48/82 = 0.585
```

Fourth powers are approximately:

```text
position 0: 0.854^4 = 0.532
position 1: 0.683^4 = 0.218
position 2: 0.585^4 = 0.117
```

Recent DCP4/MTP3 hot rows showed third-position acceptance around 11-12%, matching the `p^4` prediction closely. This strongly suggests the next patch target is the DCP speculative acceptance coordinator: for a single request, acceptance should be decided once from canonical/gathered target and draft probabilities, then broadcast/synchronized, not independently sampled on each DCP rank and reduced by intersection/min length.

Current recommendation remains:

```text
Best 128K production candidate now: DCP4 / MTP2 / 128K / full iterative top-k recompute
Best proof that MTP3 can work: DCP1 / MTP3 / 16K, but it lacks the 128K capacity target
Next engineering target: fix or bypass DCP4 MTP3 acceptance synchronization so DCP4 behaves closer to DCP1 acceptance economics
```

## 2026-06-26 mtpdiag9: broadcast rejection sampler output across DCP ranks

Patch: `VLLM_DCP_SYNC_REJECTION_OUTPUT=1` broadcasts `output_token_ids` from DCP rank 0 immediately after `rejection_sample(...)` in `vllm/v1/sample/rejection_sampler.py`.

Image: `glm-darkdevotion-b12x:20260626-arm64-mtpdiag9-sync-rejection-output`
Env snapshot: `glm52-dcp4-mtp3-16k-sync-rejection.env`
Runtime shape: TP4 / PP1 / DCP4 / MTP3, 16K, fp8 KV, B12X MLA sparse attention, flashinfer_cutlass MoE, full iterative top-k recompute.

Startup/capacity:

```text
GPU KV cache size: 138,752 tokens
Maximum concurrency for 16,384 tokens per request: 8.47x
init engine/profile/create KV/warmup: 179.09 s, compilation 43.36 s
```

Cold-ish 128-token codegen probe:

```text
completion_tokens: 128
elapsed: 27.415 s
tok/s: 4.669
drafts: 65
draft_tokens: 195
accepted_total: 62
accepted_per_pos: 41/65, 17/65, 4/65
```

Hot 256-token codegen probe:

```text
completion_tokens: 256
elapsed: 20.873 s
tok/s: 12.265
drafts: 122
draft_tokens: 366
accepted_total: 133
accepted_per_pos: 88/122, 35/122, 10/122
```

Result: broadcasting post-rejection `output_token_ids` did **not** improve DCP4/MTP3. It slightly regressed versus the prior DCP4/MTP3 diagnostic rows. This falsifies the simple theory that DCP4 was only combining independently sampled local accepted lengths after otherwise-good local accept/reject decisions.

More important diagnostic: rank0 probability logs already show severe target/draft disagreement before the broadcast point. Examples from the first measured hot run cycles include high-confidence draft tokens with target probability zero:

```text
cycle 3: draft=[2038,323,220], target_p=[0,0,0], draft_p=[0.912692,0.998647,0.985587]
cycle 4: draft=[11,323,264], target_p=[0,0,0], draft_p=[0.993306,0.998658,0.923082]
cycle 9: draft=[264,4583,2038], target_p=[1,1,0], draft_p=[0.952574,0.988917,0.180661]
cycle 21: draft=[12663,198,1499], target_p=[1,1,1], draft_p=[1,1,0.998891]
```

Interpretation: DCP4/MTP3 is not primarily failing because a good acceptance decision is later intersected across DCP ranks. It is already generating or verifying misaligned draft positions on the canonical rank. The remaining likely bug class is upstream of the rejection sampler: Step3.5 iterative MTP under DCP4 has wrong position/slot/DCP-local sequence state, stale sparse-index/top-k inputs, or target-logit alignment for draft steps 2/3. DCP1/MTP3 proves the MTP layer can be useful; DCP4/MTP2 proves one iterative step can be made useful. The failure boundary is specifically DCP4 plus the second iterative MTP step and beyond.

Current working recommendation is unchanged:

```text
Production candidate: DCP4 / MTP2 / 128K / full iterative top-k recompute
Rollback: DCP4 / MTP1 / 128K
Do not promote: DCP4 / MTP3 until Step3.5 DCP position/slot/top-k alignment is fixed
```

## 2026-06-26 mtpdiag10: DCP4/MTP3/16K with draft KV replication

Purpose: test whether the DCP4/MTP3 acceptance collapse is caused by DCP-sharded draft KV / non-local draft slot ownership. This run disables draft sharding with `VLLM_DCP_SHARD_DRAFT=0` while keeping DCP4, MTP3, fp8 KV, B12X MLA sparse attention, flashinfer_cutlass MoE, and full iterative top-k recompute.

Config snapshot:

```text
IMAGE=glm-darkdevotion-b12x:20260626-arm64-mtpdiag8-recompute-topk-window
MODEL_DIR=/var/tmp/models/Mapika/GLM-5.2-NVFP4-MTP-hybrid
SERVED_MODEL_NAME=glm52-mtpdiag10-dcp4-mtp3-16k-draftrep
TP_SIZE=4
DCP_SIZE=4
MAX_MODEL_LEN=16384
MAX_NUM_SEQS=1
MAX_NUM_BATCHED_TOKENS=1024
MAX_CUDAGRAPH_CAPTURE_SIZE=4
KV_CACHE_DTYPE=fp8
KV_CACHE_MEMORY_BYTES=1900000000
ENABLE_MTP=1
NUM_SPECULATIVE_TOKENS=3
ATTENTION_BACKEND=B12X_MLA_SPARSE
MOE_BACKEND=flashinfer_cutlass
VLLM_DCP_GLOBAL_TOPK=1
VLLM_DCP_SHARD_DRAFT=0
VLLM_MTP_RECOMPUTE_TOPK_FROM_STEP=1
ENABLE_PREFIX_CACHING=1
```

Startup/capacity:

```text
Target weight load: 921.18s
MTP overlay load: 13.42s
Model loading memory: 106.93 GiB
Initial free memory before manual KV: 111.88 GiB
Manual KV reservation: 1.77 GiB
GPU KV cache size: 132,517 tokens
Max concurrency at 16,384: 8.09x
Graph capture: 27s, 0.60 GiB
```

Throughput test after warmup, zero-context kanban prompt:

```text
warmup 64 tokens: 22.403s, 2.857 tok/s
kanban 128 tokens: 15.143s, 8.453 tok/s
kanban 256 tokens: 23.805s, 10.754 tok/s
```

Speculative metrics after the three calls:

```text
spec_decode_num_drafts_total: 242
spec_decode_num_draft_tokens_total: 726
spec_decode_num_accepted_tokens_total: 203
accepted position 0: 149 / 242 = 61.6%
accepted position 1: 46 / 242 = 19.0%
accepted position 2: 8 / 242 = 3.3%
accepted per draft token: 203 / 726 = 28.0%
```

Limited fork-side `KZ_MTP_REJECT_DIAG` logging captured 24 cycles:

```text
cycles: 24
draft tokens: 68
accepted draft tokens: 20
accepted per draft token: 29.4%
accepted per cycle histogram: {0: 10, 1: 10, 2: 2, 3: 2}
```

Interpretation: disabling DCP draft sharding did not fix DCP4/MTP3. It reduced logical KV capacity versus the sharded-draft 16K run (`132,517` vs `138,752`) and throughput remained worse than the DCP4/MTP2 production candidate. The position-2 acceptance cliff remained severe (`8/242`). This makes a simple "draft KV is non-local due to slot_mapping=-1" explanation unlikely by itself. The remaining likely bug class is DCP-specific iterative Step3.5 target/draft alignment, sparse top-k/index state, or sequence/position metadata used by the target verifier for speculative steps >=2.

## mtpdiag12: GLM skip_topk hook wired, DCP4/MTP3/16K

Image: `glm-darkdevotion-b12x:20260626-arm64-mtpdiag12-glm-skiptopk-hook`.
Served model: `glm52-mtpdiag12-dcp4-mtp3-16k-glm-skiptopk`.

Patch tested: added `Glm4MoeMultiTokenPredictor.set_skip_topk()` so Step3.5's `VLLM_MTP_RECOMPUTE_TOPK_FROM_STEP=1` can actually toggle GLM MTP sparse-attention `skip_topk` state. Broadcast draft tokens remained disabled.

Startup/capacity:

```text
Model loading took 106.93 GiB memory and 955.56 s
Initial free memory 112.11 GiB
Manual KV cache memory 1.77 GiB
GPU KV cache size 138,752 tokens
Maximum concurrency for 16,384 tokens/request: 8.47x
Graph capture took 0.64 GiB
```

Zero-context measurements, `ignore_eos` off, `temperature=0`:

```text
warmup primes 64: 64 tokens, 5.522 s, 11.590 tok/s, accepted 35/87 draft tokens, per-pos 18/29, 11/29, 6/29
kanban 128:       128 tokens, 12.097 s, 10.581 tok/s, accepted 67/180 draft tokens, per-pos 42/60, 19/60, 6/60
kanban 256:       256 tokens, 22.257 s, 11.502 tok/s, accepted 121/405 draft tokens, per-pos 75/135, 35/135, 11/135
```

Interpretation: wiring the GLM skip_topk hook did not recover DCP4/MTP3. Position-2 acceptance remains very weak, and throughput is below the earlier DCP4/MTP3 baseline. The next higher-signal test is canonicalizing draft token IDs across DCP ranks with `VLLM_MTP_BROADCAST_DRAFT_TOKENS=1`, now that the GLM hook is present.

## mtpdiag13: invalid broadcast-token test

`mtpdiag13` was intended to test `VLLM_MTP_BROADCAST_DRAFT_TOKENS=1`, but the flag was not present in either the API process or local worker environment. Root cause: `serve.sh` did not pass the variable through its `docker exec -e ...` allowlist. The run is still a repeat data point for the GLM skip_topk-hook case, but it is not evidence about draft-token broadcast.

Observed before invalidation, `ignore_eos` off, `temperature=0`:

```text
warmup primes 64: 64 tokens, 17.011 s, 3.762 tok/s, accepted 37/78 draft tokens, per-pos 22/26, 9/26, 6/26
kanban 128:       128 tokens, 11.008 s, 11.627 tok/s, accepted 67/186 draft tokens, per-pos 41/62, 20/62, 6/62
kanban 256:       256 tokens, 21.492 s, 11.911 tok/s, accepted 124/393 draft tokens, per-pos 85/131, 28/131, 11/131
```

Fix applied: added `VLLM_MTP_BROADCAST_DRAFT_TOKENS` default and `-e` propagation in `serve.sh`. Relaunching as `mtpdiag14` for the actual broadcast-token test.

## mtpdiag14: invalid worker-side broadcast-token test

`mtpdiag14` verified that `VLLM_MTP_BROADCAST_DRAFT_TOKENS=1` reached the API process, but it did not reach the local Ray worker environment. Root cause: `launch-ray.sh` also had a Ray-container env allowlist and omitted the flag. Since Step3.5/proposer code is worker-side for this path, the run is not a valid broadcast-token test.

Fix applied: added `VLLM_MTP_BROADCAST_DRAFT_TOKENS` default and `-e` propagation to `launch-ray.sh` for both head and remote worker Ray containers. Relaunching as `mtpdiag15`.

## mtpdiag15: valid worker-side draft-token broadcast, DCP4/MTP3/16K

Image: `glm-darkdevotion-b12x:20260626-arm64-mtpdiag12-glm-skiptopk-hook`.
Served model: `glm52-mtpdiag15-dcp4-mtp3-16k-glm-skiptopk-broadcast-worker`.

Patch/config tested:

```text
VLLM_MTP_RECOMPUTE_TOPK_FROM_STEP=1
VLLM_MTP_BROADCAST_DRAFT_TOKENS=1
VLLM_DCP_SHARD_DRAFT=1
VLLM_DCP_GLOBAL_TOPK=1
```

Validation: the broadcast flag was present in PID 1, the API process, and the actual Ray worker process. This is the first valid worker-side draft-token broadcast test.

Startup/capacity:

```text
Model loading took 106.93 GiB memory and 952.98 s
Initial free memory 112.08 GiB
Manual KV cache memory 1.77 GiB
GPU KV cache size 138,752 tokens
Maximum concurrency for 16,384 tokens/request: 8.47x
Graph capture took 0.63 GiB
```

Zero-context measurements, `ignore_eos` off, `temperature=0`:

```text
warmup primes 64: 64 tokens, 23.745 s, 2.695 tok/s, accepted 38/78 draft tokens, per-pos 22/26, 9/26, 7/26
kanban 128:       128 tokens, 9.835 s, 13.014 tok/s, accepted 71/168 draft tokens, per-pos 43/56, 20/56, 8/56
kanban 256:       256 tokens, 23.657 s, 10.821 tok/s, accepted 112/429 draft tokens, per-pos 88/143, 21/143, 3/143
```

Interpretation: canonicalizing/broadcasting draft token IDs across DCP ranks does not recover DCP4/MTP3. The 128-token row showed a small transient improvement, but the 256-token row regressed badly, especially at position 2. This argues against the simple “different DCP ranks emit different draft IDs” hypothesis as the primary issue. The remaining likely class is verifier/probability/index/slot metadata semantics across iterative MTP steps, not the sampled draft token ID itself.

### Cross-rank diagnostics from mtpdiag15

After the valid worker-side broadcast-token run, the same early cycles were compared across all four DCP ranks.

Findings:

```text
cycle 0 draft_stack on all ranks: [13, 15, 311]
cycle 1 draft_stack on all ranks: [3070, 16236, 2685]
cycle 2 draft_stack on all ranks: [39730, 25, 576]
cycle 3 draft_stack on all ranks: [576, 68, 46301]
cycle 4 draft_stack on all ranks: [374, 279, 353]
```

Rejection/probability diagnostics were also identical across ranks for sampled cycles. Example:

```text
cycle 3 all ranks:
  draft_head=[3070, 16236, 2685]
  target_p=[0.999995, 0.817263, 1.0]
  accepted_draft_counts=[3]

cycle 5 all ranks:
  draft_head=[576, 68, 46301]
  target_p=[0.999837, 0.0, 0.0]
  accepted_draft_counts=[1]
```

Interpretation:

The DCP4/MTP3 failure is not caused by divergent draft token IDs and is not an all-ranks-must-accept/intersection artifact. All ranks are making the same probability and rejection decision. The bad behavior is upstream: under DCP4, later iterative MTP positions frequently produce draft tokens whose target probabilities are near zero.

The remaining likely failure class is DCP-local metadata/cache semantics in iterative MTP: previous speculative token KV/index/slot state is not being made available to later MTP forward steps in the same way it is under DCP1. The logs show expected DCP sharding patterns where only one DCP rank owns a given speculative position and other ranks see `slot_mapping=-1`; if later MTP forwards require local access to previous speculative KV/index state, this can explain why MTP1 is healthy and MTP2/MTP3 degrade.

## 2026-06-26 mtpdiag16: DCP4/MTP3/16K with draft KV replication

Purpose: isolate whether the DCP4/MTP3 collapse is primarily caused by `VLLM_DCP_SHARD_DRAFT=1` owner-only draft KV rows.

Config:
- Image: `glm-darkdevotion-b12x:20260626-arm64-mtpdiag12-glm-skiptopk-hook`
- Env: `glm52-dcp4-mtp3-16k-glm-skiptopk-draftrep.env`
- Served model: `glm52-mtpdiag16-dcp4-mtp3-16k-glm-skiptopk-draftrep`
- TP4 / DCP4 / MTP3 / max_model_len 16384
- `VLLM_DCP_SHARD_DRAFT=0`
- `VLLM_MTP_RECOMPUTE_TOPK_FROM_STEP=1`
- `VLLM_MTP_BROADCAST_DRAFT_TOKENS=0`
- `NCCL_IB_DISABLE=0`, `NCCL_SOCKET_IFNAME=enP2p1s0f0np0`
- fp8 KV, `B12X_MLA_SPARSE`, `flashinfer_cutlass`

Startup/capacity:
- All four ranks loaded target+MTP weights successfully.
- Head rank model load: 106.93 GiB, 964.07s.
- Engine KV: 132,517 tokens, 8.09x concurrency at 16,384 tokens/request.
- `init engine` took 198.26s after model load, including 42.28s compilation.

Benchmark results, chat completions, temperature 0, no ignore_eos:

| Prompt | Max tokens | Seconds | Output tok/s | Drafts | Draft tokens | Accepted | Pos0 | Pos1 | Pos2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| primes warmup | 64 | 22.689 | 2.821 | 34 | 102 | 29 | 22 | 6 | 1 |
| kanban | 128 | 14.904 | 8.589 | 78 | 234 | 50 | 41 | 8 | 1 |
| kanban | 256 | 21.907 | 11.686 | 116 | 348 | 139 | 86 | 38 | 15 |

Interpretation:
- Draft KV replication helps relative to the worst valid broadcast run, especially on the 256-token row, but does not make MTP3 healthy.
- The third speculative position remains weak: 15/116 = 12.9% on the best row.
- Therefore the problem is not only divergent draft token IDs and not only non-owner ranks lacking draft KV rows. The remaining target is iterative DCP metadata/index/top-k state across MTP positions.

## 2026-06-26 mtpdiag17: Step3.5/GLM get_top_tokens local-argmax path

Purpose: test whether the MTP3 weakness is caused by expensive/full-logits draft sampling or missing step-aware `get_top_tokens` support.

Patch:
- `vllm/v1/spec_decode/step3p5.py`: Step3.5 local-argmax path now calls the base helper `_model_get_top_tokens(hidden_states, spec_step_idx)` and `_model_compute_logits(...)` rather than bypassing step-aware wrapper detection.
- `vllm/model_executor/models/glm4_moe_mtp.py`: GLM MTP now exposes step-aware `get_top_tokens()` on both the inner predictor and outer wrapper.
- `serve.sh`: added `USE_LOCAL_ARGMAX_REDUCTION=1` to emit `"use_local_argmax_reduction": true` in the speculative config.

Config:
- Image: `glm-darkdevotion-b12x:20260626-arm64-mtpdiag17-step3p5-glm-toptokens`
- Env: `glm52-dcp4-mtp3-16k-localargmax.env`
- Served model: `glm52-mtpdiag17-dcp4-mtp3-16k-localargmax`
- TP4 / DCP4 / MTP3 / max_model_len 16384
- `VLLM_DCP_SHARD_DRAFT=1`
- `VLLM_MTP_RECOMPUTE_TOPK_FROM_STEP=1`
- `VLLM_MTP_BROADCAST_DRAFT_TOKENS=0`
- `USE_LOCAL_ARGMAX_REDUCTION=1`
- `NCCL_IB_DISABLE=0`, `NCCL_SOCKET_IFNAME=enP2p1s0f0np0`
- fp8 KV, `B12X_MLA_SPARSE`, `flashinfer_cutlass`

Startup/capacity:
- Engine KV: 138,752 tokens, 8.47x concurrency at 16,384 tokens/request.
- Worker log confirms local argmax: `Using local argmax reduction for draft token generation`.

Benchmark results, chat completions, temperature 0, no ignore_eos:

| Prompt | Max tokens | Seconds | Output tok/s | Drafts | Draft tokens | Accepted | Pos0 | Pos1 | Pos2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| primes warmup | 64 | 23.768 | 2.693 | 27 | 81 | 37 | 22 | 10 | 5 |
| kanban | 128 | 10.718 | 11.942 | 62 | 186 | 66 | 41 | 19 | 6 |
| kanban | 256 | 21.581 | 11.862 | 130 | 390 | 125 | 83 | 35 | 7 |

Interpretation:
- The patch is valid and the local-argmax fast path runs.
- It improves the 128-token row versus some previous DCP4/MTP3 sharded runs, but it does not make MTP3 healthy.
- The third speculative position remains very weak: 7/130 = 5.4% on the 256-token row.
- Therefore the main MTP3 failure is not full-logits draft sampling or missing GLM `get_top_tokens`; the remaining likely class is iterative DCP metadata/index/top-k/KV state across later MTP positions.

## 2026-06-26 mtpdiag18: DCP1/MTP3/16K local-argmax discriminator

Purpose: test whether MTP3 is intrinsically weak for GLM-5.2's single MTP layer, or whether the collapse is DCP-specific.

Config:
- Image: `glm-darkdevotion-b12x:20260626-arm64-mtpdiag17-step3p5-glm-toptokens`
- Env: `glm52-dcp1-mtp3-16k-localargmax.env`
- Served model: `glm52-mtpdiag18-dcp1-mtp3-16k-localargmax`
- TP4 / DCP1 / MTP3 / max_model_len 16384
- `USE_LOCAL_ARGMAX_REDUCTION=1`
- `VLLM_MTP_RECOMPUTE_TOPK_FROM_STEP=1`
- `NCCL_IB_DISABLE=0`, `NCCL_SOCKET_IFNAME=enP2p1s0f0np0`
- fp8 KV, `B12X_MLA_SPARSE`, `flashinfer_cutlass`

Startup/capacity:
- Engine KV: 34,688 tokens, 2.12x concurrency at 16,384 tokens/request.
- Worker log confirms local argmax reduction.

Benchmark results, chat completions, temperature 0, no ignore_eos:

| Prompt | Max tokens | Seconds | Output tok/s | Drafts | Draft tokens | Accepted | Pos0 | Pos1 | Pos2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| primes warmup | 64 | 13.402 | 4.775 | 24 | 72 | 39 | 18 | 13 | 8 |
| kanban | 128 | 5.637 | 22.708 | 41 | 123 | 86 | 36 | 29 | 21 |
| kanban | 256 | 11.887 | 21.536 | 93 | 279 | 165 | 71 | 55 | 39 |

Interpretation:
- This is the cleanest discriminator so far.
- DCP1/MTP3 is healthy: the 256-token row has pos0/pos1/pos2 acceptance of 76.3%, 59.1%, and 41.9% per proposal cycle, with 21.5 tok/s.
- Therefore MTP3 is not intrinsically bad for GLM-5.2's single MTP layer under this stack.
- The MTP3 collapse is DCP-specific. The likely remaining classes are DCP iterative metadata/top-k/index/KV state or DCP rejection/synchronization semantics.
- Because recent DCP4 logs showed identical draft ids, target probabilities, and accepted lengths across ranks, a live rank-intersection rejection bug is not yet proven. Synthetic rejection remains the right direct test.

## 2026-06-26 mtpdiag19: DCP4/MTP3 synthetic rejection

Purpose: directly test the hypothesis that DCP4 speculative acceptance is accidentally an intersection/min over four rank-local rejection decisions.

Config:
- Env: `glm52-dcp4-mtp3-16k-synthetic.env`
- Served model: `glm52-mtpdiag19-dcp4-mtp3-16k-synthetic`
- Image: `glm-darkdevotion-b12x:20260626-arm64-mtpdiag17-step3p5-glm-toptokens`
- TP4 / DCP4 / MTP3 / 16K
- `kv_cache_dtype=fp8`, `B12X_MLA_SPARSE`, `flashinfer_cutlass`
- `USE_LOCAL_ARGMAX_REDUCTION=1`
- `rejection_sample_method=synthetic`
- `synthetic_acceptance_rates=[0.9,0.75,0.5]`
- KV capacity reported: 138,752 tokens; max concurrency at 16,384 tokens: 8.47x

Results:
- warmup_primes_64: 64 tokens, 15.472s, 4.136 tok/s; drafts 20; accepted per pos 20/15/9; rates 1.000/0.750/0.450
- kanban_128: 128 tokens, 7.735s, 16.547 tok/s; drafts 42; accepted per pos 38/27/20; rates 0.905/0.643/0.476
- kanban_256: 256 tokens, 13.825s, 18.517 tok/s; drafts 84; accepted per pos 71/61/41; rates 0.845/0.726/0.488

Interpretation:
- This falsifies the simple DCP4 rank-intersection/min-accepted-length hypothesis for the active rejection sampler path.
- If DCP4 were effectively AND-ing four independent accept decisions, expected observed rates would be about 0.656/0.316/0.063. The measured rates are close to the configured unconditional synthetic rates instead.
- Therefore the real MTP3 collapse is more likely upstream of rejection sampling: target/draft probability quality, DCP-local verifier/proposer state, top-k/index-cache state across iterative draft steps, or another DCP-specific draft-step-2+ metadata issue.
- Synthetic output text is intentionally incoherent/noisy and should not be used as quality evidence; the run is only a sampler-path diagnostic.

## 2026-06-26 mtpdiag21 draft-probability diagnostic

Config: `glm-darkdevotion-b12x:20260626-arm64-mtpdiag21-draftprob`, `DCP4`, `MTP3`, `max_model_len=16384`, `fp8` KV, B12X sparse MLA, FlashInfer/CUTLASS MoE, local argmax reduction enabled. The first relaunch reproduced MTP3 collapse but did not emit `KZ_MTP_DRAFT_PROB_DIAG` because `serve.sh` was not passing `VLLM_MTP_DRAFT_PROB_DIAG` / `VLLM_MTP_DRAFT_PROB_DIAG_TOPK` through to `docker exec`. `serve.sh` was patched to pass those env vars; no image rebuild was needed.

Startup after the env propagation fix was clean. Main weights loaded `47/47` in about 15 minutes, MTP overlay loaded `2/2` in about 13 seconds, KV cache reported `138,752` tokens and `8.47x` max concurrency at `16,384` context. Engine init took `175.89s`, including `42.65s` compilation. Endpoint: `http://192.168.100.1:18089/v1`, served model `glm52-mtpdiag21-dcp4-mtp3-16k-draftprob`.

Probe results:

```text
warmup_primes_64: tokens=64 seconds=22.785 tps=2.809
kanban_128:       tokens=128 seconds=11.485 tps=11.145
kanban_256:       tokens=256 seconds=22.279 tps=11.491
```

API speculative metrics still show the MTP3 failure signature:

```text
warmup window: Accepted 45 / Drafted 150, per-position acceptance 0.720, 0.180, 0.000
128-ish window: Accepted 60 / Drafted 159, per-position acceptance 0.717, 0.302, 0.113
256-ish window: Accepted 55 / Drafted 192, per-position acceptance 0.641, 0.156, 0.062
```

Rank-0 draft-prob diagnostic summary from 32 logged cycles:

```text
draft_step 0: n=32, draft_argmax_match=1.0000, draft_p_mean=0.6040, draft_p_median=0.6055, min=0.094238, max=1.000000
draft_step 1: n=32, draft_argmax_match=1.0000, draft_p_mean=0.3457, draft_p_median=0.2324, min=0.022095, max=0.890625
draft_step 2: n=32, draft_argmax_match=0.9688, draft_p_mean=0.1787, draft_p_median=0.1113, min=0.008911, max=0.878906
```

Rank-0 verifier/target diagnostic summary from paired rejection logs:

```text
target_step 0: n=32, target_argmax_match=0.7188, target_p_mean=0.6714, target_p_median=0.9198, near_zero<=1e-4: 4
target_step 1: n=30, target_argmax_match=0.4333, target_p_mean=0.4209, target_p_median=0.0615, near_zero<=1e-4: 12
target_step 2: n=30, target_argmax_match=0.1333, target_p_mean=0.1161, target_p_median=0.0000, near_zero<=1e-4: 19
```

Interpretation: this does not look like the draft sampler is emitting random/non-argmax tokens. The generated draft token matched the draft argmax on essentially all sampled steps, including step 2. The collapse is at verification: by draft step 2 the target path usually assigns near-zero probability to the draft token and target argmax match falls to ~13%. This keeps MTP3 in the bad 11-12 tok/s range.

This run does not yet prove cross-DCP-rank divergence because only `rank=0 dcp_rank=0` emitted `KZ_MTP_DRAFT_PROB_DIAG` / `KZ_MTP_REJECT_DIAG` in the head logs. The next high-signal diagnostic is to log the same draft/target summaries from all DCP ranks, or explicitly gather/broadcast them, to determine whether target/draft disagreement is intrinsic to the iterative MTP path under DCP4 or caused by DCP-rank-local state divergence.

## 2026-06-26 mtpdiag21 all-DCP-rank log scrape

Follow-up to the head-local `mtpdiag21` analysis. SSH to the worker hosts timed out, so a read-only Ray task was used on each live node to parse that node's local `/tmp/ray-vllm-worker/session_latest/logs` or `/tmp/ray-vllm-head/session_latest/logs` files.

Result: all four DCP ranks emitted the same diagnostic families locally. The apparent rank-0-only result was just a head-log visibility artifact.

Per-node summaries were identical for draft and target probability shape:

```text
rank 0/1/2/3:
  draft_step 0: n=32, argmax_match=1.0000, p_median=0.6055, p_mean=0.6040
  draft_step 1: n=32, argmax_match=1.0000, p_median=0.2324, p_mean=0.3457
  draft_step 2: n=32, argmax_match=0.9688, p_median=0.1113, p_mean=0.1787

  target_step 0: n=32, argmax_match=0.7188, p_median=0.9198, near_zero<=1e-4: 4
  target_step 1: n=30, argmax_match=0.4333, p_median=0.0615, near_zero<=1e-4: 12
  target_step 2: n=30, argmax_match=0.1333, p_median=0.0000, near_zero<=1e-4: 19

  reject_cycles=32, divergent_reject_sigs=0
```

Interpretation:
- Hidden DCP-rank divergence is not the current explanation. Draft probabilities, target probabilities, and rejection outputs are the same across ranks for the logged cycles.
- The simple rank-intersection/min-accepted-length bug remains falsified by both synthetic rejection and identical live rejection signatures.
- The MTP3 failure is therefore upstream of rejection and rank synchronization: DCP4 changes the iterative draft/target probability relationship itself.
- The likely remaining classes are DCP-mode sparse-index/top-k/KV semantics across iterative MTP steps, or an inherent incompatibility between this branch's DCP4 sparse MLA path and reusing one MTP layer for step 2+.
- The next discriminator is same-image DCP1/MTP3 draft-vs-target probability logging. DCP1/MTP3 was already healthy on throughput/acceptance; matching draft-prob logs will show whether the target-step-2 near-zero pattern is unique to DCP4.

## 2026-06-26 mtpdiag22: DCP1/MTP3 draft-probability comparison

Purpose: compare the healthy DCP1/MTP3 path against the unhealthy DCP4/MTP3 path using the same `mtpdiag21` draft-probability image lineage.

Config:
- Env: `glm52-dcp1-mtp3-16k-draftprob.env`
- Served model: `glm52-mtpdiag22-dcp1-mtp3-16k-draftprob`
- Image: `glm-darkdevotion-b12x:20260626-arm64-mtpdiag21-draftprob`
- TP4 / DCP1 / MTP3 / 16K
- fp8 KV, B12X sparse MLA, FlashInfer/CUTLASS MoE
- local argmax reduction enabled
- prefix caching enabled
- `VLLM_MTP_DRAFT_PROB_DIAG=1`, `VLLM_MTP_DRAFT_PROB_DIAG_TOPK=5`

Startup/capacity:

```text
main weights: 47/47 in ~15:17
MTP overlay: 2/2 in ~14s
GPU KV cache size: 34,688 tokens
Maximum concurrency for 16,384 tokens/request: 2.12x
engine init: 165.10s, compilation 43.01s
```

Probe results:

```text
warmup_primes_64: tokens=64  seconds=17.635 tps=3.629
kanban_128:       tokens=128 seconds=5.875  tps=21.787
kanban_256:       tokens=256 seconds=11.399 tps=22.458
```

API speculative metrics showed healthy MTP3 acceptance:

```text
window A: Mean acceptance length 3.27, per-position acceptance 0.878, 0.735, 0.653
window B: Mean acceptance length 2.97, per-position acceptance 0.853, 0.627, 0.493
window C: Mean acceptance length 2.96, per-position acceptance 0.826, 0.696, 0.435
```

All-node probability summary was identical across ranks:

```text
DCP1/MTP3:
  draft_step 0: n=32, argmax_match=1.0000, p_median=0.9824, p_mean=0.8357
  draft_step 1: n=32, argmax_match=1.0000, p_median=0.9551, p_mean=0.8455
  draft_step 2: n=32, argmax_match=1.0000, p_median=0.9805, p_mean=0.8555

  target_step 0: n=32, argmax_match=0.8125, p_median=0.9998, near_zero<=1e-4: 2
  target_step 1: n=30, argmax_match=0.8333, p_median=0.9996, near_zero<=1e-4: 2
  target_step 2: n=30, argmax_match=0.8333, p_median=1.0000, near_zero<=1e-4: 2

  reject_cycles=32, divergent_reject_sigs=0
```

Direct contrast with DCP4/MTP3 from `mtpdiag21`:

```text
DCP4/MTP3 draft medians:
  step0 0.6055
  step1 0.2324
  step2 0.1113

DCP1/MTP3 draft medians:
  step0 0.9824
  step1 0.9551
  step2 0.9805

DCP4/MTP3 target medians:
  step0 0.9198
  step1 0.0615
  step2 0.0000

DCP1/MTP3 target medians:
  step0 0.9998
  step1 0.9996
  step2 1.0000
```

Interpretation:
- This is the strongest evidence so far.
- MTP3 is healthy when DCP is not active, even with the same image, model, prompts, MTP layer, local-argmax path, fp8 KV, B12X sparse MLA, and FlashInfer/CUTLASS MoE.
- DCP4 does not merely cause rejection-sampler synchronization problems. It changes the iterative MTP draft distribution itself: draft confidence falls hard at steps 1/2, and target verification assigns near-zero probability to the step-2 draft token in most logged cycles.
- All ranks agree in both DCP1 and DCP4, so this is not hidden per-rank divergence. It is a DCP-mode semantic/path issue.
- Current best root-cause class: the DCP4 sparse MLA/index/KV path is not semantically equivalent to DCP1 for iterative multi-step MTP reuse. MTP1 survives because it does not enter the fragile later iterative positions. MTP2 can be viable because step 1 is degraded but not catastrophic. MTP3 fails because step 2 becomes mostly low-confidence/wrong under DCP4.

Practical implication:
- Production remains `DCP4 / 128K / MTP1` unless/until the DCP4 iterative MTP attention/index/KV path is fixed.
- `DCP4 / MTP2` remains worth A/B testing because it can still be near or slightly above MTP1.
- `DCP4 / MTP3` should be considered architecturally broken in this current branch/config, not merely undertuned.

Next technical discriminator if continuing:
- Run DCP4/MTP3 while toggling sparse index/cache/attention pieces, not sampler pieces. The target is to identify which DCP4 path breaks iterative MTP semantics: B12X sparse MLA, DCP global top-k/index cache, draft/index KV sharding, or another DCP-specific cache update path.

## 2026-06-26 mtpdiag23: FlashInfer sparse-MLA backend ablation failed at init

Purpose: test whether the DCP4/MTP3 collapse is specific to B12X sparse MLA by switching both target and draft attention backend to `FLASHINFER_MLA_SPARSE_DSV4` while keeping the rest of the DCP4/MTP3 diagnostic config unchanged.

Script change: `serve.sh` now supports `DRAFT_ATTENTION_BACKEND`, defaulting to `ATTENTION_BACKEND`, so the MTP draft backend is no longer hardcoded to `B12X_MLA_SPARSE`. Existing B12X envs preserve behavior because their `ATTENTION_BACKEND` remains `B12X_MLA_SPARSE`.

Config:

```text
Env: glm52-dcp4-mtp3-16k-flashinfermla-draftprob.env
Served model: glm52-mtpdiag23-dcp4-mtp3-16k-flashinfermla
Target attention: FLASHINFER_MLA_SPARSE_DSV4
Draft attention:  FLASHINFER_MLA_SPARSE_DSV4
TP4 / DCP4 / MTP3 / 16K / fp8 KV / flashinfer_cutlass MoE
```

Result: startup failed before weight load. vLLM rejected the FlashInfer sparse-MLA backend for this Spark/GLM configuration:

```text
ValueError: Selected backend AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV4 is not valid for this configuration.
Reason: ['head_size not supported', 'compute capability not supported']
```

Interpretation:
- This ablation cannot currently isolate B12X-vs-FlashInfer behavior on Spark/SM121 with this image.
- The inability to run the non-B12X sparse MLA backend is itself part of the architectural constraint: the working Spark path is effectively B12X sparse MLA for GLM-5.2 in this branch.
- Continue with B12X-internal ablations: index cache, DCP global top-k, and draft/index KV grouping rather than alternate attention backend.
## 2026-06-26 mtpdiag25: synthetic rejection diagnostic wedged before workers

Config:

- Env: `glm52-dcp4-mtp3-16k-synthetic-reject.env`
- Image: `glm-darkdevotion-b12x:20260626-arm64-mtpdiag21-draftprob`
- Served model: `glm52-mtpdiag25-dcp4-mtp3-16k-synthetic`
- Topology: TP4 / PP1 / DCP4, 16K, MTP3
- Synthetic rejection config:
  - `rejection_sample_method=synthetic`
  - `synthetic_acceptance_rates=[0.9, 0.75, 0.5]`

Expected discriminator:

- Correct one-decision-per-request behavior should observe roughly `0.90 / 0.75 / 0.50`.
- Buggy DCP4 rank-intersection/min behavior should observe roughly `0.656 / 0.316 / 0.063`.

Result:

- The rendered vLLM command and parsed `speculative_config` were correct.
- vLLM accepted the synthetic rejection fields.
- Two clean launch attempts both wedged at the same point:
  - API server parsed arguments.
  - EngineCore connected to Ray.
  - Ray placement group was created/reserved.
  - No `RayWorkerWrapper` startup/model-load logs followed.
- Endpoint never opened.
- The stuck API/EngineCore was killed after each attempt.

Interpretation:

- This did not produce usable acceptance data.
- The synthetic rejection path is not currently an actionable diagnostic in this image/config without separately debugging the pre-worker hang.
- The rank-intersection hypothesis should be tested by live sampler instrumentation instead: log per-DCP-rank local accepted length and final accepted length, or broadcast/force a canonical accepted length in the active generic proposer path.

## 2026-06-26 mtpdiag26: DCP rejection-output broadcast diagnostic also wedged

Config:

- Env: `glm52-dcp4-mtp3-16k-sync-reject.env`
- Image: `glm-darkdevotion-b12x:20260626-arm64-mtpdiag21-draftprob`
- Served model: `glm52-mtpdiag26-dcp4-mtp3-16k-sync-reject`
- Topology: TP4 / PP1 / DCP4, 16K, MTP3
- Standard rejection sampling, with `VLLM_DCP_SYNC_REJECTION_OUTPUT=1`

Intent:

- Exercise the existing sample-side rejection sampler knob that broadcasts `output_token_ids` from DCP rank 0 after rejection sampling.
- If DCP4 MTP3 acceptance collapse is caused by rank-local rejection decisions being combined as a min/intersection, this should force a canonical rejection output and materially improve MTP3.

Result:

- The rendered command was correct.
- vLLM parsed the standard MTP3 config.
- Launch wedged at the same boundary as `mtpdiag25`:
  - API server parsed arguments.
  - EngineCore connected to Ray.
  - Ray placement group was created/reserved.
  - No worker actor/model-load logs followed.
- Endpoint never opened.
- The stuck API/EngineCore was killed.

Interpretation:

- This run does not prove or disprove the rejection-intersection hypothesis.
- Because both `mtpdiag25` and `mtpdiag26` hang before workers start, the immediate launch environment or this image lineage is now a blocker for more DCP4/MTP3 launch-based diagnostics.
- Next useful step is either a clean Ray/container reset followed by a known-good baseline launch, or a code-level instrumentation patch in the active sample-side rejection sampler and then rebuild/retest from the known-good baseline lineage.

## 2026-06-26 late: mtpdiag21 baseline was slow, not wedged

After force-removing a stale `glm-dark-head` process/container and all three `glm-dark-worker` containers, the known-good DCP4/MTP3/16K draft-prob baseline was relaunched:

- Env: `glm52-dcp4-mtp3-16k-draftprob.env`
- Served model: `glm52-mtpdiag21-dcp4-mtp3-16k-draftprob`
- Topology: TP4 / PP1 / DCP4, 16K, MTP3
- KV: fp8, explicit `kv_cache_memory_bytes=1900000000`

Launch observations:

- This was not a pre-worker hang. The API log stayed quiet, but Ray worker logs showed actor/model-load progress.
- Rank 0 on the head was much slower than the other ranks:
  - ranks 1-3 target load: roughly 349-415s
  - rank 0 target load: 912.13s
  - rank 0 full target+drafter load: 952.75s
- Rank 0 was intermittently in `D` state on `folio_wait_bit_common`, but `read_bytes` kept advancing, so it was slow page/cache I/O rather than a dead actor.
- Engine cache/profiling:
  - `GPU KV cache size: 138,752 tokens`
  - `Maximum concurrency for 16,384 tokens per request: 8.47x`
  - profile/create KV/warmup took `177.41s`

Probe results:

```text
warmup_primes_64: tokens=64  seconds=21.520  tps=2.974
kanban_128:       tokens=128 seconds=9.798   tps=13.064
kanban_256:       tokens=256 seconds=20.288  tps=12.618
```

Interpretation:

- The previous `mtpdiag25` and `mtpdiag26` launches were interrupted too early to distinguish a true hang from slow head-node weight I/O.
- DCP4/MTP3 baseline remains weak but functional.
- The next rejection-sync test must be allowed to run through the full slow head-rank load path before judging it.

## 2026-06-27 corrected DCP4/MTP3 sync-rejection result

The first `mtpdiag26` sync-rejection attempt was not a valid test: `VLLM_DCP_SYNC_REJECTION_OUTPUT=1` was present in the env file but was not passed through the container/API/Ray-worker launch whitelist. I added the variable to `launch-ray.sh` and `serve.sh` so it is propagated into the head container, worker containers, and API-server `docker exec` environment.

Corrected run: `glm52-mtpdiag26-dcp4-mtp3-16k-sync-reject`, DCP4/MTP3/16K, B12X MLA sparse attention, FlashInfer CUTLASS MoE, fp8 KV, `kv_cache_memory_bytes=1900000000`, `VLLM_DCP_SYNC_REJECTION_OUTPUT=1` confirmed in the API process and in all four Ray workers.

Client probe:

```text
warmup_primes_64: tokens=64 seconds=20.471 tps=3.126
kanban_128:       tokens=128 seconds=11.345 tps=11.283
kanban_256:       tokens=256 seconds=19.501 tps=13.128
```

SpecDecode metrics remained in the weak DCP4/MTP3 band:

```text
acceptance: 0.647, 0.353, 0.039
acceptance: 0.557, 0.230, 0.049
acceptance: 0.723, 0.354, 0.062
acceptance: 0.688, 0.375, 0.125
```

Interpretation: broadcasting/synchronizing the rejection output is not sufficient to rescue MTP3. The per-cycle diagnostics still show many failures where the target verifier itself assigns near-zero probability to later draft tokens, for example second/third-step `target_p` values like `0.0`, `0.000536`, `0.001073`, `0.0`, and `0.036726` while the draft produced a concrete token. This weakens the pure "final accepted length is min/AND across DCP ranks" hypothesis. The remaining failure looks earlier: DCP4 iterative-MTP target/draft state, sparse MLA/indexer/index-cache behavior, or DCP-local metadata used by the verifier after draft step 1.

Operational note: this 16K diagnostic is still memory-tight enough to be noisy. During load vLLM reported only about 4.20 GiB available RAM and the head worker took roughly 20 minutes to finish initialization. Future MTP-logic diagnostics should lower explicit KV allocation or otherwise loosen memory pressure before comparing acceptance.

## 2026-06-27 DCP4/MTP3 no-index-cache loose result

Run: `glm52-mtpdiag27-dcp4-mtp3-16k-noindexcache-loose`.

Purpose: test whether the DCP4/MTP3 later-token collapse is caused by the sparse index cache/reuse path. This run used `HF_OVERRIDES={"use_index_cache":false}` and reduced explicit KV pressure to `KV_CACHE_MEMORY_BYTES=900000000`.

Launch fixes made on the way:

- `serve.sh` had ambiguous Bash parameter expansion for the default `HF_OVERRIDES` JSON; explicit no-index overrides were receiving an extra `}` and failing JSON parse.
- Replaced that with an explicit `if [[ -z "${HF_OVERRIDES:-}" ]]` default assignment.
- Confirmed vLLM parsed `hf_overrides` as `{'use_index_cache': False}`.

Capacity/load observations:

```text
GPU KV cache size: 57,600 tokens
Maximum concurrency for 16,384 tokens per request: 3.52x
Model loading took 107.92 GiB memory and 965.23 seconds
```

The no-index path logged `Setting kv cache block size to 64 for B12X_NON_COMPRESSED_INDEXER backend`, confirming the index-cache behavior changed.

Client probe:

```text
warmup_primes_64: tokens=64 seconds=22.035 tps=2.904
kanban_128:       tokens=128 seconds=11.101 tps=11.531
kanban_256:       tokens=256 seconds=23.434 tps=10.924
```

SpecDecode metrics remained weak:

```text
acceptance: 0.710, 0.323, 0.065
acceptance: 0.786, 0.321, 0.071
acceptance: 0.585, 0.208, 0.057
acceptance: 0.636, 0.291, 0.091
acceptance: 0.708, 0.333, 0.083
```

Interpretation: disabling index cache does not rescue MTP3 and reduces throughput. The later-token target mismatch persists, with many per-cycle target probabilities near zero for the second or third drafted token. The index cache/reuse path is therefore not the simple dominant failure. Keep index cache enabled for production and most future diagnostics unless specifically testing another indexer hypothesis.

## 2026-06-27 DCP4/MTP3 no-local-argmax loose result

Run: `glm52-mtpdiag28-dcp4-mtp3-16k-nolocalargmax-loose`.

Purpose: test whether the `use_local_argmax_reduction` draft-token path is responsible for the DCP4/MTP3 later-token collapse. This run restored index cache behavior and used `KV_CACHE_MEMORY_BYTES=900000000`, but omitted `use_local_argmax_reduction` from `speculative_config`.

Confirmed launch shape:

```text
hf_overrides: {'use_index_cache': True, 'index_topk_pattern': ...}
speculative_config: {'method': 'mtp', 'num_speculative_tokens': 3, ..., 'draft_sample_method': 'probabilistic'}
GPU KV cache size: 65,536 tokens
Maximum concurrency for 16,384 tokens per request: 4.00x
```

Client probe:

```text
warmup_primes_64: tokens=64 seconds=18.495 tps=3.460
kanban_128:       tokens=128 seconds=10.276 tps=12.456
kanban_256:       tokens=256 seconds=20.854 tps=12.276
```

SpecDecode metrics:

```text
acceptance: 0.794, 0.382, 0.176
acceptance: 0.712, 0.237, 0.119
acceptance: 0.625, 0.312, 0.094
acceptance: 0.638, 0.259, 0.086
```

Interpretation: disabling local argmax does not rescue MTP3. It may slightly move individual windows, but the overall throughput and acceptance curve remain in the weak DCP4/MTP3 band. This weakens the local-argmax hypothesis as the primary cause. The persistent signature is still target-verifier disagreement on later drafted tokens under DCP4.

## Test policy: loose KV while isolating MTP correctness

For MTP correctness diagnostics, use a deliberately loose KV/memory profile rather than the tight 128K production budget. The goal is to keep 16K DCP/MTP experiments away from borderline Linux reclaim, swap, graph/workspace pressure, and vLLM free-memory startup checks so acceptance behavior is not polluted by host-memory pressure.

Production validation remains separate: once a candidate improves MTP2/MTP3 behavior under loose 16K diagnostics, rerun it at the 128K/DCP4 production KV target and compare against the frozen DCP4/MTP1 baseline. Loose-KV TPS is useful for direction, but not sufficient to promote a config.

## 2026-06-27: DCP4/MTP3 synthetic rejection loose-memory result

Run: `glm52-mtpdiag29-dcp4-mtp3-16k-synthetic-loose`

Configuration highlights:

- DCP4 / TP4 / PP1
- MTP3
- 16K max model length
- prefix caching enabled
- fp8 KV
- B12X MLA sparse attention
- flashinfer_cutlass MoE
- explicit loose KV budget: `KV_CACHE_MEMORY_BYTES=900000000`
- `GPU_MEMORY_UTILIZATION=0.90`
- synthetic rejection rates: `[0.9, 0.75, 0.5]`

Capacity and startup:

- GPU KV cache size: `65,536` tokens
- Maximum concurrency at 16,384 tokens/request: `4.00x`
- Engine profile/KV/warmup time: `168.28s` with `43.34s` compilation

Client probe:

```text
warmup_primes_64: tokens=64 seconds=17.455 tps=3.667
kanban_128:       tokens=128 seconds=7.569  tps=16.910
kanban_256:       tokens=256 seconds=13.753 tps=18.614
```

SpecDecode windows:

```text
0.945, 0.745, 0.509
0.869, 0.705, 0.492
0.793, 0.655, 0.414
```

Interpretation:

Synthetic rejection tracks the configured rates closely enough to rule out the simple DCP4 `p^4` / all-ranks-must-accept / final-min-accepted-length hypothesis as the dominant current failure. The rejection layer can mechanically accept MTP3 under DCP4 when forced.

This also shows MTP3 overhead is not inherently fatal under DCP4: forced acceptance reaches ~18.6 tok/s on the 256-token probe, materially above the normal DCP4/MTP3 loose/baseline rows.

The remaining failure is therefore upstream of final rejection coordination: in real mode, the target verifier probability stream under iterative DCP4 MTP diverges from the draft stream by draft positions 2/3. The most likely classes are DCP-local target attention metadata, sequence lengths, slot mapping, sparse MLA/indexer state, or top-k/index sharing across iterative draft steps.

## 2026-06-27: DCP4/MTP3 no-async loose-memory result

Run: `glm52-mtpdiag30-dcp4-mtp3-16k-noasync-loose`

Configuration delta from normal loose DCP4/MTP3:

- `DISABLE_ASYNC_SCHEDULING=1`
- vLLM command included `--no-async-scheduling`
- vLLM parsed `async_scheduling=False`
- Same explicit loose KV budget: `KV_CACHE_MEMORY_BYTES=900000000`

Capacity and startup:

- GPU KV cache size: `65,536` tokens
- Maximum concurrency at 16,384 tokens/request: `4.00x`
- Engine profile/KV/warmup time: `152.04s` with `42.08s` compilation

Client probe:

```text
warmup_primes_64: tokens=64 seconds=18.582 tps=3.444
kanban_128:       tokens=128 seconds=10.848 tps=11.799
kanban_256:       tokens=256 seconds=21.805 tps=11.741
```

SpecDecode windows:

```text
0.767, 0.349, 0.047
0.600, 0.250, 0.067
0.661, 0.258, 0.048
0.613, 0.194, 0.065
```

Interpretation:

Disabling async scheduling did not rescue DCP4/MTP3. Throughput and acceptance remain in the bad normal DCP4/MTP3 band rather than moving toward the synthetic-rejection row or DCP1/MTP3 behavior.

This weakens the hypothesis that the main bug is simply async scheduler CPU upper-bound metadata. The remaining target is more likely DCP4 iterative draft/verify state itself: sparse MLA/indexer top-k state, per-rank position/slot semantics, or DCP-local target/draft context consistency after draft step 1.

## 2026-06-27: loose-memory MTP diagnostics and FlashInfer attention backend check

For MTP correctness diagnostics, prefer a loose memory profile first: lower `gpu_memory_utilization` and/or explicit smaller `kv_cache_memory_bytes` so Linux/Ray/vLLM memory pressure does not contaminate acceptance or throughput results. Once a config shows valid MTP2/MTP3 behavior, retest with the tight 128K production KV budget.

Attempted `DCP4 / MTP3 / 16K / fp8 KV / loose KV` using `FLASHINFER_MLA_SPARSE_DSV4` for target and draft attention. This did not reach serving. vLLM rejected the backend during model construction with: `Selected backend AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV4 is not valid for this configuration. Reason: ['head_size not supported', 'compute capability not supported']`. Treat this as a backend/build-guard incompatibility on Spark SM121, not as an MTP performance datapoint. Do not retry this exact backend path unless the FlashInfer/vLLM guards are patched or rebuilt for this configuration.

## 2026-06-27: DCP4/MTP3 loose-memory real rejection baseline

Run: `glm52-mtpdiag32-dcp4-mtp3-16k-loose`, using `DCP4 / TP4 / MTP3 / max_model_len=16384 / max_num_seqs=1 / fp8 KV / B12X_MLA_SPARSE / flashinfer_cutlass MoE / prefix caching on / async scheduling on`, with loose memory settings `gpu_memory_utilization=0.90` and `kv_cache_memory_bytes=900000000`.

Startup capacity: `GPU KV cache size: 65,536 tokens`; `Maximum concurrency for 16,384 tokens per request: 4.00x`.

Probe results:

```text
warmup_primes_64: tokens=64 seconds=24.744 tps=2.587
kanban_128:       tokens=128 seconds=10.925 tps=11.716
kanban_256:       tokens=256 seconds=22.454 tps=11.401
```

SpecDecode windows stayed in the bad MTP3 pattern:

```text
0.577, 0.385, 0.077
0.690, 0.276, 0.103
0.721, 0.279, 0.049
0.607, 0.131, 0.066
0.435, 0.174, 0.087
```

Interpretation: loosening KV/memory pressure does not recover real DCP4/MTP3. Combined with the synthetic rejection result under a similar loose profile, this makes Linux swap/memory pressure unlikely to be the primary cause of MTP3 acceptance collapse. The remaining target is real draft/target probability alignment in iterative DCP MTP, likely around DCP-local metadata, slot/page mapping, or sparse MLA/indexer state after draft step 1.

## 2026-06-27: draft_p=None interpretation

`KZ_MTP_PROB_DIAG` showed `draft_p=None` in the real-rejection loose MTP3 run. This is expected for the current probe requests because they use `temperature=0.0`: `LLMBaseProposer._sample_draft_tokens()` returns greedy draft tokens and no draft probability tensor when `sampling_metadata.all_greedy` is true. Therefore the sampler accepts greedy MTP tokens according to raw target probability / target agreement rather than a `p/q` stochastic rejection ratio.

This is not by itself a handoff bug. It changes the next diagnostic: rerun the same loose DCP4/MTP3 setup with `temperature>0` so `draft_sample_method=probabilistic` produces real `draft_probs`. If acceptance recovers only for non-greedy requests, focus on greedy/local-argmax/DCP agreement. If acceptance remains bad with `draft_p` populated, focus on target-logit / metadata alignment for iterative DCP MTP steps.

## 2026-06-27: DCP4/MTP3 loose-memory non-greedy probe

Reran `glm52-mtpdiag32-dcp4-mtp3-16k-loose` and sent identical short codegen prompts with nonzero temperature so `LLMBaseProposer._sample_draft_tokens()` returns real `draft_probs` and rejection uses a target/draft probability ratio.

Probe results:

```text
temp=0.2 warmup_primes_64: tokens=64  seconds=23.174 tps=2.762
temp=0.2 kanban_128:       tokens=128 seconds=10.950 tps=11.689
temp=0.2 kanban_256:       tokens=256 seconds=20.896 tps=12.251
temp=0.7 warmup_primes_64: tokens=64  seconds=5.612  tps=11.404
temp=0.7 kanban_128:       tokens=128 seconds=10.979 tps=11.658
temp=0.7 kanban_256:       tokens=256 seconds=21.189 tps=12.082
```

SpecDecode windows stayed weak despite populated `draft_p`:

```text
0.776, 0.276, 0.121
0.724, 0.207, 0.069
0.651, 0.254, 0.079
0.632, 0.246, 0.088
0.600, 0.267, 0.100
0.483, 0.117, 0.050
0.609, 0.359, 0.172
```

`KZ_MTP_PROB_DIAG` confirms real draft probabilities are present, but target and draft often disagree catastrophically. Examples include high-confidence draft rows where target probability for the draft token is zero or near-zero, even for step-2/step-3 rows. This rules out the simple explanations: missing draft probability handoff, greedy-only local argmax behavior, and final rejection coordination. The remaining likely class is draft/target state divergence or row/position misalignment in iterative DCP4 MTP metadata, especially sparse MLA/indexer/slot mapping across draft steps 1+.

## 2026-06-27: DCP4/MTP3 B12X exact sequence length diagnostic

This run tested whether B12X MLA sparse attention's use of `seq_lens_cpu_upper_bound` in the multi-token path was the dominant DCP4/MTP3 failure. A runtime patch added `VLLM_B12X_MLA_EXACT_SEQ_LENS=1`, forcing the backend to use exact `seq_lens_cpu` instead of the upper bound. The launch used DCP4/MTP3/16K, fp8 KV, B12X MLA sparse attention, `flashinfer_cutlass` MoE, prefix caching enabled, async scheduling enabled, `KV_CACHE_MEMORY_BYTES=900000000`, and `GPU_MEMORY_UTILIZATION=0.90`.

Startup reported `GPU KV cache size: 65,536 tokens` and `Maximum concurrency for 16,384 tokens per request: 4.00x`. Probe results were: temp=0.0 warmup 64 tokens at 2.647 tok/s, 128-token kanban at 11.999 tok/s, 256-token kanban at 12.981 tok/s; temp=0.2 warmup 64 at 11.309 tok/s, 128 at 12.260 tok/s, 256 at 11.950 tok/s; temp=0.7 warmup 64 at 11.130 tok/s, 128 at 11.650 tok/s, 256 at 11.464 tok/s.

Speculative acceptance windows remained weak for MTP3: `0.500,0.083,0.083`, `0.789,0.421,0.088`, `0.705,0.262,0.082`, `0.629,0.339,0.113`, `0.691,0.273,0.091`, `0.705,0.230,0.066`, `0.641,0.188,0.062`, `0.627,0.305,0.153`, `0.691,0.236,0.036`, `0.484,0.161,0.097`, `0.565,0.177,0.016`, `0.726,0.355,0.065`. This slightly improved some windows and one hot temp=0 run, but it did not recover MTP3 toward MTP1 or healthy DCP1 behavior.

Interpretation: B12X's upper-bound sequence length choice is not the dominant MTP3 bug. The remaining likely class is still target/draft state divergence or flattened row/position misalignment in the iterative MTP path under DCP4, especially after draft step 1. Also note that this run's probability diagnostic limit saturated early, so later non-greedy requests did not produce enough additional `KZ_MTP_PROB_DIAG` rows.

Operational note: for future MTP correctness tests, prefer a loose-memory diagnostic profile first: one 16K request, explicit small KV allocation, and lower `gpu_memory_utilization`. Only restore 128K/prod KV pressure after a valid acceptance/perf improvement is demonstrated.

## 2026-06-27: DCP4/MTP3 16K low-memory 500MB KV startup stall

Attempted a looser diagnostic launch named `glm52-mtpdiag34-dcp4-mtp3-16k-lowmem` using DCP4/MTP3/16K, fp8 KV, B12X MLA sparse, `flashinfer_cutlass` MoE, prefix caching enabled, async enabled, `GPU_MEMORY_UTILIZATION=0.86`, and `KV_CACHE_MEMORY_BYTES=500000000`. The goal was to reduce memory-pressure confounding while preserving enough KV for a single 16K request.

The launch started Ray across all four nodes and patched diagnostics. All four ranks allocated approximately 108 GiB of GPU memory and stayed active in `ray::RayWorkerProc.initialize_worker`, but the API never started listening and no `GPU KV cache size`, `Maximum concurrency`, or `Application startup complete` lines appeared after a bounded wait. Head memory during the stall was roughly 7-8 GiB available, with swap free around 11.9 GiB. No explicit Python exception was observed.

Interpretation: 500MB explicit KV may be too aggressive for this MTP3/DCP4 startup path, or it exposed a startup/profile stall before useful MTP acceptance data. Do not use this row as an MTP performance datapoint. Next diagnostic should use a moderate KV budget, e.g. 700-800MB, to stay below the earlier 900MB profile while avoiding this startup stall.

## 2026-06-27: DCP4/MTP3 16K mid-memory 750MB KV result

Launch `glm52-mtpdiag35-dcp4-mtp3-16k-midmem` used DCP4/MTP3/16K, fp8 KV, B12X MLA sparse attention, `flashinfer_cutlass` MoE, prefix caching enabled, async enabled, `GPU_MEMORY_UTILIZATION=0.88`, `KV_CACHE_MEMORY_BYTES=750000000`, `VLLM_MTP_DCP_DIAG_LIMIT=128`, and runtime diagnostics patched in.

Startup succeeded but was slow: base GLM weights loaded in 916.20s, MTP overlay loaded in 14.21s, graph capture took 25s and 0.54 GiB. vLLM reported `GPU KV cache size: 54,784 tokens` and `Maximum concurrency for 16,384 tokens per request: 3.34x`. This is a valid serving profile and sits below the earlier 900MB loose profile, but it still ended with tight head memory: roughly 1.2 GiB available, 1.16 GiB swap cached, and about 5.3 GiB swap used.

Probe results:

```text
temp=0.0 warmup_primes_64: tokens=64 seconds=23.763 tps=2.693
temp=0.0 kanban_128:       tokens=128 seconds=11.472 tps=11.158
temp=0.0 kanban_256:       tokens=256 seconds=21.183 tps=12.085
temp=0.2 warmup_primes_64: tokens=64 seconds=4.524  tps=14.147
temp=0.2 kanban_128:       tokens=128 seconds=11.291 tps=11.336
temp=0.2 kanban_256:       tokens=256 seconds=19.422 tps=13.181
temp=0.7 warmup_primes_64: tokens=64 seconds=4.927  tps=12.990
temp=0.7 kanban_128:       tokens=128 seconds=10.582 tps=12.097
temp=0.7 kanban_256:       tokens=256 seconds=20.013 tps=12.791
```

SpecDecoding windows remained in the same weak MTP3 band:

```text
0.684, 0.316, 0.158
0.727, 0.364, 0.164
0.614, 0.281, 0.070
0.562, 0.234, 0.078
0.769, 0.462, 0.212
0.627, 0.254, 0.051
0.721, 0.295, 0.164
0.707, 0.345, 0.190
0.721, 0.279, 0.115
0.633, 0.217, 0.033
0.766, 0.344, 0.125
0.536, 0.393, 0.071
```

Interpretation: lowering the explicit KV pool from 900MB to 750MB did not rescue DCP4/MTP3. Position 3 still collapses frequently, and hot throughput remains roughly 11-13 tok/s rather than approaching the MTP1/MTP2 band. The diagnostic limit was high enough to capture more cycles, but the probability rows still saturated during the early greedy phase (`draft_p=None` expected for greedy), so this row is primarily an acceptance/perf datapoint rather than a non-greedy probability datapoint.

## 2026-06-27: DCP4/MTP2 16K mid-memory 750MB KV A/B result

Launch `glm52-mtpdiag36-dcp4-mtp2-16k-midmem` used the same profile as the MTP3 mid-memory run except `NUM_SPECULATIVE_TOKENS=2`: DCP4/16K, fp8 KV, B12X MLA sparse attention, `flashinfer_cutlass` MoE, prefix caching enabled, async enabled, `GPU_MEMORY_UTILIZATION=0.88`, `KV_CACHE_MEMORY_BYTES=750000000`, and runtime diagnostics patched in.

Startup matched the MTP3 capacity profile: `GPU KV cache size: 54,784 tokens` and `Maximum concurrency for 16,384 tokens per request: 3.34x`. Base weight load took 925.47s, MTP overlay 14.06s, graph capture 25s and 0.46 GiB. End-state memory was still tight, roughly 0.9 GiB available with ~1.46 GiB swap cached and ~5.3 GiB swap used.

Probe results:

```text
temp=0.0 warmup_primes_64: tokens=64 seconds=17.115 tps=3.739
temp=0.0 kanban_128:       tokens=128 seconds=9.401  tps=13.615
temp=0.0 kanban_256:       tokens=256 seconds=18.783 tps=13.629
temp=0.2 warmup_primes_64: tokens=64 seconds=4.761  tps=13.443
temp=0.2 kanban_128:       tokens=128 seconds=8.860  tps=14.446
temp=0.2 kanban_256:       tokens=256 seconds=17.978 tps=14.240
temp=0.7 warmup_primes_64: tokens=64 seconds=4.879  tps=13.118
temp=0.7 kanban_128:       tokens=128 seconds=9.823  tps=13.030
temp=0.7 kanban_256:       tokens=256 seconds=16.872 tps=15.173
```

SpecDecoding windows:

```text
0.745, 0.373
0.731, 0.299
0.722, 0.222
0.708, 0.323
0.732, 0.282
0.689, 0.338
0.672, 0.313
0.690, 0.225
0.622, 0.284
0.719, 0.421
```

Interpretation: under identical mid-memory conditions, MTP2 remains materially healthier than MTP3. Hot codegen lands mostly in the 13-15 tok/s band, while MTP3 stayed mostly 11-13 tok/s and repeatedly collapsed at draft position 3. This rules against memory pressure as the primary explanation for MTP3's poor behavior. The next code-level target should be the third iterative draft step: target/draft row alignment, DCP-local metadata after step 2, slot mapping/page mapping, or reused sparse/indexer state specifically entering draft position 3.

## 2026-06-26 rowdiag parse: DCP4 / MTP3 / 16K under pressure

Parsed the rowdiag run after adding verifier-row diagnostics to `rejection_sampler.py`.

Observed:

- Logged probability cycles: 96.
- Logged DCP slot-mapping cycles: 96.
- Source-row invariant was clean after warmup: `src_delta=[1,1,1]`; only cycles 0 and 1 had startup one-row forms.
- Mean accept probabilities by draft position: `[0.6830, 0.3450, 0.2051]`.
- Mean target/draft argmax match by draft position: `[0.6354, 0.3191, 0.1702]`.
- Head-rank local-slot correlation did not explain the collapse:
  - step 0 local=0 mean 0.7349, local=1 mean 0.5356
  - step 1 local=0 mean 0.3408, local=1 mean 0.3589
  - step 2 local=0 mean 0.2195, local=1 mean 0.1606

Interpretation:

- This does not look like a verifier row slicing bug. The target rows being compared against draft tokens are aligned as expected.
- The remaining MTP3 issue is more likely in DCP-distributed draft/KV/indexer semantics, target/draft divergence after iterative draft updates, or the basic economics of the third draft pass under DCP4.
- This run was under severe host memory pressure (`MemAvailable` around 1.5 GB and swap in use), so its absolute TPS should not be used as a serving-performance datapoint. It is valid mainly as instrumentation evidence.

Next diagnostic rule:

- Use deliberately smaller explicit KV pools / lower memory pressure for MTP2 and MTP3 debugging. If a lower-pressure setup improves MTP2/MTP3 acceptance or throughput, the previous tight 128K profile was contaminating the result. If it does not, keep the 128K/MTP1 production point frozen and continue debugging the iterative MTP path.

## 2026-06-27 clean-start mid-memory MTP2 check

Profile:

- Env file: `glm52-dcp4-mtp2-16k-midmem.env`
- `DCP4`, `TP4`, `MTP2`, `MAX_MODEL_LEN=16384`
- `KV_CACHE_MEMORY_BYTES=750000000`
- `GPU_MEMORY_UTILIZATION=0.88`
- `KV_CACHE_DTYPE=fp8`
- B12X sparse MLA attention, FlashInfer/CUTLASS MoE
- Prefix caching enabled, CUDA graphs enabled, async scheduling enabled
- IB enabled: `NCCL_IB_DISABLE=0`, `NCCL_SOCKET_IFNAME=enP2p1s0f0np0`

Pre-launch cleanup:

- Removed GLM containers on head/workers.
- Drained swap with `swapoff -a && swapon -a`.
- Dropped page cache.
- Head returned to about 122 GB available before launch.

Startup:

- API ready at 2026-06-27 06:08:50.
- KV cache size: 54,784 tokens.
- Maximum concurrency for 16,384 tokens/request: 3.34x.
- Engine init/profile/cache/warmup: 153.54s, compilation 42.84s.
- Host still became memory-tight after model load/cache setup, so this is cleaner than the 128K profile but not entirely free of unified-memory pressure.

Client-side completions benchmark:

- Warmup, 128 tokens: 27.802s, 4.604 tok/s.
- Hot codegen 384 tokens #1: 26.030s, 14.752 tok/s.
- Hot codegen 384 tokens #2: 26.007s, 14.765 tok/s.
- Hot codegen 384 tokens #3: 24.254s, 15.832 tok/s.

Interpretation:

- MTP2 is still viable under the looser KV profile.
- This does not yet prove MTP2 beats the frozen MTP1 production profile, but it confirms MTP2 is not intrinsically broken like MTP3 has looked.
- Next comparison should use the same clean-start mid-memory procedure for MTP3 before deciding whether the MTP3 issue is memory-pressure contamination or an iterative-DCP problem.

## 2026-06-27 clean-start mid-memory MTP3 check

Profile:

- Env file: `glm52-dcp4-mtp3-16k-midmem.env`
- Same as the MTP2 mid-memory check except `NUM_SPECULATIVE_TOKENS=3`.
- `DCP4`, `TP4`, `MAX_MODEL_LEN=16384`
- `KV_CACHE_MEMORY_BYTES=750000000`
- `GPU_MEMORY_UTILIZATION=0.88`
- `KV_CACHE_DTYPE=fp8`
- B12X sparse MLA attention, FlashInfer/CUTLASS MoE
- Prefix caching enabled, CUDA graphs enabled, async scheduling enabled
- IB enabled: `NCCL_IB_DISABLE=0`, `NCCL_SOCKET_IFNAME=enP2p1s0f0np0`

Pre-launch cleanup:

- Removed GLM containers on head/workers.
- Drained swap and dropped caches.
- Head returned to about 122 GB available before launch.

Startup:

- API ready at 2026-06-27 06:32:41.
- KV cache size: 54,784 tokens.
- Maximum concurrency for 16,384 tokens/request: 3.34x.
- Engine init/profile/cache/warmup: 166.07s, compilation 44.94s.
- As with MTP2, host memory became tight after load/cache setup, so this is a cleaner but not perfectly pristine benchmark.

Client-side completions benchmark:

- Warmup, 128 tokens: 22.935s, 5.581 tok/s.
- Hot codegen 384 tokens #1: 27.420s, 14.005 tok/s.
- Hot codegen 384 tokens #2: 23.858s, 16.095 tok/s.
- Hot codegen 384 tokens #3: 25.660s, 14.965 tok/s.

Interpretation:

- MTP3 did not show the earlier catastrophic third-token collapse in this matched 16K / 750MB-KV / clean-start test.
- Server-side hot windows reached 14.9-18.7 tok/s, and third-position acceptance reached useful values in good windows instead of staying near zero.
- Client-side MTP3 is broadly comparable to MTP2 here, not clearly superior. It may still be uneconomic once the extra draft pass is counted, but the previous MTP3 failure is at least partly configuration/state sensitive.
- This strengthens the case for testing MTP2/MTP3 at a slightly lower-memory 128K-ish profile before declaring the iterative-DCP path broken.

## 2026-06-27 120K-loose MTP3 first attempt

Profile attempted:

- Env file: `glm52-dcp4-mtp3-120k-loose.env`
- `NUM_SPECULATIVE_TOKENS=3`
- `MAX_MODEL_LEN=122880`
- `KV_CACHE_MEMORY_BYTES=1680000000`
- `GPU_MEMORY_UTILIZATION=0.88`

Result:

- Failed during KV sizing, after weights and MTP overlay loaded.
- vLLM reported: 1.57 GiB KV needed, 1.56 GiB available.
- Estimated maximum model length: 122,624 tokens.

Interpretation:

- This is a near-threshold sizing miss, not a structural failure.
- Retrying with `MAX_MODEL_LEN=120000` and the same explicit KV pool.

## 2026-06-27 120K-loose MTP3 result

Profile:

- Env file: `glm52-dcp4-mtp3-120k-loose.env`
- `MAX_MODEL_LEN=120000`
- `KV_CACHE_MEMORY_BYTES=1680000000`
- `GPU_MEMORY_UTILIZATION=0.88`
- `NUM_SPECULATIVE_TOKENS=3`

Startup:

- Passed KV sizing.
- GPU KV cache size: 122,558 tokens.
- Maximum concurrency for 120,000 tokens/request: 1.02x.
- Engine init/profile/cache/warmup: 174.69s, compilation 44.00s.

Client-side completions benchmark:

- Warmup, 128 tokens: 24.930s, 5.134 tok/s.
- Hot codegen 384 tokens #1: 29.090s, 13.200 tok/s.
- Hot codegen 384 tokens #2: 33.177s, 11.574 tok/s.
- Hot codegen 384 tokens #3: 34.921s, 10.996 tok/s.

Server-side behavior:

- Decode windows mostly 10-13 tok/s, with one better 15.7 tok/s window.
- Third-position acceptance was weak again in many windows: examples include 0.036, 0.083, 0.079, 0.063.
- One good window reached third-position acceptance 0.317, but this did not hold.

Interpretation:

- MTP3 is healthy-ish at 16K but degrades again at near-production context/KV sizing.
- This points back to context/KV/DCP-path behavior rather than a universal MTP3 implementation failure.
- Next matched comparison: `MAX_MODEL_LEN=120000`, same explicit KV pool, MTP2.

## 2026-06-27 120K-loose MTP2 result

Profile:

- Env file: `glm52-dcp4-mtp2-120k-loose.env`
- `MAX_MODEL_LEN=120000`
- `KV_CACHE_MEMORY_BYTES=1680000000`
- `GPU_MEMORY_UTILIZATION=0.88`
- `NUM_SPECULATIVE_TOKENS=2`

Startup:

- Passed KV sizing.
- GPU KV cache size: 122,558 tokens.
- Maximum concurrency for 120,000 tokens/request: 1.02x.
- Engine init/profile/cache/warmup: 178.37s, compilation 44.05s.

Client-side completions benchmark:

- Warmup, 128 tokens: 22.102s, 5.791 tok/s.
- Hot codegen 384 tokens #1: 22.814s, 16.832 tok/s.
- Hot codegen 384 tokens #2: 24.739s, 15.522 tok/s.
- Hot codegen 384 tokens #3: 24.166s, 15.890 tok/s.

Server-side behavior:

- Decode windows: 14.2, 17.2, 14.8, 16.8, 17.6 tok/s.
- Second-position acceptance in good windows reached roughly 0.41-0.53.

Interpretation:

- MTP2 is the clear winner over MTP3 at near-production context in this profile.
- MTP3 improves at 16K but regresses at 120K; MTP2 remains healthy.
- Next test: restore the 128K-sized explicit KV pool (`MAX_MODEL_LEN=131072`, `KV_CACHE_MEMORY_BYTES=1810000000`) with MTP2.

## 2026-06-27 128K MTP2 first attempt

Profile attempted:

- Env file: `glm52-dcp4-mtp2-128k.env`
- `MAX_MODEL_LEN=131072`
- `KV_CACHE_MEMORY_BYTES=1810000000`
- `GPU_MEMORY_UTILIZATION=0.915`
- `NUM_SPECULATIVE_TOKENS=2`

Result:

- Failed at vLLM startup free-memory guard before weight loading progressed.
- Root cause: free memory 110.66 GiB, requested utilization target 111.35 GiB.
- This is not a KV capacity failure.

Change:

- Lowering `GPU_MEMORY_UTILIZATION` to `0.90` while keeping explicit KV bytes and max context unchanged.

## 2026-06-27 full 128K MTP2 result

Profile:

- Env file: `glm52-dcp4-mtp2-128k.env`
- `MAX_MODEL_LEN=131072`
- `KV_CACHE_MEMORY_BYTES=1810000000`
- First attempt with `GPU_MEMORY_UTILIZATION=0.915` failed the vLLM startup free-memory guard.
- Retried with `GPU_MEMORY_UTILIZATION=0.90`, explicit KV unchanged.
- `NUM_SPECULATIVE_TOKENS=2`

Startup:

- Passed KV sizing.
- GPU KV cache size: 132,096 tokens.
- Maximum concurrency for 131,072 tokens/request: 1.01x.
- Engine init/profile/cache/warmup: 166.36s, compilation 42.32s.

Client-side completions benchmark:

- Warmup, 128 tokens: 24.288s, 5.270 tok/s.
- Hot codegen 384 tokens #1: 31.271s, 12.280 tok/s.
- Hot codegen 384 tokens #2: 26.689s, 14.388 tok/s.
- Hot codegen 384 tokens #3: 31.636s, 12.138 tok/s.

Server-side behavior:

- Decode windows mostly 11-13 tok/s, with one 16.5 tok/s window.
- Second-position acceptance was weaker than the 120K/MTP2 run, often around 0.13-0.29 and sometimes around 0.43.

Interpretation:

- MTP2 is strong at 120K with the same model/kernel path, but degrades at the full 128K-sized KV pool under current memory pressure.
- Full 128K/MTP2 is not a replacement for the frozen 128K/MTP1 production point based on this run.
- Best new experimental point: 120K/MTP2, 1.68GB explicit KV, `GPU_MEMORY_UTILIZATION=0.88`.
- Best production-safe full-context point remains 128K/MTP1 until MTP2 can hold its acceptance/perf at full capacity.

## 2026-06-27 matched 120K MTP1 baseline

Profile:

- Env file: `glm52-dcp4-mtp1-120k-loose.env`
- Same image/config family as the 120K MTP2/MTP3 tests.
- `MAX_MODEL_LEN=120000`
- `KV_CACHE_MEMORY_BYTES=1680000000`
- `GPU_MEMORY_UTILIZATION=0.88`
- `NUM_SPECULATIVE_TOKENS=1`

Startup:

- Passed KV sizing.
- GPU KV cache size: 122,558 tokens.
- Maximum concurrency for 120,000 tokens/request: 1.02x.
- Engine init/profile/cache/warmup: 193.51s, compilation 44.23s.

Client-side completions benchmark:

- Warmup, 128 tokens: 13.234s, 9.672 tok/s.
- Hot codegen 384 tokens #1: 28.676s, 13.391 tok/s.
- Hot codegen 384 tokens #2: 27.388s, 14.021 tok/s.
- Hot codegen 384 tokens #3: 27.948s, 13.740 tok/s.

Comparison at 120K:

- MTP1 hot: 13.39, 14.02, 13.74 tok/s.
- MTP2 hot: 16.83, 15.52, 15.89 tok/s.
- MTP3 hot: 13.20, 11.57, 11.00 tok/s.

Interpretation:

- MTP2 is a real same-image improvement over MTP1 at 120K.
- MTP3 is worse than MTP1/MTP2 at 120K despite looking acceptable at 16K.

## 2026-06-27 matched 128K MTP1 baseline

Profile:

- Env file: `glm52-dcp4-mtp1-128k.env`
- Same image/config family as the 128K MTP2 test.
- `MAX_MODEL_LEN=131072`
- `KV_CACHE_MEMORY_BYTES=1810000000`
- `GPU_MEMORY_UTILIZATION=0.90`
- `NUM_SPECULATIVE_TOKENS=1`

Startup:

- Passed KV sizing.
- GPU KV cache size: 132,096 tokens.
- Maximum concurrency for 131,072 tokens/request: 1.01x.
- Engine init/profile/cache/warmup: 174.59s, compilation 43.64s.

Client-side completions benchmark:

- Warmup, 128 tokens: 12.725s, 10.059 tok/s.
- Hot codegen 384 tokens #1: 26.863s, 14.295 tok/s.
- Hot codegen 384 tokens #2: 29.045s, 13.221 tok/s.
- Hot codegen 384 tokens #3: 25.741s, 14.918 tok/s.

Matched comparison at 128K:

- MTP1 hot: 14.30, 13.22, 14.92 tok/s.
- MTP2 hot: 12.28, 14.39, 12.14 tok/s.
- Historical MTP3 hot: around 11-13 tok/s, with weak third-position acceptance.

Interpretation:

- At full 128K capacity, MTP1 remains the better production profile.
- MTP2 wins at 120K, but the gain does not survive the full 128K KV/memory-pressure profile.
- MTP3 is not a production candidate at 120K or 128K.

Current best points:

- Full context: `128K / DCP4 / MTP1`, fp8 KV, B12X sparse MLA, FlashInfer/CUTLASS MoE.
- Faster experimental long context: `120K / DCP4 / MTP2`, same stack, `KV_CACHE_MEMORY_BYTES=1680000000`.

## 2026-06-27 loose-memory diagnostic policy

For MTP2/MTP3 diagnosis, do not push the same memory envelope as the 128K production candidate. The 128K/MTP1 point remains the production reference, but iterative-MTP correctness/performance tests should first run with reduced max context, smaller explicit KV allocation, and lower GPU memory utilization so Linux swap/cache pressure and allocator edge behavior do not contaminate acceptance measurements.

Current diagnostic target: hold DCP4/fp8 KV/B12X sparse MLA constant, lower max context to 96K, set explicit KV around 1.4GB/rank, and use lower gpu_memory_utilization. If MTP3 still shows poor third-position acceptance there, the failure is unlikely to be only memory pressure. If it recovers materially, bracket the context/memory inflection before returning to 120K/128K.

### DCP4 / MTP3 / 96K under-pressure run

Settings: max_model_len=98,304, explicit KV=1.4GB/rank, gpu_memory_utilization=0.84, fp8 KV, B12X sparse MLA, flashinfer_cutlass MoE, DCP4, three speculative tokens.

Capacity result: GPU KV cache size 102,144 tokens; maximum concurrency for 98,304 tokens/request 1.04x.

Client short-prompt codegen after warmup: 13.152, 13.366, 12.371 tok/s. This is only a small improvement over the 120K MTP3 row and remains below the useful MTP2/MTP1 candidates. Server acceptance still showed unstable third-position behavior, with representative per-position windows including 0.541/0.197/0.033, 0.600/0.317/0.067, 0.677/0.419/0.210, and one better 0.600/0.433/0.367 window.

Interpretation: lowering context/KV pressure to 96K does not make MTP3 healthy. Memory pressure may contribute variance, but the third recursive draft step remains the dominant failure signature.

### DCP4 / MTP2 / 96K under-pressure run

Settings matched the 96K MTP3 run except NUM_SPECULATIVE_TOKENS=2. Capacity was identical: GPU KV cache size 102,144 tokens; maximum concurrency for 98,304 tokens/request 1.04x.

Client short-prompt codegen after warmup: 13.746, 13.248, 12.199 tok/s. Server-side windows showed second-position acceptance frequently weak/variable, e.g. 0.536/0.145, 0.676/0.338, 0.676/0.397, 0.566/0.250, 0.342/0.145.

Interpretation: the under-pressure 96K setting did not reproduce the strong 120K/MTP2 result. Lowering explicit KV and gpu_memory_utilization alone does not solve iterative-MTP variance; weight-resident memory pressure remains substantial, and the recursive same-layer MTP warning remains active.

### DCP4 / 96K under-pressure matched MTP1/MTP2/MTP3 comparison

All three rows used max_model_len=98,304, explicit KV=1.4GB/rank, gpu_memory_utilization=0.84, DCP4, fp8 KV, B12X sparse MLA, flashinfer_cutlass MoE, and the same 102,144-token KV capacity / 1.04x concurrency.

Client short-prompt codegen after warmup:

| Speculative tokens | Hot samples tok/s | Read |
| --- | --- | --- |
| MTP1 | 14.816, 15.758, 14.328 | best and stable |
| MTP2 | 13.746, 13.248, 12.199 | worse than MTP1 under the same envelope |
| MTP3 | 13.152, 13.366, 12.371 | worse than MTP1; third-position acceptance still unstable |

This closes the lower-memory diagnostic: reducing the KV/context/gpu-util pressure does not make iterative MTP2/MTP3 beat MTP1. The prior 120K/MTP2 win remains a real but non-robust outlier/variant worth preserving for research, not a production replacement for MTP1.

The vLLM runtime warning remains central: this checkpoint exposes one MTP layer, and num_speculative_tokens > 1 recursively forwards through that same layer. The matched 96K comparison supports the architectural conclusion that this GLM-5.2 hybrid checkpoint/runtime is healthy for MTP1, but MTP2/MTP3 are not reliably beneficial without deeper proposer/acceptance changes or a checkpoint with additional trained MTP layers.

## 2026-06-27 completion audit: why MTP1 -> MTP2 -> MTP3 is not achievable by tuning this stack

Objective audited: make MTP1 -> MTP2 -> MTP3 improve, or identify a clear architectural reason it cannot.

Current-state evidence:

- Checkpoint metadata in `/var/tmp/models/Mapika/GLM-5.2-NVFP4-MTP-hybrid/config.json` reports `num_hidden_layers=78` and `num_nextn_predict_layers=1`.
- The safetensors index for the same checkpoint contains `model.layers.78.*` only in `model-mtp.safetensors` and `model-mtp-inputscales.safetensors`; there are no `model.layers.79.*` or `model.layers.80.*` weights.
- Therefore the checkpoint provides exactly one trained next-token prediction layer, not independent heads/layers for MTP2 or MTP3.
- vLLM's speculative config path warns that for non-Step3.5 MTP, `num_speculative_tokens > 1` runs multiple forwards on the same MTP layer and may lower acceptance.
- The active generic proposer path loops over `self.num_speculative_tokens`; iterative MTP2/MTP3 are recursive reuse of the same one-step predictor, not use of additional trained MTP layers.
- The matched 96K under-pressure matrix held DCP4, fp8 KV, B12X sparse MLA, flashinfer_cutlass MoE, max context, explicit KV, and gpu-memory utilization constant. It still produced MTP1 > MTP2/MTP3:
  - MTP1: 14.816, 15.758, 14.328 tok/s.
  - MTP2: 13.746, 13.248, 12.199 tok/s.
  - MTP3: 13.152, 13.366, 12.371 tok/s.
- Server logs from the same matrix show MTP3 third-position acceptance remains unstable even under the looser memory envelope.

Conclusion: for this checkpoint and this vLLM/B12X/DCP4 runtime, monotonic improvement from MTP1 -> MTP2 -> MTP3 is not an environment-tuning target. MTP1 uses the one trained MTP layer in the way the checkpoint actually supports. MTP2/MTP3 recursively reuse that same layer, compounding draft error and paying extra DCP/sparse-index/KV work without reliably amortizing it. Further improvement would require a different architecture-level input: a checkpoint with additional trained next-token prediction layers, or a materially different proposer/acceptance implementation that makes recursive same-layer drafting worthwhile.

## 2026-06-27 live swap check during MTP1/128K

We should not hand-wave swap on this unified-memory system. A live MTP1/128K run was launched with the known-good DCP4 profile, then active Ray GPU worker PIDs were inspected directly before and after a 512-token decode.

Capacity/perf for the run: GPU KV cache size 132,096 tokens; maximum concurrency for 131,072 tokens/request 1.01x. Short benchmark: warmup 128 tokens at 10.144 tok/s; hot 384-token samples at 13.397 and 13.656 tok/s; extra 512-token sample at 13.588 tok/s.

Idle baseline before model launch had no NVIDIA compute PIDs. Existing swapped pages on worker hosts belonged to ordinary system daemons such as dockerd, containerd, polkitd, snapd, fwupd, NetworkManager, and desktop/dashboard services. That part is benign.

During the live model run, each Ray GPU worker itself had swapped pages:

| Host | GPU worker VmSwap before extra decode | GPU worker major faults before | GPU worker VmSwap after | GPU worker major faults after | Delta read |
| --- | ---: | ---: | ---: | ---: | --- |
| relic | 1,948,176 kB | 225,936 | 1,947,668 kB | 226,040 | +104 major faults; pswpin +1,590 pages |
| soulkiller | 1,298,540 kB | 283,189 | 1,653,636 kB | 284,825 | +1,636 major faults; pswpin +2,871 pages; pswpout +81,524 pages |
| cynosure | 1,065,992 kB | 192,172 | 1,065,912 kB | 192,184 | +12 major faults; pswpin +222 pages |
| blackwall | 1,284,752 kB | 231,008 | 1,284,652 kB | 231,024 | +16 major faults; pswpin +197 pages |

Interpretation: this is not only idle Linux junk being swapped. The serving Ray worker processes themselves have ~1.1-1.9GB of swapped address space, and at least soulkiller actively swapped during decode. The magnitude of page-in during a 512-token decode was not enormous in bytes, but the major-fault count and soulkiller's additional swap-out are enough to treat this as a real performance/variance risk.

The original sampler had a bug (`pipefail` plus `head` could terminate after one sample). `kw_swap_diag.sh` has been updated to avoid that failure and to log NVIDIA compute PIDs with `VmSwap` and major-fault counters each sample. Future memory/perf experiments should run the fixed sampler through model load and decode, and should compare GPU-worker major-fault deltas directly rather than only global `SwapFree`.
